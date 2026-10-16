"""Descriptive error inspection matched by image ID, independent of rejection."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_image_predictions(path, family, model, metadata, truth, predictions):
    rows = metadata.reset_index(drop=True).copy()
    if len(rows) != len(truth) or len(rows) != len(predictions):
        raise ValueError("Image metadata and prediction lengths must match.")
    rows["family"], rows["model"] = family, model
    rows["true_label"] = np.asarray(truth).astype(str)
    rows["predicted_label"] = np.asarray(predictions).astype(str)
    rows["correct"] = rows.true_label.eq(rows.predicted_label)
    if "sample_key" not in rows or rows.sample_key.duplicated().any():
        raise ValueError("Unique sample_key required for every image prediction.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    rows.loc[~rows.correct].to_csv(path.with_name(
        path.stem.removesuffix("_test_predictions") + "_test_errors.csv"), index=False)
    return path


def _image_path(group, image_root):
    if "image_path" in group:
        for value in group.image_path.dropna():
            path = Path(str(value))
            if path.is_file():
                return str(path.resolve())
    if image_root is not None:
        root = Path(image_root).resolve()
        path = (root / str(group.sample_key.iloc[0])).resolve()
        if path.is_relative_to(root) and path.is_file():
            return str(path)
    return ""


def _write_gallery(joined, overview, out_dir):
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for row in overview.loc[overview.error_count.gt(0)].itertuples(index=False):
        predictions = joined.loc[joined.sample_key.eq(row.sample_key)].sort_values("evaluator")
        image = (f'<a href="{escape(Path(row.image_path).as_uri())}"><img loading="lazy" '
                 f'src="{escape(Path(row.image_path).as_uri())}" alt="{escape(row.sample_key)}"></a>'
                 if row.image_path else '<p class="missing">Image file unavailable</p>')
        results = "".join(
            f'<tr class="{"correct" if p.correct else "wrong"}"><td>{escape(p.evaluator)}</td>'
            f'<td>{escape(p.predicted_label)}</td><td>{"Correct" if p.correct else "Error"}</td></tr>'
            for p in predictions.itertuples(index=False))
        cards.append(f'<article><h2>{escape(row.sample_key)}</h2>{image}'
                     f'<p>True species: <strong>{escape(row.true_label)}</strong> | '
                     f'{escape(row.error_group)} | {row.error_count}/{row.evaluations} errors</p>'
                     f'<table><thead><tr><th>Classifier</th><th>Prediction</th><th>Result</th></tr>'
                     f'</thead><tbody>{results}</tbody></table></article>')
    content = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Image Error Review</title>
<style>body{font:14px Arial,sans-serif;margin:24px;color:#222;background:#fff}
h1{font-size:24px}h2{font-size:16px;overflow-wrap:anywhere}main{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px}
article{border-top:2px solid #555;padding:12px 0;min-width:0}img{width:100%;height:260px;
object-fit:contain;background:#f3f3f3}table{width:100%;border-collapse:collapse;table-layout:fixed}
td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}
.wrong{color:#a02035}.correct{color:#176958}.missing{height:100px;background:#eee;padding:16px}
</style><h1>Image Error Review</h1>
<p>Raw species predictions, without rejection. Shared-image errors are descriptive,
not a pooled performance ranking. Only images with at least one error appear below.</p><main>'''
    (out_dir / "error_gallery.html").write_text(
        content + "".join(cards) + "</main></html>", encoding="utf-8")


def _plot_errors(matrix, overview, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    selected = overview.loc[overview.error_count.gt(0), "sample_key"].head(60)
    if selected.empty:
        fig = Figure(figsize=(8, 3))
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.5, "No classification errors in the provided image predictions.",
                 ha="center", va="center")
        fig.savefig(path, dpi=160)
        fig.clear()
        return
    values = matrix.reindex(selected)
    fig = Figure(figsize=(max(8, 1.2 * len(values.columns) + 5),
                          max(4, 0.25 * len(values) + 3)), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    cmap = ListedColormap(["#168575", "#c13f57"])
    cmap.set_bad("#dddddd")
    ax.imshow(values.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(values.columns)), values.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(values)), values.index, fontsize=7)
    ax.set_title("Image errors: red = wrong, green = correct, gray = not evaluated\n"
                 "Up to 60 images with most errors; complete results in CSV/gallery")
    fig.savefig(path, dpi=160)
    fig.clear()


def summarize_image_errors(paths, out_dir, annotations=None, image_root=None):
    """Match raw predictions by exact sample_key; never infer bird identities or causes."""
    if not paths:
        raise ValueError("Provide at least one prediction file.")
    tables = []
    seen_sources = set()
    for path in paths:
        source = str(Path(path).resolve())
        if source in seen_sources:
            raise ValueError(f"Duplicate prediction source: {path}")
        seen_sources.add(source)
        table = pd.read_csv(path, dtype={"sample_key": str, "family": str, "model": str,
                                         "true_label": str, "predicted_label": str})
        required = {"sample_key", "family", "model", "true_label", "predicted_label"}
        if not required.issubset(table) or table.empty or table[list(required)].isna().any().any():
            raise ValueError(f"Missing or empty image prediction data in {path}")
        if not table.family.isin(["SL", "DL"]).all():
            raise ValueError("Prediction family must be SL or DL.")
        if table.duplicated(["family", "model", "sample_key"]).any():
            raise ValueError(f"Duplicate image predictions in {path}")
        table["correct"] = table.true_label.eq(table.predicted_label)
        table["source"] = source
        # Keep identical model names from different runs distinct.
        tag = hashlib.sha256(source.encode()).hexdigest()[:8]
        table["evaluator"] = table.family + "/" + table.model + "@" + tag
        tables.append(table)
    joined = pd.concat(tables, ignore_index=True)
    if joined.groupby("sample_key").true_label.nunique().gt(1).any():
        raise ValueError("Conflicting true labels for the same sample_key; verify image correspondence.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotation = pd.DataFrame({"sample_key": sorted(joined.sample_key.unique()),
                               "conditions": "", "notes": ""})
    # Never erase a template the user may already have filled in.
    if not (out_dir / "annotation_template.csv").exists():
        annotation.to_csv(out_dir / "annotation_template.csv", index=False)
    if annotations is not None:
        annotated = pd.read_csv(annotations, dtype={"sample_key": str})
        if "sample_key" not in annotated or annotated.sample_key.isna().any() or annotated.sample_key.duplicated().any():
            raise ValueError("Annotation sample_key must be present and unique.")
        columns = [c for c in ["sample_key", "conditions", "notes", "bird_id"] if c in annotated]
        joined = joined.drop(columns=[c for c in columns if c != "sample_key" and c in joined])
        joined = joined.merge(annotated[columns], on="sample_key", how="left", validate="many_to_one")
    joined.to_csv(out_dir / "all_predictions.csv", index=False)
    joined.loc[~joined.correct].to_csv(out_dir / "error_review.csv", index=False)
    species = joined.groupby(["family", "model", "source", "true_label"]).agg(
        samples=("correct", "size"), errors=("correct", lambda x: int((~x).sum())),
        accuracy=("correct", "mean"))
    species["error_rate"] = 1 - species.accuracy
    species.to_csv(out_dir / "species_error_summary.csv")
    joined.loc[~joined.correct].groupby(
        ["family", "model", "source", "true_label", "predicted_label"]).size().rename(
            "errors").to_csv(out_dir / "confusion_pairs.csv")
    sources = joined[["evaluator", "family", "model", "source"]].drop_duplicates()
    sources.to_csv(out_dir / "evaluator_sources.csv", index=False)
    overview_rows = []
    for key, group in joined.groupby("sample_key", sort=True):
        sl = group.loc[group.family.eq("SL")]
        dl = group.loc[group.family.eq("DL")]
        sl_errors, dl_errors = int((~sl.correct).sum()), int((~dl.correct).sum())
        matched = len(sl) > 0 and len(dl) > 0
        category = ("not_evaluated_by_both_families" if not matched else
                    "both_families_error" if sl_errors and dl_errors else
                    "sl_only_error" if sl_errors else "dl_only_error" if dl_errors else
                    "no_errors")
        overview_rows.append({"sample_key": key, "true_label": group.true_label.iloc[0],
                              "image_path": _image_path(group, image_root),
                              "evaluations": len(group), "correct_count": int(group.correct.sum()),
                              "error_count": int((~group.correct).sum()),
                              "sl_evaluations": len(sl), "sl_errors": sl_errors,
                              "dl_evaluations": len(dl), "dl_errors": dl_errors,
                              "evaluated_by_both_families": matched, "error_group": category})
    overview = pd.DataFrame(overview_rows).sort_values(
        ["error_count", "sample_key"], ascending=[False, True])
    overview.to_csv(out_dir / "image_error_frequency.csv", index=False)
    for name in ["both_families_error", "sl_only_error", "dl_only_error"]:
        overview.loc[overview.error_group.eq(name)].to_csv(out_dir / f"{name}.csv", index=False)
    overview.loc[~overview.evaluated_by_both_families].to_csv(
        out_dir / "unmatched_images.csv", index=False)
    matrix = joined.assign(error=(~joined.correct).astype(int)).pivot(
        index="sample_key", columns="evaluator", values="error")
    matrix.reindex(overview.sample_key).to_csv(out_dir / "image_error_matrix.csv")
    joined.pivot(index="sample_key", columns="evaluator", values="predicted_label").to_csv(
        out_dir / "image_prediction_matrix.csv")
    overlap = []
    evaluators = list(sources.evaluator)
    for i, left in enumerate(evaluators):
        a = joined.loc[joined.evaluator.eq(left)].set_index("sample_key")
        for right in evaluators[i + 1:]:
            b = joined.loc[joined.evaluator.eq(right)].set_index("sample_key")
            common = a.index.intersection(b.index)
            x, y = a.loc[common], b.loc[common]
            both = ~x.correct & ~y.correct
            overlap.append({"left_evaluator": left, "right_evaluator": right,
                            "left_source": a.source.iloc[0], "right_source": b.source.iloc[0],
                            "common_images": len(common), "left_only_images": len(a) - len(common),
                            "right_only_images": len(b) - len(common),
                            "same_image_errors": int(both.sum()),
                            "both_correct": int((x.correct & y.correct).sum()),
                            "left_wrong_right_correct": int((~x.correct & y.correct).sum()),
                            "left_correct_right_wrong": int((x.correct & ~y.correct).sum()),
                            "same_wrong_species": int((both & x.predicted_label.eq(y.predicted_label)).sum())})
    pd.DataFrame(overlap, columns=["left_evaluator", "right_evaluator", "left_source", "right_source",
                                  "common_images", "left_only_images", "right_only_images",
                                  "same_image_errors", "both_correct", "left_wrong_right_correct",
                                  "left_correct_right_wrong", "same_wrong_species"]).to_csv(
                                      out_dir / "descriptive_error_overlap.csv", index=False)
    if "conditions" in joined:
        tagged = joined.dropna(subset=["conditions"]).copy()
        tagged["condition"] = tagged.conditions.astype(str).str.split(";")
        tagged = tagged.explode("condition")
        tagged["condition"] = tagged.condition.str.strip()
        tagged = tagged.loc[tagged.condition.ne("")].drop_duplicates(["evaluator", "sample_key", "condition"])
        condition = tagged.groupby(["evaluator", "family", "model", "condition"]).agg(
            samples=("correct", "size"), errors=("correct", lambda x: int((~x).sum())),
            accuracy=("correct", "mean"))
        condition["error_rate"] = 1 - condition.accuracy
        condition.to_csv(out_dir / "condition_summary.csv")
    else:
        pd.DataFrame(columns=["evaluator", "family", "model", "condition", "samples",
                              "errors", "accuracy", "error_rate"]).to_csv(
                                  out_dir / "condition_summary.csv", index=False)
    # Retain compatibility for older callers, but no individual ID is requested.
    if "bird_id" in joined:
        identified = joined.dropna(subset=["bird_id"])
        identified = identified.loc[identified.bird_id.astype(str).str.strip().ne("")]
        identified.groupby("bird_id").agg(evaluations=("correct", "size"),
                                          accuracy=("correct", "mean")).to_csv(out_dir / "bird_error_summary.csv")
    _plot_errors(matrix, overview, out_dir / "image_errors.png")
    _write_gallery(joined, overview, out_dir)
    summary = {"images": len(overview), "evaluators": len(sources),
               "images_evaluated_by_both_families": int(overview.evaluated_by_both_families.sum()),
               "images_with_errors_in_both_families": int(overview.error_group.eq("both_families_error").sum()),
               "images_missing_files": int(overview.image_path.eq("").sum()),
               "definition": "Family error means at least one evaluated classifier in that family was wrong.",
               "scope": "Descriptive shared-image inspection, not a pooled ranking or causal analysis.",
               "rejection_used": False}
    (out_dir / "error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return overview
