#!/usr/bin/env python3
"""
Rename images based on an Excel list (1:1, in order).

USAGE (walk all subfolders, skip first row header, keep only ML-ids):
python rename_images_from_excel.py --img-dir "/path/to/root" --walk --excel-pattern "*.xlsx" --col "A" --skip-rows 1 --name-pattern "^ML\\d+$" --dry-run
-------------------------------------------
python ./rename_images_from_excel.py --img-dir "F:/01_Univalle/01_TG/extra_dataset" --walk --excel-pattern "*.xlsx" --col A --dry-run
---------------------------------------------
      
If your column has a header like 'ID_turdus_ignobilis' in row 1, you can also use --auto-skip-header:
    python rename_images_from_excel.py --img-dir "/path/to/root" --walk --excel-pattern "*.xlsx" --col "A" --auto-skip-header --name-pattern "^ML\\d+$"

WHAT IT DOES
- Reads names from an Excel column (first sheet auto-picked if --sheet omitted).
- Optional header handling: --skip-rows N or --auto-skip-header.
- Optional regex filter: --name-pattern to keep only matching names.
- Collects image files in the target folder (non-recursive), sorted in natural order.
- Renames image i -> "<excel_name_i>.<ext>" (keeping original extension).
- Writes a reversible CSV "rename_mapping.csv" in the same folder.
- Walk mode processes each immediate subfolder of --img-dir.

SAFEGUARDS
- Use --dry-run to preview.
- Use --strict to require images(after offset) == names; otherwise skip folder.
- Use --force to allow overwriting an existing target file.
"""

import argparse
from pathlib import Path
import pandas as pd
import re
import csv
from typing import List, Tuple, Optional, Union

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def collect_images(img_dir: Path) -> List[Path]:
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.stem + p.suffix))

def _read_excel_generic(xlsx_path: Path, sheet: Optional[str]) -> pd.DataFrame:
    if sheet is None:
        obj = pd.read_excel(xlsx_path, sheet_name=None)
        if isinstance(obj, dict):
            if not obj:
                raise ValueError(f"No sheets found in {xlsx_path.name}")
            first_sheet_name = list(obj.keys())[0]
            df = obj[first_sheet_name]
        else:
            df = obj
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return df

def read_names_from_excel(
    xlsx_path: Path,
    sheet: Optional[str],
    col: Optional[str],
    skip_rows: int = 0,
    name_pattern: Optional[str] = None,
    auto_skip_header: bool = False,
    dropna: bool = True
) -> List[str]:
    df = _read_excel_generic(xlsx_path, sheet)
    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {xlsx_path.name}")

    # Allow selecting by letter like 'A' or by header name
    if col is None:
        use_col = df.columns[0]
    else:
        if col in df.columns:
            use_col = col
        else:
            # maybe user passed Excel letter like 'A'/'B'
            if isinstance(col, str) and len(col) == 1 and col.isalpha():
                idx = ord(col.upper()) - ord('A')
                if 0 <= idx < df.shape[1]:
                    use_col = df.columns[idx]
                else:
                    raise ValueError(f"Column '{col}' out of range in {xlsx_path.name}. Available: {list(df.columns)}")
            else:
                raise ValueError(f"Column '{col}' not found in {xlsx_path.name}. Available: {list(df.columns)}")

    series = df[use_col].astype(str).str.strip()

    # Optional header skip by count
    if skip_rows > 0:
        series = series.iloc[skip_rows:]

    # Optional auto header skip: if first value looks like a header (starts with 'ID_' or doesn't match pattern), drop it
    if auto_skip_header and len(series) > 0:
        first = series.iloc[0]
        looks_header = first.startswith("ID_")
        if name_pattern is not None:
            if re.fullmatch(name_pattern, first) is None:
                looks_header = True
        if looks_header:
            series = series.iloc[1:]

    # Optional regex filter for valid names (e.g., '^ML\\d+$')
    if name_pattern is not None:
        series = series[series.apply(lambda s: re.fullmatch(name_pattern, s) is not None)]

    if dropna:
        series = series[series != ""].dropna()

    return series.tolist()

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def plan_operations(images: List[Path], names: List[str], start_index: int=0, prefix: str="", suffix: str="") -> List[Tuple[Path, Path]]:
    pairs = []
    if start_index < 0 or start_index >= len(images):
        raise ValueError(f"start_index {start_index} is out of range for {len(images)} images.")
    usable_images = images[start_index: start_index + len(names)]
    if len(usable_images) < len(names):
        raise ValueError(f"Not enough images: have {len(usable_images)}, need {len(names)}.")
    for img, raw in zip(usable_images, names):
        base = sanitize_filename(f"{prefix}{raw}{suffix}")
        target = img.with_name(base + img.suffix.lower())
        pairs.append((img, target))
    return pairs

def detect_collisions(ops: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path, str]]:
    issues = []
    for src, dst in ops:
        if dst.exists() and src.resolve() != dst.resolve():
            issues.append((src, dst, "target_exists"))
    return issues

def write_mapping_csv(ops: List[Tuple[Path, Path]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["old_path", "new_path"])
        for src, dst in ops:
            w.writerow([str(src), str(dst)])

def process_single_folder(
    excel: Path|None,
    img_dir: Path,
    sheet: Optional[str],
    col: Optional[str],
    start_index: int,
    prefix: str,
    suffix: str,
    dry_run: bool,
    force: bool,
    mapping_csv: Path|None,
    strict: bool,
    skip_rows: int,
    name_pattern: Optional[str],
    auto_skip_header: bool
) -> int:
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory {img_dir} does not exist.")

    if excel is None:
        choices = sorted(list(img_dir.glob("*.xlsx")) + list(img_dir.glob("*.xls")) + list(img_dir.glob("*.xlsm")))
        if not choices:
            print(f"[SKIP] No Excel file found in {img_dir}")
            return 0
        excel = choices[0]

    names = read_names_from_excel(excel, sheet, col, skip_rows=skip_rows, name_pattern=name_pattern, auto_skip_header=auto_skip_header)
    images = collect_images(img_dir)
    if len(images) == 0:
        print(f"[SKIP] No images found in {img_dir}.")
        return 0

    if strict and (len(images) - start_index != len(names)):
        print(f"[STRICT SKIP] {img_dir}: images(after offset)={len(images)-start_index} != names={len(names)}")
        return 0

    ops = plan_operations(images, names, start_index=start_index, prefix=prefix, suffix=suffix)

    issues = detect_collisions(ops)
    if issues and not force:
        print(f"[COLLISION] {img_dir} - use --force to overwrite:")
        for src, dst, why in issues:
            print(f"  - {why}: {src.name} -> {dst.name}")
        print("  -> Skipping this folder.\n")
        return 0

    out_csv = mapping_csv or (img_dir / "rename_mapping.csv")
    write_mapping_csv(ops, out_csv)

    if dry_run:
        print(f"[DRY RUN] {img_dir} planned operations ({len(ops)}):")
        for src, dst in ops[:10]:
            print(f"  {src.name}  =>  {dst.name}")
        if len(ops) > 10:
            print(f"  ... and {len(ops) - 10} more")
        print(f"  Mapping: {out_csv}\n")
        return 0
    else:
        for src, dst in ops:
            if dst.exists() and force and src.resolve() != dst.resolve():
                dst.unlink()
            src.rename(dst)
        print(f"[RENAMED] {img_dir}: {len(ops)} files. Mapping: {out_csv}")
        return len(ops)

def walk_and_process(
    root: Path,
    excel_pattern: str,
    sheet: Optional[str],
    col: Optional[str],
    start_index: int,
    prefix: str,
    suffix: str,
    dry_run: bool,
    force: bool,
    strict: bool,
    skip_rows: int,
    name_pattern: Optional[str],
    auto_skip_header: bool
) -> int:
    total = 0
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root} does not exist.")
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        matches = sorted(list(sub.glob(excel_pattern)))
        excel = matches[0] if matches else None
        try:
            total += process_single_folder(
                excel=excel,
                img_dir=sub,
                sheet=sheet,
                col=col,
                start_index=start_index,
                prefix=prefix,
                suffix=suffix,
                dry_run=dry_run,
                force=force,
                mapping_csv=None,
                strict=strict,
                skip_rows=skip_rows,
                name_pattern=name_pattern,
                auto_skip_header=auto_skip_header
            )
        except Exception as e:
            print(f"[ERROR] {sub}: {e}")
    print(f"\n[SUMMARY] Total files processed: {total}")
    return total

def main():
    ap = argparse.ArgumentParser(description="Rename images based on Excel names (1:1). Single folder or walk mode.")
    ap.add_argument("--excel", type=Path, help="Path to Excel file (single-folder mode). If omitted, first *.xlsx/*.xls/*.xlsm in the folder is used.")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: FIRST sheet automatically)")
    ap.add_argument("--col", default=None, help="Column name or letter (e.g., 'Name' or 'A'). If omitted, first column is used.")
    ap.add_argument("--img-dir", required=True, type=Path, help="Folder containing images (or the root folder when --walk is set)")
    ap.add_argument("--start-index", type=int, default=0, help="Offset into the sorted image list (default: 0)")
    ap.add_argument("--prefix", default="", help="Optional prefix for new names")
    ap.add_argument("--suffix", default="", help="Optional suffix for new names")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without renaming files")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files if necessary (use with care)")
    ap.add_argument("--mapping-csv", type=Path, default=None, help="(Single-folder) Path to write the mapping CSV (default: img-dir/rename_mapping.csv)")
    ap.add_argument("--walk", action="store_true", help="Walk all immediate subfolders of --img-dir and process each")
    ap.add_argument("--excel-pattern", default="*.xlsx", help="Glob pattern to find Excel in each subfolder when using --walk (default: *.xlsx)")

    # New header/validation controls
    ap.add_argument("--skip-rows", type=int, default=0, help="Skip the first N rows from the Excel column (default: 0)")
    ap.add_argument("--name-pattern", default=None, help="Regex to select valid names, e.g., '^ML\\d+$'")
    ap.add_argument("--auto-skip-header", action="store_true", help="Auto-drop the first cell if it looks like a header (e.g., starts with 'ID_' or doesn't match --name-pattern)")

    # Strictness toggle
    ap.add_argument("--strict", action="store_true", help="Require images(after offset) == names; otherwise skip folder")

    args = ap.parse_args()

    if args.walk:
        walk_and_process(
            root=args.img_dir,
            excel_pattern=args.excel_pattern,
            sheet=args.sheet,
            col=args.col,
            start_index=args.start_index,
            prefix=args.prefix,
            suffix=args.suffix,
            dry_run=args.dry_run,
            force=args.force,
            strict=args.strict,
            skip_rows=args.skip_rows,
            name_pattern=args.name_pattern,
            auto_skip_header=args.auto_skip_header
        )
    else:
        processed = process_single_folder(
            excel=args.excel,
            img_dir=args.img_dir,
            sheet=args.sheet,
            col=args.col,
            start_index=args.start_index,
            prefix=args.prefix,
            suffix=args.suffix,
            dry_run=args.dry_run,
            force=args.force,
            mapping_csv=args.mapping_csv,
            strict=args.strict,
            skip_rows=args.skip_rows,
            name_pattern=args.name_pattern,
            auto_skip_header=args.auto_skip_header
        )
        if args.dry_run:
            print(f"[DRY RUN] Summary: {processed} files would be renamed.")
        else:
            print(f"[DONE] Renamed {processed} files.")

if __name__ == "__main__":
    main()
