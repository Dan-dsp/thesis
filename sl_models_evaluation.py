"""
eval_sl_models.py

Load saved models from models_sl/, evaluate them on the test split,
generate metrics, confusion matrices, PCA plots, and a comparison table.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D PCA plot

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

METADATA_COLUMNS = {"sample_id", "sample_name", "orig_filename", "species", "split"}
DUPLICATE_DESCRIPTIVE_FEATURES = {"affine_6"}
DUPLICATE_LEGACY_FEATURES = {"f39"}



USE_ROC_AUC = True  # you can set False if it becomes too heavy

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    legacy_f_cols = [
        c for c in numeric_cols
        if c.startswith("f") and c not in DUPLICATE_LEGACY_FEATURES
    ]
    descriptive_cols = [
        c for c in numeric_cols
        if not c.startswith("f") and c not in DUPLICATE_DESCRIPTIVE_FEATURES
    ]

    if descriptive_cols:
        return descriptive_cols
    return legacy_f_cols


def load_dataset(data_csv_path: Path):
    """
    Load feature dataset from CSV and return:
    - X_test: np.ndarray, test features
    - y_test: np.ndarray, test labels
    - classes: np.ndarray, sorted unique class names
    - X_scaled_for_pca: np.ndarray, scaled test features for PCA
    """
    df = pd.read_csv(data_csv_path)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["species"].values
    splits = df["split"].values

    # Select only test split
    X_test = X[splits == "test"]
    y_test = y[splits == "test"]

    print(f"[INFO] Test data size: {X_test.shape}")

    # For PCA visualization only
    scaler_for_pca = StandardScaler()
    X_scaled_for_pca = scaler_for_pca.fit_transform(X_test)

    classes = np.unique(y_test)

    return X_test, y_test, classes, X_scaled_for_pca

# df = pd.read_csv(DATA_CSV_PATH)

# # Feature columns are all columns that start with "f"
# feature_cols = [c for c in df.columns if c.startswith("f")]
# X = df[feature_cols].values
# y = df["species"].values
# splits = df["split"].values

# # We evaluate only on the test split
# X_test = X[splits == "test"]
# y_test = y[splits == "test"]

# print(f"[INFO] Test data size: {X_test.shape}")

# # For PCA visualization, scaled features to have zero mean & unit variance.
# scaler_for_pca = StandardScaler()
# X_scaled_for_pca = scaler_for_pca.fit_transform(X_test)

# -------------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------------
def load_models(models_dir: Path):
    """
    Load all available shallow-learning models from models_dir.
    Returns a dict: {model_name: model_instance}.
    """
    model_files = {
        "svm_rbf":       models_dir / "svm_rbf.pkl",
        "random_forest": models_dir / "random_forest.pkl",
        "knn":           models_dir / "knn.pkl",
        "logreg":        models_dir / "logreg.pkl",
    }

    models = {}
    for name, path in model_files.items():
        if not path.exists():
            print(f"[WARN] Model file not found: {path}, skipping.")
            continue
        models[name] = joblib.load(path)
        print(f"[INFO] Loaded model: {name} from {path}")

    if not models:
        raise RuntimeError("No models loaded. Check paths and training step.")

    return models

# model_files = {
#     "svm_rbf":          MODELS_DIR / "svm_rbf.pkl",
#     "random_forest":    MODELS_DIR / "random_forest.pkl",
#     "knn":              MODELS_DIR / "knn.pkl",
#     "logreg":           MODELS_DIR / "logreg.pkl",
# }

# models = {}
# for name, path in model_files.items():
#     if not path.exists():
#         print(f"[WARN] Model file not found: {path}, skipping.")
#         continue
#     models[name] = joblib.load(path)
#     print(f"[INFO] Loaded model: {name} from {path}")

# if not models:
#     raise RuntimeError("No models loaded. Check paths and training step.")

# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------
def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    y_test_bin: np.ndarray | None,
    use_roc_auc: bool = True,
):
    """
    Evaluate a single model on the test set.

    Returns:
    - metrics: dict with accuracy, precision_macro, recall_macro, f1_macro, roc_auc_macro
    - cm: confusion matrix (np.ndarray)
    """
    print(f"\n[INFO] Evaluating {name} ...")

    # 1. Predictions
    y_pred = model.predict(X_test)

    # 2. Macro metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # 3. ROC AUC (optional)
    auc_macro = np.nan
    if use_roc_auc and hasattr(model, "predict_proba") and y_test_bin is not None:
        try:
            y_prob = model.predict_proba(X_test)
            auc_macro = roc_auc_score(
                y_test_bin, y_prob, average="macro", multi_class="ovr"
            )
        except Exception as e:
            print(f"[WARN] Could not compute ROC AUC for {name}: {e}")
            auc_macro = np.nan

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # 5. Print summary + detailed report
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1-score:  {f1:.4f} (macro)")
    if use_roc_auc:
        print(f"  ROC AUC:   {auc_macro:.4f} (macro, OVR)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "model":           name,
        "accuracy":        acc,
        "precision_macro": prec,
        "recall_macro":    rec,
        "f1_macro":        f1,
        "roc_auc_macro":   auc_macro if use_roc_auc else np.nan,
    }

    return metrics, cm

# results = []

# # Prepare for ROC AUC if needed
# # For ROC AUC, we need a one-vs-rest binarized version of y
# classes = np.unique(y_test)
# y_test_bin = label_binarize(y_test, classes=classes) if USE_ROC_AUC else None

# for name, model in models.items():
#     print(f"\n[INFO] Evaluating {name} ...")
#     y_pred = model.predict(X_test)

#     # Basic metrics (macro averages across classes)
#     acc = accuracy_score(y_test, y_pred)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_test, y_pred, average="macro", zero_division=0
#     )

#     # ROC AUC (macro) if model supports probability
#     auc_macro = np.nan
#     if USE_ROC_AUC and hasattr(model, "predict_proba"):
#         try:
#             y_prob = model.predict_proba(X_test)
#             auc_macro = roc_auc_score(
#                 y_test_bin, y_prob, average="macro", multi_class="ovr"
#             )
#         except Exception as e:
#             print(f"[WARN] Could not compute ROC AUC for {name}: {e}")
#             auc_macro = np.nan

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred, labels=classes)

#     # Print quick summary and classification report
#     print(f"  Accuracy:  {acc:.4f}")
#     print(f"  Precision: {prec:.4f} (macro)")
#     print(f"  Recall:    {rec:.4f} (macro)")
#     print(f"  F1-score:  {f1:.4f} (macro)")
#     if USE_ROC_AUC:
#         print(f"  ROC AUC:   {auc_macro:.4f} (macro, OVR)")
#     print("\nClassification report:")
#     print(classification_report(y_test, y_pred, zero_division=0))

#     # Store results for comparison table
#     results.append({
#         "model": name,
#         "accuracy": acc,
#         "precision_macro": prec,
#         "recall_macro": rec,
#         "f1_macro": f1,
#         "roc_auc_macro": auc_macro if USE_ROC_AUC else np.nan,
#     })

    # Plot confusion matrix
    # fig, ax = plt.subplots(figsize=(6, 5))
    # im = ax.imshow(cm, interpolation="nearest")
    # ax.set_title(f"Confusion Matrix - {name}")
    # plt.colorbar(im, ax=ax)
    # ax.set_xticks(np.arange(len(classes)))
    # ax.set_yticks(np.arange(len(classes)))
    # ax.set_xticklabels(classes, rotation=90)
    # ax.set_yticklabels(classes)
    # ax.set_xlabel("Predicted")
    # ax.set_ylabel("True")

    # # annotate each cell with its numeric value
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, cm[i, j],
    #                 ha="center", va="center", fontsize=7, color="white" if cm[i,j] > cm.max()/2 else "black")

    # plt.tight_layout()
    # plt.show()

def plot_confusion_matrix(cm: np.ndarray, classes: np.ndarray, title: str):
    """
    Plot a confusion matrix with counts in each cell.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # annotate cells
    max_val = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > max_val / 2 else "black"
            ax.text(j, i, val, ha="center", va="center", fontsize=7, color=color)
 
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# COMPARISON TABLE
# -------------------------------------------------------------------
def save_comparison_table(results: list[dict], models_dir: Path):
    """
    Save the list of metrics dicts as a CSV comparison table.
    """
    df_results = pd.DataFrame(results)
    print("\n[RESULTS] Model comparison table:\n")
    print(df_results)

    out_csv = models_dir / "model_comparison.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved comparison table to {out_csv}")


# df_results = pd.DataFrame(results)
# print("\n[RESULTS] Model comparison table:\n")
# print(df_results)

# df_results.to_csv("F:/01_Univalle/01_TG/sl_results/model_comparison.csv", index=False)
# print("\n[INFO] Saved comparison table to sl_results/model_comparison.csv")

# -------------------------------------------------------------------
# PCA VISUALIZATIONS (2D and 3D)
# -------------------------------------------------------------------
def run_pca_plots(X_scaled_for_pca: np.ndarray, y_test: np.ndarray, classes: np.ndarray):
    """
    Compute 2D and 3D PCA on the scaled test features and plot them.
    """

    print("\n[INFO] Computing PCA (2D and 3D) on test features ...")

    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled_for_pca)

    plt.figure(figsize=(7, 6))
    for cls in classes:
        idx = (y_test == cls)
        plt.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1], label=cls, s=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D - Test Features (shallow learning)")
    plt.legend(markerscale=2, fontsize=7)
    plt.tight_layout()
    plt.show()

    # 3D PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled_for_pca)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cls in classes:
        idx = (y_test == cls)
        ax.scatter(
            X_pca_3d[idx, 0],
            X_pca_3d[idx, 1],
            X_pca_3d[idx, 2],
            label=cls,
            s=15,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D - Test Features (shallow learning)")
    ax.legend(markerscale=2, fontsize=7)
    plt.tight_layout()
    plt.show()


# print("\n[INFO] Computing PCA (2D and 3D) on test features ...")

# # 2D PCA
# pca_2d = PCA(n_components=2)
# X_pca_2d = pca_2d.fit_transform(X_scaled_for_pca)

# plt.figure(figsize=(7, 6))
# for cls in classes:
#     idx = (y_test == cls)
#     plt.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1], label=cls, s=15)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA 2D - Test Features (shallow learning)")
# plt.legend(markerscale=2, fontsize=7)
# plt.tight_layout()
# plt.show()

# # 3D PCA
# pca_3d = PCA(n_components=3)
# X_pca_3d = pca_3d.fit_transform(X_scaled_for_pca)

# fig = plt.figure(figsize=(7, 6))
# ax = fig.add_subplot(111, projection="3d")
# for cls in classes:
#     idx = (y_test == cls)
#     ax.scatter(X_pca_3d[idx, 0], X_pca_3d[idx, 1], X_pca_3d[idx, 2], label=cls, s=15)
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.set_title("PCA 3D - Test Features (shallow learning)")
# ax.legend(markerscale=2, fontsize=7)
# plt.tight_layout()
# plt.show()
