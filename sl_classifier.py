"""
train_sl_models.py

Train 4 classical ML models on the shallow-learning dataset and save them to disk:
- SVM (RBF)
- Random Forest
- KNN
- Logistic Regression

The data is expected to come from shallow_learning_birds.csv with columns:
    sample_id, sample_name, orig_filename, species, split, and either:
    - descriptive feature names such as hu_1, glcm_contrast, ...
    - or legacy aliases f0, f1, ..., fN
"""

import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

METADATA_COLUMNS = {"sample_id", "sample_name", "orig_filename", "species", "split"}
DUPLICATE_DESCRIPTIVE_FEATURES = {"affine_6"}
DUPLICATE_LEGACY_FEATURES = {"f39"}


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

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_CSV_PATH = r"F:/Univalle/01_TG/shallow_learning_birds.csv"
MODELS_DIR = Path("models_sl")

# Toggle to use / not use StandardScaler for all models that need it.
USE_STANDARD_SCALER = True  # <--- change to False if you want to disable scaling

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = pd.read_csv(DATA_CSV_PATH)

# Features are descriptive columns when available; otherwise fall back to legacy f-columns.
feature_cols = get_feature_columns(df)
X = df[feature_cols].values
y = df["species"].values
splits = df["split"].values

# We trust your split column, so:
X_train = X[(splits == "train") | (splits == "val")]  # train+val
y_train = y[(splits == "train") | (splits == "val")]

# We will NOT touch the test split here; it is for eval later.
print(f"[INFO] Training data size: {X_train.shape}, classes: {len(set(y_train))}")

# -------------------------------------------------------------------
# PREPARE MODELS
# -------------------------------------------------------------------
scaler_step = ("scaler", StandardScaler()) if USE_STANDARD_SCALER else None

def make_pipeline(clf):
    """
    Helper to build a sklearn Pipeline with optional StandardScaler.
    """
    steps = []
    if scaler_step is not None:
        steps.append(scaler_step)
    steps.append(("clf", clf))
    return Pipeline(steps)

models = {
    "svm_rbf": make_pipeline(
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    ),
    "random_forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )
    ),
    "knn": make_pipeline(
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski"
        )
    ),
    "logreg": make_pipeline(
        LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            n_jobs=-1
        )
    ),
}

# -------------------------------------------------------------------
# TRAIN AND SAVE MODELS
# -------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    print(f"[INFO] Training {name} ...")
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Saved {name} to {model_path}")

print("[INFO] All models trained and saved.")
