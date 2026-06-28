"""
train_model.py
===============
Trains and compares two models for synthetic identity fraud detection:

1. LOGISTIC REGRESSION on WoE-transformed features (a scorecard, in the
   classic credit-risk sense) -- interpretable, auditable, regulator-
   friendly.
2. HistGradientBoostingClassifier on raw/lightly-encoded features -- a
   higher-capacity "challenger" model that can pick up non-linear
   interactions (e.g. between graph features and attribute features)
   that the linear scorecard cannot.

NOTE ON XGBOOST SUBSTITUTION
-----------------------------
This environment has no internet access, so `pip install xgboost` could
not run and the true XGBoost library is unavailable here. We use
scikit-learn's HistGradientBoostingClassifier instead -- the same
algorithmic family (histogram-based gradient boosted decision trees)
with a very similar API and performance profile. On a machine with
normal pip access, swap in:

    from xgboost import XGBClassifier
    tree_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=<neg/pos ratio>, eval_metric="aucpr",
        random_state=42,
    )

...as a drop-in replacement for the HistGradientBoostingClassifier below;
the rest of the pipeline (train/test split, evaluation, SHAP) is written
to work with either.

LEAKAGE-SAFE TRAIN/TEST SPLIT (BY COMPONENT, NOT BY ROW)
-----------------------------------------------------------
Two applications that share an identity element (same ring, same legit
household) live in the SAME connected component of the identity graph.
If we split rows into train/test independently of this structure, two
siblings from the same ring/household can end up on opposite sides of
the split. The graph features computed for the training-side sibling
(e.g. "my component has 6 applications") would then implicitly encode
information about a specific test-side application's existence and
label correlation -- a real, if subtle, form of train/test leakage that
is easy to miss with graph features specifically (it doesn't happen with
plain tabular features). We therefore split at the COMPONENT level:
every application in a given connected component goes entirely to
train or entirely to test, never split across both.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import (
    build_woe_feature_matrix, apply_fitted_woe_tables, NUMERIC_FEATURES_FOR_BINNING,
    BOOLEAN_FEATURES, CATEGORICAL_FEATURES, TARGET_COL,
)

RANDOM_STATE = 42


def component_level_train_test_split(df: pd.DataFrame, test_size: float = 0.25,
                                      random_state: int = RANDOM_STATE):
    """
    Splits the dataset by connected component, not by row, so that no
    component (ring or legit cluster) is split across train and test.
    Stratifies the split by each component's fraud status (a component is
    "fraud" if ANY member is fraud) to keep the train/test fraud rate
    reasonably balanced despite splitting at the coarser component level.
    """
    component_labels = (
        df.groupby("component_id")[TARGET_COL].max().rename("component_is_fraud")
    )
    components = component_labels.index.values
    comp_is_fraud = component_labels.values

    train_components, test_components = train_test_split(
        components, test_size=test_size, random_state=random_state, stratify=comp_is_fraud
    )

    train_df = df[df["component_id"].isin(train_components)].reset_index(drop=True)
    test_df = df[df["component_id"].isin(test_components)].reset_index(drop=True)
    return train_df, test_df


def prepare_tree_feature_matrix(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Builds the raw/lightly-encoded feature matrix for the tree model.
    Unlike the WoE matrix, trees don't need binning -- they find their
    own splits -- so we keep numeric features as-is and only ordinal-
    encode the categorical email_domain column (HistGradientBoosting
    handles ordinal-encoded categoricals fine since it splits on
    thresholds, not on the encoding's implied ordering).
    """
    feature_cols = NUMERIC_FEATURES_FOR_BINNING + BOOLEAN_FEATURES
    X = df[feature_cols].copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if fit:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[col] = enc.fit_transform(df[[col]])
            encoders[col] = enc
        else:
            enc = encoders[col]
            X[col] = enc.transform(df[[col]])

    return X, encoders


def train_logistic_scorecard(train_df: pd.DataFrame, test_df: pd.DataFrame, min_iv: float = 0.02):
    """
    Trains the WoE/IV-based logistic regression scorecard, and applies
    the SAME fitted WoE tables (fit on train only) to the test set, so
    the returned bundle is ready for immediate evaluation without the
    caller needing to repeat any fitting logic.
    """
    X_train_woe, woe_tables, iv_summary = build_woe_feature_matrix(train_df, min_iv=min_iv)
    y_train = train_df[TARGET_COL].values

    model = LogisticRegression(
        class_weight="balanced",  # see README for tradeoff vs. SMOTE
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_woe, y_train)

    X_test_woe = apply_fitted_woe_tables(test_df, woe_tables)
    y_test = test_df[TARGET_COL].values

    return {
        "model": model,
        "X_train": X_train_woe, "y_train": y_train,
        "X_test": X_test_woe, "y_test": y_test,
        "woe_tables": woe_tables, "iv_summary": iv_summary,
        "feature_names": X_train_woe.columns.tolist(),
    }


def train_tree_challenger(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains the HistGradientBoostingClassifier (XGBoost stand-in) and
    applies the SAME fitted ordinal encoder to the test set, returning a
    bundle ready for immediate evaluation.
    """
    X_train, encoders = prepare_tree_feature_matrix(train_df, fit=True)
    y_train = train_df[TARGET_COL].values

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    # HistGradientBoostingClassifier doesn't take scale_pos_weight directly;
    # use sample_weight instead to achieve the same class-imbalance correction
    # XGBoost's scale_pos_weight would provide.
    sample_weight = np.where(y_train == 1, n_neg / n_pos, 1.0)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    X_test, _ = prepare_tree_feature_matrix(test_df, encoders=encoders, fit=False)
    y_test = test_df[TARGET_COL].values

    return {
        "model": model,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "encoders": encoders,
        "feature_names": X_train.columns.tolist(),
    }


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/synthetic_identity_fraud/data/applications_with_graph_features.csv")
    train_df, test_df = component_level_train_test_split(df)

    print(f"Train: {len(train_df)} apps ({train_df[TARGET_COL].mean():.4f} fraud rate)")
    print(f"Test:  {len(test_df)} apps ({test_df[TARGET_COL].mean():.4f} fraud rate)")

    # Sanity check: no component should appear in both splits
    overlap = set(train_df["component_id"]) & set(test_df["component_id"])
    print(f"Component overlap between train/test (should be 0): {len(overlap)}")

    logit_bundle = train_logistic_scorecard(train_df, test_df)
    print(f"\nLogistic scorecard trained on {len(logit_bundle['feature_names'])} WoE features (IV >= 0.02).")
    print(f"Features: {logit_bundle['feature_names']}")

    tree_bundle = train_tree_challenger(train_df, test_df)
    print(f"\nTree challenger trained on {len(tree_bundle['feature_names'])} raw features.")

    train_df.to_csv("/home/claude/synthetic_identity_fraud/data/train.csv", index=False)
    test_df.to_csv("/home/claude/synthetic_identity_fraud/data/test.csv", index=False)
    print("\nSaved train/test splits.")
