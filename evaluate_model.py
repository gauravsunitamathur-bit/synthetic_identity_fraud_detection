"""
evaluate_model.py
===================
Evaluates and compares the logistic scorecard vs. the tree challenger
model the way a fraud risk team actually reviews model performance:

- AUC-PR as the PRIMARY metric (not ROC-AUC), since at a ~2% fraud rate
  ROC-AUC is easily inflated by the model's ability to correctly rank the
  huge majority of true negatives, which isn't the hard part of this
  problem. AUC-PR focuses on precision/recall tradeoffs in exactly the
  regime that matters for a rare-event detection task.
- A confusion matrix evaluated at a BUSINESS-relevant threshold (not the
  default 0.5, which is meaningless for a 2% base rate) -- specifically,
  the threshold that flags the top X% of applications for manual review,
  matching how a real fraud ops queue is sized and staffed.
- A score-band / decile table: applications are bucketed into risk
  deciles by predicted score, and we report the fraud rate observed in
  each decile. This is the artifact a risk team actually uses to decide
  cutoffs and to communicate model lift to non-technical stakeholders
  ("the riskiest 10% of applications account for X% of all fraud").
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix, classification_report,
    roc_auc_score,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import apply_fitted_woe_tables, TARGET_COL
from src.train_model import prepare_tree_feature_matrix


def get_scores(model_type: str, test_df: pd.DataFrame, fitted_objects: dict) -> np.ndarray:
    """Returns predicted fraud probabilities for the test set from either model."""
    if model_type == "logistic":
        model = fitted_objects["logit_model"]
        woe_tables = fitted_objects["woe_tables"]
        X_test_woe = apply_fitted_woe_tables(test_df, woe_tables)
        # Ensure column order matches what the model was trained on
        X_test_woe = X_test_woe[fitted_objects["woe_cols"]]
        return model.predict_proba(X_test_woe)[:, 1]
    elif model_type == "tree":
        model = fitted_objects["tree_model"]
        encoders = fitted_objects["encoders"]
        X_test, _ = prepare_tree_feature_matrix(test_df, encoders=encoders, fit=False)
        X_test = X_test[fitted_objects["tree_cols"]]
        return model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def business_threshold_confusion_matrix(y_true: np.ndarray, y_scores: np.ndarray,
                                         flag_rate: float = 0.05) -> dict:
    """
    Evaluates the confusion matrix at the threshold that flags the top
    `flag_rate` fraction of applications for manual review -- e.g. 5% ->
    a fraud ops team sized to review the riskiest 1-in-20 applications.
    This mirrors how a real risk team sets a cutoff: by review CAPACITY,
    not by an arbitrary fixed probability like 0.5.
    """
    threshold = np.quantile(y_scores, 1 - flag_rate)
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "flag_rate": flag_rate, "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": precision, "recall": recall,
        "n_flagged": int(tp + fp),
    }


def score_band_table(y_true: np.ndarray, y_scores: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """
    Builds a decile-style score-band table: applications ranked by score
    and bucketed into n_bands equal-SIZE groups (band 1 = riskiest), with
    the observed fraud rate and cumulative fraud capture per band -- the
    standard artifact a risk team uses to set cutoffs and explain model
    lift to stakeholders.
    """
    df = pd.DataFrame({"y_true": y_true, "score": y_scores})
    df["band"] = pd.qcut(df["score"].rank(method="first"), q=n_bands, labels=False)
    df["band"] = n_bands - df["band"]  # band 1 = highest scores = riskiest

    total_fraud = df["y_true"].sum()
    summary = df.groupby("band").agg(
        n_applications=("y_true", "count"),
        n_fraud=("y_true", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index().sort_values("band")

    summary["fraud_rate"] = summary["n_fraud"] / summary["n_applications"]
    summary["pct_of_total_fraud_captured"] = summary["n_fraud"] / total_fraud
    summary["cumulative_fraud_captured"] = summary["pct_of_total_fraud_captured"].cumsum()
    summary["cumulative_pct_of_applications"] = (summary["n_applications"].cumsum() / len(df))

    return summary


def evaluate_model(model_type: str, test_df: pd.DataFrame, fitted_objects: dict,
                    flag_rate: float = 0.05) -> dict:
    y_true = test_df[TARGET_COL].values
    y_scores = get_scores(model_type, test_df, fitted_objects)

    roc_auc = roc_auc_score(y_true, y_scores)
    auc_pr = compute_auc_pr(y_true, y_scores)
    cm = business_threshold_confusion_matrix(y_true, y_scores, flag_rate=flag_rate)
    bands = score_band_table(y_true, y_scores)

    return {
        "model_type": model_type, "roc_auc": roc_auc, "auc_pr": auc_pr,
        "confusion_matrix": cm, "score_bands": bands, "y_scores": y_scores, "y_true": y_true,
    }


def print_evaluation_report(results: dict):
    print(f"\n{'='*60}")
    print(f"MODEL: {results['model_type'].upper()}")
    print(f"{'='*60}")
    print(f"ROC-AUC:  {results['roc_auc']:.4f}  (reference metric, inflated by class imbalance)")
    print(f"AUC-PR:   {results['auc_pr']:.4f}  (PRIMARY metric for this imbalanced task)")

    cm = results["confusion_matrix"]
    print(f"\nConfusion matrix @ top {cm['flag_rate']*100:.0f}% flagged (threshold={cm['threshold']:.4f}):")
    print(f"  Flagged for review: {cm['n_flagged']}")
    print(f"  True Positives:  {cm['tp']:5d}   False Positives: {cm['fp']:5d}")
    print(f"  False Negatives: {cm['fn']:5d}   True Negatives:  {cm['tn']:5d}")
    print(f"  Precision: {cm['precision']:.4f}   Recall: {cm['recall']:.4f}")

    print(f"\nScore band table (decile, band 1 = riskiest):")
    bands = results["score_bands"][[
        "band", "n_applications", "n_fraud", "fraud_rate",
        "pct_of_total_fraud_captured", "cumulative_fraud_captured"
    ]]
    pd.set_option("display.float_format", "{:.4f}".format)
    print(bands.to_string(index=False))


if __name__ == "__main__":
    from src.train_model import train_logistic_scorecard, train_tree_challenger

    train_df = pd.read_csv("/home/claude/synthetic_identity_fraud/data/train.csv")
    test_df = pd.read_csv("/home/claude/synthetic_identity_fraud/data/test.csv")

    logit_bundle = train_logistic_scorecard(train_df, test_df)
    tree_bundle = train_tree_challenger(train_df, test_df)

    fitted_objects = {
        "logit_model": logit_bundle["model"], "woe_tables": logit_bundle["woe_tables"],
        "woe_cols": logit_bundle["feature_names"],
        "tree_model": tree_bundle["model"], "encoders": tree_bundle["encoders"],
        "tree_cols": tree_bundle["feature_names"],
    }

    logit_results = evaluate_model("logistic", test_df, fitted_objects)
    tree_results = evaluate_model("tree", test_df, fitted_objects)

    print_evaluation_report(logit_results)
    print_evaluation_report(tree_results)
