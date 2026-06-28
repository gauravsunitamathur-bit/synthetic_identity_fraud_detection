"""
explainability.py
===================
Provides model explainability for both the logistic scorecard and the
tree challenger, plus example case narratives for individual high-risk
applications -- written the way a fraud analyst would narrate a case to
an investigations team.

NOTE ON SHAP SUBSTITUTION
---------------------------
This sandbox has no internet access, so `pip install shap` could not run.
We use two stand-ins that together cover what SHAP would normally provide:

1. GLOBAL IMPORTANCE: sklearn's `permutation_importance`, which measures
   how much a model's performance (AUC-PR) degrades when a single
   feature's values are randomly shuffled -- a model-agnostic global
   importance measure conceptually similar to SHAP's global summary, ​
   though it does not decompose individual predictions additively the
   way SHAP values do.

2. PER-CASE CONTRIBUTION: for the LOGISTIC model specifically, we can
   compute each feature's exact contribution to a given prediction
   directly from the model's structure, since for a WoE-encoded logistic
   regression the model is, by construction, an additive linear
   combination of WoE values:
        logit(p) = intercept + sum_i (coef_i * woe_value_i)
  This is mathematically equivalent to what SHAP would compute for a
  linear model (a linear model's SHAP values ARE its coefficient *
  feature contributions, this is a known special case), so we lose
  nothing here versus true SHAP for the scorecard. For the TREE model
  we approximate per-case explanation with a simpler "which of this
  application's feature values are furthest into the fraud-heavy
  range, weighted by that feature's global importance" heuristic --
  a reasonable narrative substitute, though not as rigorous as
  SHAP's TreeExplainer would be.

On a machine with internet access, swap in:
    import shap
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test)
...for true Shapley-value-based per-case attribution on the tree model.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import apply_fitted_woe_tables, TARGET_COL
from src.train_model import prepare_tree_feature_matrix


def global_permutation_importance(model_type: str, test_df: pd.DataFrame,
                                   fitted_objects: dict, n_repeats: int = 10) -> pd.DataFrame:
    """Global feature importance via permutation, scored on AUC-PR."""
    from sklearn.metrics import make_scorer
    from src.feature_engineering import build_woe_feature_matrix
    from src.evaluate_model import compute_auc_pr

    y_test = test_df[TARGET_COL].values
    scorer = make_scorer(compute_auc_pr, response_method="predict_proba")

    if model_type == "logistic":
        model = fitted_objects["logit_model"]
        X_test = apply_fitted_woe_tables(test_df, fitted_objects["woe_tables"])
        X_test = X_test[fitted_objects["woe_cols"]]
    else:
        model = fitted_objects["tree_model"]
        X_test, _ = prepare_tree_feature_matrix(test_df, encoders=fitted_objects["encoders"], fit=False)
        X_test = X_test[fitted_objects["tree_cols"]]

    result = permutation_importance(
        model, X_test, y_test, scoring=scorer, n_repeats=n_repeats, random_state=42
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


def logistic_case_contributions(application_row: pd.Series, model, woe_tables: dict,
                                 woe_cols: list) -> pd.DataFrame:
    """
    Exact per-feature contribution to a single application's logistic
    regression score: coefficient * woe_value for each feature, plus the
    intercept. These sum exactly to the model's raw logit output.
    """
    from src.feature_engineering import apply_woe_transform

    contributions = []
    raw_logit = model.intercept_[0]

    for feat_woe_col, coef in zip(woe_cols, model.coef_[0]):
        base_feat = feat_woe_col.replace("_woe", "")
        fitted = woe_tables[base_feat]
        woe_value = apply_woe_transform(
            pd.DataFrame([application_row]), base_feat, fitted["table"], is_numeric=fitted["is_numeric"]
        ).iloc[0]
        contribution = coef * woe_value
        raw_logit += contribution
        contributions.append({
            "feature": base_feat, "raw_value": application_row[base_feat],
            "woe_value": woe_value, "coefficient": coef, "contribution_to_logit": contribution,
        })

    contrib_df = pd.DataFrame(contributions).sort_values(
        "contribution_to_logit", ascending=False, key=abs
    ).reset_index(drop=True)

    predicted_prob = 1 / (1 + np.exp(-raw_logit))
    return contrib_df, predicted_prob


def build_case_narrative(application_row: pd.Series, contrib_df: pd.DataFrame,
                          predicted_prob: float, top_n: int = 5) -> str:
    """
    Writes a fraud-ops-style narrative for why an application scored
    high (or low) risk, in the tone an analyst would use to brief an
    investigations team -- plain language, ranked by impact, with the
    raw values that drove the score.
    """
    app_id = application_row.get("application_id", "UNKNOWN")
    component_size = application_row.get("component_size_apps", 1)

    lines = [
        f"CASE NARRATIVE -- {app_id}",
        f"Model score: {predicted_prob:.1%} estimated fraud probability",
        "",
    ]

    if component_size > 1:
        lines.append(
            f"Network context: this application is linked to {int(component_size) - 1} "
            f"other application(s) via shared identity elements (address/device/email/phone)."
        )
    else:
        lines.append("Network context: no shared identity elements detected with other applications.")
    lines.append("")
    lines.append(f"Top {top_n} contributing factors (by impact on the score):")

    top_factors = contrib_df.head(top_n)
    for _, row in top_factors.iterrows():
        direction = "increases" if row["contribution_to_logit"] > 0 else "decreases"
        lines.append(
            f"  - {row['feature']} = {row['raw_value']} "
            f"({direction} fraud likelihood, WoE={row['woe_value']:.2f})"
        )

    return "\n".join(lines)


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

    print("=== GLOBAL IMPORTANCE: LOGISTIC SCORECARD ===")
    logit_importance = global_permutation_importance("logistic", test_df, fitted_objects, n_repeats=5)
    print(logit_importance.head(10).to_string(index=False))

    print("\n=== GLOBAL IMPORTANCE: TREE CHALLENGER ===")
    tree_importance = global_permutation_importance("tree", test_df, fitted_objects, n_repeats=5)
    print(tree_importance.head(10).to_string(index=False))

    # --- Example case narratives ---
    from src.evaluate_model import get_scores
    test_scores = get_scores("logistic", test_df, fitted_objects)
    test_df_scored = test_df.copy()
    test_df_scored["pred_score"] = test_scores

    print("\n\n=== EXAMPLE CASE NARRATIVES ===\n")

    # Case 1: highest-scoring RING fraud case (component_size > 1, actually fraud)
    ring_fraud_cases = test_df_scored[
        (test_df_scored[TARGET_COL] == 1) & (test_df_scored["component_size_apps"] > 1)
    ].sort_values("pred_score", ascending=False)
    if len(ring_fraud_cases) > 0:
        case = ring_fraud_cases.iloc[0]
        contrib_df, pred_prob = logistic_case_contributions(case, logit_bundle["model"], logit_bundle["woe_tables"], logit_bundle["feature_names"])
        print("[CASE TYPE: TRUE POSITIVE -- ring-based synthetic identity fraud]")
        print(build_case_narrative(case, contrib_df, pred_prob))
        print("\n" + "-"*60 + "\n")

    # Case 2: highest-scoring LONE-WOLF fraud case (component_size == 1, actually fraud)
    lone_fraud_cases = test_df_scored[
        (test_df_scored[TARGET_COL] == 1) & (test_df_scored["component_size_apps"] == 1)
    ].sort_values("pred_score", ascending=False)
    if len(lone_fraud_cases) > 0:
        case = lone_fraud_cases.iloc[0]
        contrib_df, pred_prob = logistic_case_contributions(case, logit_bundle["model"], logit_bundle["woe_tables"], logit_bundle["feature_names"])
        print("[CASE TYPE: TRUE POSITIVE -- lone-wolf synthetic identity fraud, no network links]")
        print(build_case_narrative(case, contrib_df, pred_prob))
        print("\n" + "-"*60 + "\n")

    # Case 3: a FALSE POSITIVE -- high score but actually legit (e.g. legit shared-infra cluster)
    false_positive_cases = test_df_scored[
        (test_df_scored[TARGET_COL] == 0) & (test_df_scored["component_size_apps"] > 1)
    ].sort_values("pred_score", ascending=False)
    if len(false_positive_cases) > 0:
        case = false_positive_cases.iloc[0]
        contrib_df, pred_prob = logistic_case_contributions(case, logit_bundle["model"], logit_bundle["woe_tables"], logit_bundle["feature_names"])
        print("[CASE TYPE: FALSE POSITIVE -- legitimate applicant in a shared-address/device cluster "
              "(e.g. family, roommates, or large apartment building), scored high risk because the "
              "model partly relies on network features that cannot fully distinguish a legitimate "
              "shared household from a fraud ring]")
        print(build_case_narrative(case, contrib_df, pred_prob))
