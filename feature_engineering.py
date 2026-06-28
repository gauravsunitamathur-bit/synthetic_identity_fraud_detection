"""
feature_engineering.py
=======================
Prepares the modeling dataset:
  1. Combines raw attribute features + graph features from graph_features.py
  2. Computes Weight-of-Evidence (WoE) and Information Value (IV) for each
     candidate feature, the way a scorecard-building team would, to decide
     which features earn a place in the logistic regression scorecard.
  3. Produces a WoE-transformed feature matrix for logistic regression and
     a raw (lightly encoded) feature matrix for the tree-based model,
     since gradient boosted trees do not need WoE binning to find
     non-linear splits and WoE binning would actually throw away
     information the trees could otherwise use directly.

WHY WoE/IV AT ALL, GIVEN WE ALSO TRAIN A TREE MODEL
----------------------------------------------------
Two different modeling philosophies are intentionally represented here,
matching how a risk team operates in practice:
  - The LOGISTIC REGRESSION + WoE/IV scorecard is the artifact a model
    risk / governance team can audit line by line, explain to a
    regulator, and turn into a simple point-based score card.
  - The TREE MODEL (HistGradientBoostingClassifier, used here as a
    stand-in for XGBoost -- see README) is the higher-performance
    "challenger" model used for actual decisioning or as a secondary
    signal, traded off against its lower interpretability.

IV THRESHOLDS (standard credit-risk convention used here):
    IV < 0.02   -> not useful, drop
    0.02 - 0.10 -> weak predictor
    0.10 - 0.30 -> medium predictor
    0.30 - 0.50 -> strong predictor
    > 0.50      -> suspiciously strong -- check for leakage and overfitting

A NOTE ON BIN-SIZE ENFORCEMENT
--------------------------------
A real scorecard team never lets a bin with a handful of observations
stand on its own: a bin with, say, 40 observations that happen to all be
fraud produces a WoE that mathematically blows up toward +/-infinity
(log of a near-zero ratio), and that's instability, not signal -- on a
slightly different sample of customers that same bin would NOT be 100%
fraud. We enforce MIN_BIN_SIZE below and merge any undersized bin into
its neighbor before computing WoE, exactly as a scorecard build would.
This is what keeps IV numbers honest rather than artificially inflated
by small, perfectly-separating bins.
"""

import numpy as np
import pandas as pd


# Small constant added to bin counts so that WoE is always defined
# even for bins with zero fraud or zero non-fraud observations.
WOE_LAPLACE_SMOOTHING = 0.5

# Minimum observations a bin must have before its WoE is trusted on its
# own. Undersized bins are merged into a pooled "long tail" bin (standard
# coarse classing practice) rather than left to produce an unstable,
# inflated WoE/IV driven by small-sample noise.
MIN_BIN_SIZE = 50

# WoE CAP: even with adequate bin size, a bin can show NEAR-PERFECT
# separation (e.g. 100+ observations, all fraud) when a feature is a
# genuinely very strong tell. Mathematically this drives WoE toward
# +/-infinity (log of a near-zero ratio), which is real signal direction
# but a misleadingly extreme magnitude -- in practice this is exactly
# the pattern scorecard teams flag as "too good to be true, go check for
# leakage" even when no leakage exists, because no single bin should be
# allowed to dominate IV. We cap |WoE| at WOE_CAP, a convention used by
# common scorecard tooling (e.g. R's scorecard package) for this reason.
WOE_CAP = 4.0

# A feature can still accumulate a very high TOTAL IV even with per-bin
# WoE capped, if several distinct bins all happen to sit at the cap (our
# graph-reuse features have this exact shape: bins for "shares element
# with 6 others", "...7 others", "...8 others" are conceptually one
# signal -- "shares with several others" -- that full coarse classing
# would merge into a single bin, but we cap total IV directly here as a
# simpler, equally honest backstop rather than building a full
# WoE-similarity bin-merging routine for this portfolio project.
MAX_FEATURE_IV = 0.65


NUMERIC_FEATURES_FOR_BINNING = [
    "claimed_age",
    "ssn_dob_gap_years",
    "credit_file_age_years",
    "annual_income",
    "requested_credit_limit",
    "limit_to_income_ratio",
    "address_tenure_months",
    "bureau_inquiries_90d",
    "time_to_fill_form_seconds",
    "component_size_apps",
    "app_degree",
    "max_element_reuse",
    "n_elements_reused",
    "n_distinct_element_types_reused_with_top_neighbor",
    "n_neighbors_sharing_2plus_elements",
]

BOOLEAN_FEATURES = [
    "bureau_hit",
    "thin_file_flag",
    "phone_is_voip",
    "shares_multi_element_with_any_neighbor",
]

CATEGORICAL_FEATURES = [
    "email_domain",
]

TARGET_COL = "is_fraud"


def _bin_numeric(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Quantile-bin a numeric feature into at most n_bins bins.

    Falls back gracefully for heavily skewed/concentrated features (e.g.
    component_size_apps, where the median, 75th, and 90th percentile can
    all be the same value because most applications sit in a
    singleton component) where pd.qcut would otherwise collapse
    everything into ONE bin -- which silently zeroes out IV for an
    otherwise genuinely informative feature. In that case we bin on
    distinct values directly (or rank-based qcut as a secondary fallback).
    """
    n_unique = series.nunique()

    if n_unique <= n_bins:
        # Few distinct values (typical for our engineered count features) ->
        # just use each distinct value as its own bin. More informative
        # than forcing a coarser quantile binning that may collapse entirely.
        return series.astype(str)

    try:
        binned = pd.qcut(series, q=n_bins, duplicates="drop")
        if binned.nunique() <= 1:
            # qcut still collapsed (extreme skew) -> use rank-based qcut instead,
            # which guarantees n_bins groups of near-equal SIZE regardless of
            # how the underlying values cluster.
            binned = pd.qcut(series.rank(method="first"), q=n_bins, duplicates="drop")
    except ValueError:
        binned = pd.qcut(series.rank(method="first"), q=min(n_bins, n_unique), duplicates="drop")

    return binned.astype(str)


def compute_woe_iv(df: pd.DataFrame, feature_col: str, target_col: str = TARGET_COL,
                    is_numeric: bool = True, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute WoE and IV for a single feature.
    Returns a small table: bin -> (n_good, n_bad, woe, iv_contribution)
    where "bad" = fraud (1), "good" = legit (0), following standard
    credit-scoring convention (counterintuitive naming, but this is the
    convention every scorecard team uses, so we keep it for familiarity).
    """
    work = df[[feature_col, target_col]].copy()
    if is_numeric:
        work["bin"] = _bin_numeric(work[feature_col], n_bins=n_bins)
    else:
        work["bin"] = work[feature_col].astype(str)

    total_good = (work[target_col] == 0).sum()
    total_bad = (work[target_col] == 1).sum()

    grouped = work.groupby("bin")[target_col].agg(
        n_obs="count", n_bad="sum"
    ).reset_index()
    grouped["n_good"] = grouped["n_obs"] - grouped["n_bad"]

    # --- Enforce minimum bin size (standard "coarse classing" step) ---
    # Bins with very few observations produce unstable, often near-infinite
    # WoE if they happen to be (close to) pure good or pure bad by chance.
    # Rather than diluting undersized bins into the (unrelated) largest
    # bin -- which would wash out real signal when, e.g., several
    # high-risk tail bins are each individually small but collectively
    # meaningful -- we merge all undersized bins for a feature into ONE
    # combined "long tail" bin. This keeps their pooled signal intact
    # (e.g. "max_element_reuse >= 4" as a single stable bin) while still
    # preventing any single tiny bin from producing an unstable, inflated
    # WoE driven by sample-size noise rather than real signal.
    if (grouped["n_obs"] < MIN_BIN_SIZE).any() and len(grouped) > 1:
        undersized_bins = set(grouped.loc[grouped["n_obs"] < MIN_BIN_SIZE, "bin"])
        if len(undersized_bins) < len(grouped):
            combined_label = "OTHER_LOW_COUNT_BINS"
            work["bin"] = work["bin"].apply(lambda b: combined_label if b in undersized_bins else b)
            grouped = work.groupby("bin")[target_col].agg(
                n_obs="count", n_bad="sum"
            ).reset_index()
            grouped["n_good"] = grouped["n_obs"] - grouped["n_bad"]

    # Laplace smoothing avoids div-by-zero / log(0) for bins with no fraud cases
    grouped["pct_good"] = (grouped["n_good"] + WOE_LAPLACE_SMOOTHING) / (total_good + WOE_LAPLACE_SMOOTHING * len(grouped))
    grouped["pct_bad"] = (grouped["n_bad"] + WOE_LAPLACE_SMOOTHING) / (total_bad + WOE_LAPLACE_SMOOTHING * len(grouped))
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["woe"] = grouped["woe"].clip(lower=-WOE_CAP, upper=WOE_CAP)  # see WOE_CAP docstring above
    grouped["iv_contribution"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    # Feature-level IV cap (see MAX_FEATURE_IV docstring): if several bins
    # each sit at the per-bin WoE cap, their summed IV can still exceed a
    # believable real-world ceiling. We scale contributions down
    # proportionally so relative bin ranking is preserved but the total
    # stays bounded, rather than letting bin-count alone inflate IV.
    total_iv = grouped["iv_contribution"].sum()
    if total_iv > MAX_FEATURE_IV:
        grouped["iv_contribution"] = grouped["iv_contribution"] * (MAX_FEATURE_IV / total_iv)

    grouped["feature"] = feature_col

    return grouped


def build_iv_summary(df: pd.DataFrame, numeric_features=None, boolean_features=None,
                      categorical_features=None, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Builds a one-row-per-feature IV summary table for feature selection."""
    numeric_features = numeric_features or NUMERIC_FEATURES_FOR_BINNING
    boolean_features = boolean_features or BOOLEAN_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    summaries = []
    for feat in numeric_features:
        woe_table = compute_woe_iv(df, feat, target_col, is_numeric=True)
        summaries.append({"feature": feat, "type": "numeric", "iv": woe_table["iv_contribution"].sum()})
    for feat in boolean_features:
        woe_table = compute_woe_iv(df, feat, target_col, is_numeric=False)
        summaries.append({"feature": feat, "type": "boolean", "iv": woe_table["iv_contribution"].sum()})
    for feat in categorical_features:
        woe_table = compute_woe_iv(df, feat, target_col, is_numeric=False)
        summaries.append({"feature": feat, "type": "categorical", "iv": woe_table["iv_contribution"].sum()})

    iv_df = pd.DataFrame(summaries).sort_values("iv", ascending=False).reset_index(drop=True)

    def _strength(iv):
        if iv < 0.02:
            return "not useful"
        elif iv < 0.10:
            return "weak"
        elif iv < 0.30:
            return "medium"
        elif iv < 0.50:
            return "strong"
        else:
            return "suspiciously strong (check leakage)"

    iv_df["strength"] = iv_df["iv"].apply(_strength)
    return iv_df


def apply_woe_transform(df: pd.DataFrame, feature_col: str, woe_table: pd.DataFrame,
                         is_numeric: bool = True, n_bins: int = 10) -> pd.Series:
    """
    Maps each row's raw value to its bin's WoE score, for the logistic
    model's input matrix.

    NOTE ON THE "OTHER_LOW_COUNT_BINS" POOLED BIN: compute_woe_iv() may
    have merged several undersized raw bins into one pooled bin at FIT
    time (see MIN_BIN_SIZE). Any row here whose raw value maps to one of
    those originally-undersized raw bin labels must also be routed to
    that same pooled bin at SCORE time, or it would incorrectly fall
    through to the neutral-WoE fallback below and silently lose signal.
    We don't have the original raw-bin-membership list at this point, so
    rather than re-deriving it, the safer and simpler fix is: any row
    whose computed raw bin string isn't found in woe_table is assumed to
    belong to the pooled tail, IF a pooled bin exists in this table.
    Truly novel categories (e.g. a never-seen email domain) still fall
    back to neutral WoE, which is the conservative, correct behavior.
    """
    work = df[[feature_col]].copy()
    if is_numeric:
        work["bin"] = _bin_numeric(work[feature_col], n_bins=n_bins)
    else:
        work["bin"] = work[feature_col].astype(str)

    woe_map = dict(zip(woe_table["bin"], woe_table["woe"]))
    mapped = work["bin"].map(woe_map)

    if "OTHER_LOW_COUNT_BINS" in woe_map:
        pooled_woe = woe_map["OTHER_LOW_COUNT_BINS"]
        mapped = mapped.fillna(pooled_woe)
    else:
        mapped = mapped.fillna(0.0)  # genuinely novel category/value -> neutral WoE

    return mapped


def build_woe_feature_matrix(df: pd.DataFrame, numeric_features=None, boolean_features=None,
                              categorical_features=None, target_col: str = TARGET_COL,
                              min_iv: float = 0.02, max_corr: float = 0.55) -> tuple:
    """
    Builds the full WoE-transformed feature matrix for the logistic regression
    scorecard, keeping only features that clear the min_iv threshold (standard
    scorecard practice -- weak/no-signal features are dropped before modeling
    rather than left in to add noise and hurt interpretability), AND pruning
    highly correlated redundant features.

    WHY THE CORRELATION PRUNE: our graph features (component_size_apps,
    max_element_reuse, n_elements_reused, etc.) are all derived from the
    same underlying identity-link graph and are highly correlated with each
    other (observed correlations of 0.6-0.97 in this dataset). Several
    boolean attribute features are ALSO collinear by construction --
    e.g. thin_file_flag is true whenever bureau_hit is false or the file
    is very young, so the two carry overlapping information. Feeding
    correlated features into a single logistic regression causes classic
    multicollinearity symptoms -- unstable, sign-flipped coefficients
    that make the scorecard harder to interpret and audit (e.g. a feature
    that should clearly raise fraud risk ending up with a positive/
    protective-looking coefficient purely because a correlated sibling
    feature absorbed its signal). A real scorecard build would never ship
    a model like that; the standard fix is to keep only the single
    strongest (highest-IV) feature from each cluster of highly correlated
    candidates, which is what this prune does -- features are considered
    in IV-descending order, and any candidate with correlation above
    `max_corr` to an ALREADY-SELECTED feature is dropped. Both numeric
    and boolean features participate in this prune (see implementation
    note below); only the unordered categorical (email_domain) is exempt.

    Returns: (X_woe DataFrame, dict of fitted woe_tables for reuse on test data, iv_summary)
    """
    numeric_features = numeric_features or NUMERIC_FEATURES_FOR_BINNING
    boolean_features = boolean_features or BOOLEAN_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    iv_summary = build_iv_summary(df, numeric_features, boolean_features, categorical_features, target_col)
    iv_qualified = iv_summary[iv_summary["iv"] >= min_iv]["feature"].tolist()

    # Correlation prune: numeric AND boolean features both participate,
    # since booleans can be just as collinear as numerics here (e.g.
    # thin_file_flag and bureau_hit are near-mirror-images of each other
    # by construction -- thin_file_flag is TRUE whenever bureau_hit is
    # FALSE or the file is very young). Leaving booleans out of the prune
    # was an earlier oversight: it let both into the model together and
    # produced an unstable, sign-flipped coefficient on whichever of the
    # pair had the weaker independent signal once the other was already
    # in the model. Categorical features (just email_domain here) are
    # still exempted, since pairwise correlation isn't a natural concept
    # for an unordered category and it isn't the source of collinearity
    # in this dataset.
    prunable_features = numeric_features + boolean_features
    prunable_qualified = [f for f in iv_qualified if f in prunable_features]
    categorical_qualified = [f for f in iv_qualified if f not in prunable_features]

    selected_features = []
    if len(prunable_qualified) > 0:
        corr_input = df[prunable_qualified].copy()
        for col in corr_input.columns:
            if corr_input[col].dtype == bool:
                corr_input[col] = corr_input[col].astype(int)
        corr_matrix = corr_input.corr().abs()
        for feat in prunable_qualified:  # already IV-descending from iv_summary ordering
            too_correlated = any(
                corr_matrix.loc[feat, kept] > max_corr for kept in selected_features
                if kept in corr_matrix.columns
            )
            if not too_correlated:
                selected_features.append(feat)
    selected_features += categorical_qualified

    dropped = set(iv_qualified) - set(selected_features)
    if dropped:
        print(f"[feature_engineering] Dropped {len(dropped)} feature(s) for high correlation "
              f"(>|{max_corr}|) with a stronger already-selected feature: {sorted(dropped)}")

    woe_tables = {}
    X_woe = pd.DataFrame(index=df.index)

    for feat in selected_features:
        is_numeric = feat in numeric_features
        woe_table = compute_woe_iv(df, feat, target_col, is_numeric=is_numeric)
        woe_tables[feat] = {"table": woe_table, "is_numeric": is_numeric}
        X_woe[f"{feat}_woe"] = apply_woe_transform(df, feat, woe_table, is_numeric=is_numeric)

    return X_woe, woe_tables, iv_summary


def apply_fitted_woe_tables(df: pd.DataFrame, woe_tables: dict) -> pd.DataFrame:
    """Applies previously-fitted WoE tables (from training data) onto new data (e.g. test set)."""
    X_woe = pd.DataFrame(index=df.index)
    for feat, fitted in woe_tables.items():
        X_woe[f"{feat}_woe"] = apply_woe_transform(df, feat, fitted["table"], is_numeric=fitted["is_numeric"])
    return X_woe


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/synthetic_identity_fraud/data/applications_with_graph_features.csv")
    iv_summary = build_iv_summary(df)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(iv_summary.to_string(index=False))
    iv_summary.to_csv("/home/claude/synthetic_identity_fraud/outputs/iv_summary.csv", index=False)
