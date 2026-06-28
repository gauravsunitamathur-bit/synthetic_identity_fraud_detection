"""
eda.py
======
Exploratory data analysis for the synthetic identity fraud dataset:
  1. Class imbalance summary
  2. Feature distributions split by fraud label (the attribute-level
     "tells" a fraud analyst would eyeball first)
  3. Correlation analysis among numeric/graph features (motivates the
     correlation-prune step in feature_engineering.py)
  4. Graph structure: connected component size distribution, and how
     many components are pure-legit, pure-fraud, or mixed (the legit
     shared-infra "look-alike" clusters we deliberately injected)

Run this AFTER data_generator.py and graph_features.py have produced
data/applications_with_graph_features.csv.

Saves chart PNGs to outputs/ and prints summary tables to stdout.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/home/claude/synthetic_identity_fraud/data/applications_with_graph_features.csv"
OUTPUT_DIR = "/home/claude/synthetic_identity_fraud/outputs"

sns.set_style("whitegrid")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # legit=blue, fraud=red, consistent across all charts


def class_imbalance_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. CLASS IMBALANCE")
    print("=" * 60)
    counts = df["is_fraud"].value_counts()
    rates = df["is_fraud"].value_counts(normalize=True)
    print(f"Total applications: {len(df):,}")
    print(f"Legit (0): {counts[0]:,}  ({rates[0]:.2%})")
    print(f"Fraud (1): {counts[1]:,}  ({rates[1]:.2%})")
    print(f"Imbalance ratio: 1 fraud per {counts[0]/counts[1]:.1f} legit applications")
    print()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Plots distributions for the key attribute-level tells, split by
    label -- the first thing a fraud analyst looks at before any model
    is even built.
    """
    features = [
        ("ssn_dob_gap_years", "SSN Issuance Year vs. Claimed Birth Year (gap, years)"),
        ("credit_file_age_years", "Bureau File Age (years)"),
        ("limit_to_income_ratio", "Requested Limit / Income Ratio"),
        ("time_to_fill_form_seconds", "Time to Fill Application Form (seconds)"),
        ("address_tenure_months", "Address Tenure (months)"),
        ("bureau_inquiries_90d", "Bureau Inquiries, Last 90 Days"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, features):
        for label, color in PALETTE.items():
            subset = df[df["is_fraud"] == label][col]
            ax.hist(subset, bins=30, alpha=0.55, label=("Fraud" if label else "Legit"),
                    color=color, density=True)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/eda_feature_distributions.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def plot_boolean_feature_rates(df: pd.DataFrame) -> None:
    """Bar chart of fraud rate within each level of key boolean/categorical tells."""
    bool_features = ["bureau_hit", "thin_file_flag", "phone_is_voip"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, bool_features):
        rates = df.groupby(col)["is_fraud"].mean()
        ax.bar(rates.index.astype(str), rates.values, color=["#4C72B0", "#C44E52"])
        ax.set_title(f"Fraud rate by {col}", fontsize=10)
        ax.set_ylabel("Fraud rate")
        ax.set_ylim(0, rates.values.max() * 1.25)  # headroom scaled to this chart's own range
        offset = rates.values.max() * 0.04
        for i, v in enumerate(rates.values):
            ax.text(i, v + offset, f"{v:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/eda_boolean_fraud_rates.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Correlation heatmap among the numeric/graph features -- the
    diagnostic that motivates the correlation-prune step in
    feature_engineering.py. Graph-derived features cluster tightly
    together since they're all functions of the same underlying graph.
    """
    print("=" * 60)
    print("3. CORRELATION ANALYSIS (numeric + graph features)")
    print("=" * 60)

    numeric_cols = [
        "claimed_age", "ssn_dob_gap_years", "credit_file_age_years", "annual_income",
        "requested_credit_limit", "limit_to_income_ratio", "address_tenure_months",
        "bureau_inquiries_90d", "time_to_fill_form_seconds",
        "component_size_apps", "app_degree", "max_element_reuse", "n_elements_reused",
        "n_distinct_element_types_reused_with_top_neighbor", "n_neighbors_sharing_2plus_elements",
    ]
    corr = df[numeric_cols].corr()

    # Flag the highly-correlated graph feature cluster explicitly
    graph_cols = ["component_size_apps", "max_element_reuse", "n_elements_reused",
                  "n_distinct_element_types_reused_with_top_neighbor", "n_neighbors_sharing_2plus_elements"]
    graph_corr = df[graph_cols].corr()
    print("Correlation among graph-derived features (motivates the correlation prune):")
    print(graph_corr.round(2).to_string())
    print()

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
    plt.title("Feature Correlation Matrix (numeric + graph features)", fontsize=12)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/eda_correlation_heatmap.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")
    print()


def graph_structure_analysis(df: pd.DataFrame) -> None:
    """
    Describes the identity-link graph's component structure: how many
    components exist, their size distribution, and critically -- how
    many components are PURE LEGIT (e.g. real families/roommates sharing
    an address), PURE FRAUD (rings), or MIXED (a ring member coincidentally
    linked to an unrelated legit application via a shared large-building
    address, or similar edge cases). This is the chart that tells the
    story of WHY pure component-size cannot be the only fraud signal.
    """
    print("=" * 60)
    print("4. GRAPH STRUCTURE (identity-link components)")
    print("=" * 60)

    comp_summary = df.groupby("component_id").agg(
        size=("application_id", "count"),
        n_fraud=("is_fraud", "sum"),
    ).reset_index()
    comp_summary["composition"] = np.where(
        comp_summary["n_fraud"] == 0, "pure_legit",
        np.where(comp_summary["n_fraud"] == comp_summary["size"], "pure_fraud", "mixed")
    )

    print(f"Total connected components: {len(comp_summary):,}")
    print(f"Singleton components (no shared identity elements): "
          f"{(comp_summary['size'] == 1).sum():,} "
          f"({(comp_summary['size'] == 1).mean():.1%})")
    print(f"Multi-application components: {(comp_summary['size'] > 1).sum():,}")
    print()
    print("Composition of MULTI-application components:")
    multi = comp_summary[comp_summary["size"] > 1]
    print(multi["composition"].value_counts().to_string())
    print()
    print("Component size distribution (multi-application components only):")
    print(multi["size"].describe().to_string())
    print()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    multi["size"].clip(upper=12).value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="#55A868"
    )
    axes[0].set_title("Component Size Distribution (multi-app components, 12+ capped)")
    axes[0].set_xlabel("Component size (applications)")
    axes[0].set_ylabel("Number of components")

    composition_counts = multi["composition"].value_counts()
    axes[1].pie(composition_counts.values, labels=composition_counts.index, autopct="%1.1f%%",
                colors=["#4C72B0", "#C44E52", "#DD8452"])
    axes[1].set_title("Composition of Multi-Application Components\n(pure_legit = families/roommates, "
                       "pure_fraud = rings,\nmixed = edge cases)", fontsize=9)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/eda_graph_structure.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")
    print()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    class_imbalance_summary(df)

    print("=" * 60)
    print("2. FEATURE DISTRIBUTIONS BY LABEL")
    print("=" * 60)
    plot_feature_distributions(df)
    plot_boolean_feature_rates(df)
    print()

    correlation_analysis(df)
    graph_structure_analysis(df)

    print("EDA complete. Charts saved to outputs/.")
