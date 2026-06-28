# Synthetic Identity Fraud Detection — Application-Stage Model

A production-style Python project detecting **synthetic identity fraud** at
the credit card / loan / BNPL application stage, built as a portfolio piece
demonstrating entity-resolution / graph-based fraud detection — the kind of
approach core to vendors like Sardine, Feedzai, and DataVisor.

This is **not** a transaction-fraud model. The fraud happens before the
account ever exists: at the point a fabricated or "Frankenstein" identity
(a mix of real and fake identity elements — SSN, name, DOB, address) applies
to open a new line of credit.

---

## 1. What's in this repo

```
src/
  data_generator.py      # synthetic application dataset (rings + lone-wolf fraud + legit look-alikes)
  graph_features.py      # bipartite identity-link graph + entity-resolution features
  feature_engineering.py # WoE/IV scorecard feature selection
  train_model.py         # leakage-safe split + logistic scorecard + tree challenger model
  evaluate_model.py       # AUC-PR, business-threshold confusion matrix, score-band/decile table
  explainability.py       # global importance + per-case fraud-ops-style narratives
  eda.py                  # class imbalance, distributions, correlation, graph structure charts
data/                     # generated datasets (raw, with graph features, train/test splits)
outputs/                  # charts (PNG) and tabular reports (CSV/TXT)
```

Run order: `data_generator.py` → `graph_features.py` → `eda.py` (optional,
for charts) → `train_model.py` → `evaluate_model.py` / `explainability.py`.

---

## 2. Key design decisions

### 2.1 Bipartite graph, not an application-to-application projection

The identity-link graph is built as **bipartite**: one node type for
applications, one node type for identity elements (SSN, address, device
fingerprint, email, phone). An edge connects an application to each of its
5 identity elements. Two applications are linked *indirectly*, by both
touching the same element node.

This was chosen over a direct application-to-application projection for
three reasons:

1. **Efficiency.** A projection graph requires materializing an edge for
   every pair of applications that share an element. A single popular
   address used by 50 applications would alone create ~1,225 projected
   edges. The bipartite form stays sparse — every application has exactly
   5 edges, full stop, regardless of how many others share its elements.
2. **Interpretability.** "This application connects to this address node,
   which connects to 5 other applications" is a sentence a fraud ops
   analyst can follow. An abstract weighted projection edge is harder to
   explain in a case review.
3. **Multi-relational signal for free.** The bipartite structure makes it
   trivial to ask "how many *distinct element types* does this application
   share with its strongest-linked neighbor" — which turns out to be a much
   stronger signal than raw component size (see §4).

### 2.2 Graph features computed

| Feature | What it measures |
|---|---|
| `component_size_apps` | Size of the connected component this application belongs to ("ring size") |
| `app_degree` | Number of distinct identity elements (sanity-check, always 5 here) |
| `max_element_reuse` | Of this application's 5 elements, the highest reuse count by any single element |
| `n_elements_reused` | How many of the 5 elements are shared with ≥1 other application |
| `n_distinct_element_types_reused_with_top_neighbor` | Strongest signal: does this application share **multiple** element types with the *same* neighbor (e.g. both address AND device) — a much rarer, more deliberate-looking pattern than coincidentally sharing just one element |
| `shares_multi_element_with_any_neighbor` | Boolean flag version of the above |
| `n_neighbors_sharing_2plus_elements` | Count of neighbors meeting that multi-element bar |

### 2.3 Train/test split: by connected component, not by row

Two applications in the same component are not independent from the
graph's point of view. Randomly splitting rows can put two ring siblings
on opposite sides of train/test — the model would then partially "see"
that ring's shared device/address during training and get an unrealistic
head start recognizing its sibling at test time. This is a leakage mode
that's easy to miss with graph features specifically (it doesn't occur
with plain tabular features). **The split here is done at the
`component_id` level**: every application in a component goes entirely to
one side. The split is stratified on whether each component contains any
fraud, to keep train/test fraud rates comparable despite the coarser
splitting unit.

### 2.4 WoE/IV scorecard feature selection, with two extra safeguards

Information Value (IV) is used to rank candidate features for the logistic
scorecard, following standard credit-risk convention. Two things came up
during development that are worth documenting because they're genuine,
common scorecard pitfalls:

- **Per-bin WoE capping.** Several graph features have bins with
  near-perfect separation (e.g. "shares an element with 6+ other
  applications" is, in this dataset, ~100% fraud). Mathematically, WoE for
  such a bin is `log(near-zero / large-number)`, which blows toward
  ±infinity. That's an artifact of small-sample/extreme-bin behavior, not
  a sign the feature is more informative than it really is. We cap
  `|WoE|` at 4.0 per bin (a convention used by common scorecard tooling)
  and additionally cap total feature-level IV at 0.65, scaling bin
  contributions down proportionally if needed.
- **Correlation pruning across BOTH numeric and boolean features.** The
  graph-derived features are highly correlated with each other (0.6–0.98
  in this data — see `outputs/eda_correlation_heatmap.png`), and so are
  several boolean attribute features (`thin_file_flag` is mechanically
  almost the inverse of `bureau_hit`). Feeding correlated features into
  the same logistic regression produces unstable, sometimes sign-flipped
  coefficients — a real bug we caught during development, where
  `thin_file_flag`'s coefficient came out *positive* (implying it lowered
  fraud risk, the opposite of reality) purely because `bureau_hit` and
  `credit_file_age_years` were already absorbing that signal. The fix:
  rank IV-qualified features by IV, and greedily keep a feature only if
  its correlation with every already-kept feature is ≤ 0.55.

### 2.5 Class imbalance handling

At a ~2.2% fraud rate, two standard approaches were considered:

- **Class weighting** (used here): reweights the loss function so
  misclassifying a fraud case costs more, without altering the underlying
  data distribution. Chosen for the logistic scorecard because it leaves
  the WoE bin statistics — fit on the real, unresampled population —
  intact. SMOTE on WoE-encoded features would synthesize new points by
  interpolating between WoE values, which doesn't have a coherent
  interpretation (a WoE value isn't really a continuous quantity that's
  meaningful to interpolate; it's an estimated log-odds derived from
  *which bin* a raw value fell into).
- **SMOTE** (not used, but documented as the alternative): oversamples the
  minority class by generating synthetic points in feature space. More
  common when the model has no natural weighting mechanism, or when the
  modeler wants to literally rebalance the training set seen by the
  optimizer. The tradeoff is risk of generating unrealistic synthetic
  fraud examples, especially in a space (WoE) where interpolation is
  semantically murky, and the loss of the "this exact loss-weighting
  reflects true business cost of a missed fraud" interpretability that
  class weighting preserves.

For the tree model (HistGradientBoostingClassifier), class imbalance is
handled via `sample_weight` at `.fit()` time, set to the negative/positive
class ratio — the sklearn-idiomatic equivalent of XGBoost's
`scale_pos_weight`.

---

## 3. How this maps to real-world vendor terminology

| This project | Industry term |
|---|---|
| Bipartite identity-element graph | **Entity resolution** / identity graph |
| Connected component | **Consortium link** / fraud ring / network cluster |
| `n_distinct_element_types_reused_with_top_neighbor` | **Multi-attribute linkage** — the strength signal vendors like Sardine and DataVisor emphasize over simple shared-attribute counts |
| Cross-application identity element reuse | **Velocity / reuse signals** in a consortium data model |
| Component-based train/test split | Standard practice when validating any **graph/network feature** model, to avoid "the model already saw this ring" leakage |

A real Sardine/DataVisor/Feedzai-style system extends this same idea
**across issuers** (a cross-institution consortium graph), not just within
one company's application data — see Limitations below.

---

## 4. What the EDA and model results actually show

Some of the most useful findings came from looking at the *false positive
risk* before even building a model:

- Of 2,587 multi-application connected components, **2,499 (96.6%) are
  pure-legit** (families, roommates, large-apartment-building neighbors
  coincidentally sharing an address) and only **80 (3.1%) are pure
  fraud rings**, with 8 mixed edge cases. (`outputs/eda_graph_structure.png`)
- This is why `component_size_apps` alone is a weak design choice for a
  cutoff rule — most multi-application components are completely innocent.
  The stronger signal is **multi-element overlap with the same specific
  neighbor** (`n_neighbors_sharing_2plus_elements`), which is far rarer for
  legitimate shared households (who'd need to coincidentally share *two*
  separate identity elements with the same other applicant) than for a
  fraud operator deliberately reusing infrastructure.

Final model comparison (component-based test set, ~2.0% fraud rate):

| Metric | Logistic Scorecard | Tree Challenger (HistGB) |
|---|---|---|
| AUC-PR (primary) | 0.449 | 0.561 |
| ROC-AUC (reference) | 0.839 | 0.918 |
| Recall @ top 5% flagged | 51.2% | 64.0% |
| Precision @ top 5% flagged | 20.9% | 26.1% |
| Top decile fraud capture | 58.6% | 74.4% |

The tree model outperforms the linear scorecard, as expected — it can
exploit non-linear interactions between graph and attribute features
(e.g. "high limit-to-income ratio is only really alarming *combined with*
a thin file AND a shared device") that a linear model structurally cannot.
The scorecard remains valuable as the auditable, regulator-explainable
artifact; the tree model is the higher-performance secondary signal,
matching how many risk teams actually deploy a champion/challenger pair.

---

## 5. Limitations

- **Single-issuer simulation.** This dataset simulates one company's
  application data. Real synthetic identity detection gets dramatically
  stronger with **cross-issuer consortium data** (the actual product
  Sardine/DataVisor/Feedzai sell) — an identity element reused across
  *different companies'* applications is a much stronger tell than reuse
  within one company's data alone, since a fraud operator targeting a
  single issuer repeatedly is a narrower, easier-to-catch case than one
  spreading attempts across many issuers.
- **Synthetic data, not real bureau/application data.** Feature
  distributions and fraud "tells" were deliberately engineered to be
  realistic but are simulated. Several feature strengths were intentionally
  tuned down during development after an initial pass produced Information
  Values far outside any believable real-world range (a useful exercise in
  itself — the IV-capping and correlation-pruning logic in §2.4 exists
  because of exactly that). A production model would discover these
  relationships empirically rather than have them designed in.
- **XGBoost → HistGradientBoostingClassifier substitution.** This was
  built in a sandboxed environment without internet access, so
  `pip install xgboost` could not run. `HistGradientBoostingClassifier`
  (scikit-learn's native histogram-based gradient boosting) was used as
  the closest available substitute — same algorithmic family, similar
  performance profile on tabular data this size. On a machine with normal
  package access, swap in:
  ```python
  from xgboost import XGBClassifier
  tree_model = XGBClassifier(
      n_estimators=300, max_depth=5, learning_rate=0.05,
      scale_pos_weight=<n_neg/n_pos>, eval_metric="aucpr", random_state=42,
  )
  ```
  as a drop-in replacement in `train_model.py`; the rest of the pipeline
  (split, evaluation, explainability) is written to work with either.
- **SHAP → permutation importance + exact-linear-contribution
  substitution.** Same root cause — `pip install shap` was unavailable.
  Global importance uses `sklearn.inspection.permutation_importance`
  (AUC-PR-scored). Per-case narratives for the logistic scorecard use the
  model's *exact* linear decomposition (`coefficient × WoE value` per
  feature, which sums exactly to the model's logit — mathematically
  equivalent to what SHAP computes for a linear model, since a linear
  model's SHAP values reduce to coefficient × feature contribution as a
  known special case). The tree model's narratives use a simpler
  importance-weighted heuristic rather than true Shapley values. On a
  machine with internet access:
  ```python
  import shap
  explainer = shap.TreeExplainer(tree_model)
  shap_values = explainer.shap_values(X_test)
  ```
  would give exact per-case attribution for the tree model.
- **Coarse classing simplification.** Full WoE coarse-classing practice
  would merge adjacent bins by similarity of WoE/risk, not just by raw bin
  size or a flat feature-level cap. The simpler size-and-cap-based approach
  used here is documented in `feature_engineering.py` and was a deliberate
  scope tradeoff for this portfolio project.

---

## 6. Reproducing

```bash
cd src
python3 data_generator.py        # -> data/applications_raw.csv
python3 graph_features.py        # -> data/applications_with_graph_features.csv
python3 eda.py                   # -> outputs/*.png
python3 train_model.py           # -> data/train.csv, data/test.csv
python3 evaluate_model.py        # -> prints evaluation report
python3 explainability.py        # -> prints importance + case narratives
```

All scripts are deterministic (`random_state=42` throughout) for
reproducibility.
