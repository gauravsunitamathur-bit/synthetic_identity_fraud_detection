"""
data_generator.py
==================
Generates a synthetic application-fraud dataset for SYNTHETIC IDENTITY
detection at the credit/loan/BNPL application stage.

DESIGN PHILOSOPHY
------------------
Real synthetic identity fraud rarely looks like isolated bad applications.
It typically shows up as:
  (a) "Ring" fraud — an operator reuses a small set of infrastructure
      (an address, a device, a few burner emails/phones) across several
      fabricated identities to build multiple credit files in parallel.
  (b) "Lone wolf" synthetic identities — a single fabricated identity
      with weak/no links to anything else, harder to catch via network
      signals alone and needs to be caught by attribute-level signals
      (SSN/DOB inconsistency, thin file, etc.)

We deliberately generate BOTH so that a model relying purely on
"is this applicant in a big connected component" cannot achieve high
recall — it must also learn the attribute-level tells. This avoids
building a dataset with a trivial shortcut.

We also inject LEGITIMATE look-alikes (recent immigrants, young first-time
applicants, family members sharing an address, business addresses shared by
coworkers) so that graph/thin-file features alone are not perfectly
separable from genuine cases — mirroring the real-world false-positive
risk that any genuine synthetic-ID model has to manage.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

RNG_SEED = 42


@dataclass
class GenConfig:
    n_applications: int = 40_000
    fraud_rate: float = 0.022          # ~2.2% overall, between the requested 1.5-3%
    ring_fraud_share: float = 0.55     # 55% of fraud apps belong to a "ring", 45% are lone-wolf synthetic IDs
    legit_shared_infra_rate: float = 0.18  # 18% of LEGIT apps legitimately share an address/device (family, roommates, coworkers, apartment buildings, business addresses) -- realistically dilutes the graph signal so ring features aren't a near-perfect tell
    seed: int = RNG_SEED


def _make_pool(n, prefix):
    """Utility: generate a pool of unique-looking identity-element strings."""
    return np.array([f"{prefix}_{i:07d}" for i in range(n)])


def generate_applications(cfg: GenConfig = GenConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_applications
    n_fraud = int(n * cfg.fraud_rate)
    n_legit = n - n_fraud

    n_ring_fraud = int(n_fraud * cfg.ring_fraud_share)
    n_lone_fraud = n_fraud - n_ring_fraud

    rows = []
    app_id_counter = 0

    # ---- Identity element allocation strategy ----
    # KEY FIX: random sampling-with-replacement from a pool only ~1x the
    # population size produces heavy incidental collisions at this scale
    # (birthday-paradox effect), which would bridge the entire population
    # into one giant connected component -- unrealistic, since in real
    # data the vast majority of SSNs/addresses/devices are NOT shared.
    #
    # So: every application gets a UNIQUE SSN/address/device by default
    # (via simple incrementing counters), and identity-element SHARING is
    # introduced ONLY deliberately, in the specific blocks designed for it
    # (legit shared-infra clusters, ring fraud clusters). This guarantees
    # the link graph reflects intentional design, not pool-size accidents.
    _unique_ssn_counter = [0]
    _unique_addr_counter = [0]
    _unique_device_counter = [0]

    def next_unique_ssn():
        _unique_ssn_counter[0] += 1
        return f"SSN_{_unique_ssn_counter[0]:08d}"

    def next_unique_addr():
        _unique_addr_counter[0] += 1
        return f"ADDR_{_unique_addr_counter[0]:08d}"

    def next_unique_device():
        _unique_device_counter[0] += 1
        return f"DEV_{_unique_device_counter[0]:08d}"

    email_domain_pool_legit = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "hotmail.com"]
    email_domain_pool_risky = ["tempmail.com", "mailinator.com", "protonmail.com", "yopmail.com", "guerrillamail.com"]

    # =========================================================
    # 1) LEGITIMATE APPLICATIONS
    # =========================================================
    n_legit_shared = int(n_legit * cfg.legit_shared_infra_rate)
    n_legit_normal = n_legit - n_legit_shared

    # 1a) Normal legit applicants — mostly unique everything
    for _ in range(n_legit_normal):
        age = int(np.clip(rng.normal(40, 13), 19, 85))
        birth_year = 2026 - age
        # SSN issuance year roughly consistent with birth year for genuine people.
        # Small share have a slightly larger admin-lag gap (late registration,
        # naturalized citizens, paperwork delays) so this feature is informative
        # but not a perfectly clean discriminator on its own.
        if rng.random() < 0.04:
            ssn_issue_year = birth_year + rng.integers(3, 9)
        else:
            ssn_issue_year = birth_year + rng.integers(0, 2)
        bureau_hit = rng.random() > 0.04  # 96% of genuine adults have a bureau file
        file_age_years = max(0, age - 18 + rng.integers(-2, 3)) if bureau_hit else 0
        thin_file = bureau_hit and file_age_years < 1.5
        # young or recent-immigrant legit look-alikes: thin file despite being adult
        if age < 23 or rng.random() < 0.05:
            thin_file = thin_file or (rng.random() < 0.4)
            if rng.random() < 0.3:
                bureau_hit = False

        income = max(15000, rng.normal(55000, 24000))
        requested_limit = max(500, rng.normal(0.18, 0.05) * income)

        email_domain = rng.choice(email_domain_pool_legit)
        rows.append(_row(
            app_id_counter, rng, ssn=next_unique_ssn(), address=next_unique_addr(),
            device=next_unique_device(), email_domain=email_domain,
            phone_voip=rng.random() < 0.03, age=age, ssn_issue_year=ssn_issue_year,
            birth_year=birth_year, bureau_hit=bureau_hit, thin_file=thin_file,
            file_age_years=round(file_age_years, 1), income=round(income, 2),
            requested_limit=round(requested_limit, 2),
            address_tenure_months=int(np.clip(rng.exponential(36), 0, 360)),
            inquiry_count_90d=int(np.clip(rng.poisson(1.1), 0, 15)),
            time_to_fill_seconds=int(np.clip(rng.normal(240, 90), 30, 1800)),
            label=0,
        ))
        app_id_counter += 1

    # 1b) Legit shared-infrastructure cases (families, roommates, coworkers)
    #     -> deliberately creates legit clusters that LOOK like rings on
    #        pure link-count features, forcing the model to use richer signal.
    #     Cluster sizes are kept small (2-5) -- intentionally distinct from
    #     ring fraud's wider 2-9 range -- and we track remaining row BUDGET
    #     explicitly so total legit row count stays correct rather than
    #     overshooting based on an upfront cluster-count estimate.
    legit_shared_remaining = n_legit_shared
    while legit_shared_remaining > 0:
        shared_addr = next_unique_addr()  # one fresh address shared by this cluster
        cluster_size = min(int(rng.integers(2, 5)), legit_shared_remaining)
        # a family/roommate cluster may also share a household device (smart TV/router-based fingerprint proxy)
        shared_device = next_unique_device() if rng.random() < 0.4 else None
        for _ in range(cluster_size):
            age = int(np.clip(rng.normal(35, 14), 18, 80))
            birth_year = 2026 - age
            if rng.random() < 0.04:
                ssn_issue_year = birth_year + rng.integers(3, 9)
            else:
                ssn_issue_year = birth_year + rng.integers(0, 2)
            bureau_hit = rng.random() > 0.05
            file_age_years = max(0, age - 18 + rng.integers(-2, 3)) if bureau_hit else 0
            income = max(15000, rng.normal(50000, 22000))
            requested_limit = max(500, rng.normal(0.18, 0.05) * income)
            rows.append(_row(
                app_id_counter, rng, ssn=next_unique_ssn(), address=shared_addr,
                device=shared_device if shared_device is not None else next_unique_device(),
                email_domain=rng.choice(email_domain_pool_legit),
                phone_voip=rng.random() < 0.03, age=age, ssn_issue_year=ssn_issue_year,
                birth_year=birth_year, bureau_hit=bureau_hit,
                thin_file=bureau_hit and file_age_years < 1.5,
                file_age_years=round(file_age_years, 1), income=round(income, 2),
                requested_limit=round(requested_limit, 2),
                address_tenure_months=int(np.clip(rng.exponential(48), 0, 360)),
                inquiry_count_90d=int(np.clip(rng.poisson(1.0), 0, 15)),
                time_to_fill_seconds=int(np.clip(rng.normal(230, 85), 30, 1800)),
                label=0,
            ))
            app_id_counter += 1
        legit_shared_remaining -= cluster_size

    # =========================================================
    # 2) LONE-WOLF SYNTHETIC IDENTITY FRAUD
    #    Weak/no network links -> must be caught via attribute tells:
    #    SSN/DOB mismatch, thin file despite claimed age, risky email,
    #    VOIP phone, high limit-to-income ask, fast form-fill.
    # =========================================================
    for _ in range(n_lone_fraud):
        claimed_age = int(np.clip(rng.normal(36, 10), 21, 70))
        birth_year = 2026 - claimed_age
        # SSN issuance year inconsistent with claimed birth year (the classic synthetic-ID tell)
        # 15% of the time the gap is small (operator got a more convincing SSN source),
        # so this single feature is informative but not a perfect giveaway on its own.
        if rng.random() < 0.15:
            ssn_issue_year = birth_year + rng.integers(0, 4)
        else:
            ssn_issue_year = birth_year + rng.integers(10, 35)
        bureau_hit = rng.random() < 0.55   # synthetic IDs often have thin/no file, but plenty pass a soft bureau check
        # Occasionally a synthetic ID has "aged" a file via piggybacking/tradeline abuse
        # before applying here, so a meaningful share are NOT thin-file -> avoids near-perfect separability.
        if bureau_hit and rng.random() < 0.30:
            file_age_years = round(rng.uniform(1.5, 6.0), 1)
        else:
            file_age_years = round(rng.uniform(0, 1.4), 1) if bureau_hit else 0
        thin_file = (not bureau_hit) or file_age_years < 1.5

        income = max(20000, rng.normal(58000, 20000))  # often a fabricated, plausible-looking income
        requested_limit = max(1000, rng.normal(0.26, 0.09) * income)  # asks for somewhat more relative to income, with overlap

        email_domain = rng.choice(email_domain_pool_risky if rng.random() < 0.40 else email_domain_pool_legit)
        rows.append(_row(
            app_id_counter, rng, ssn=next_unique_ssn(), address=next_unique_addr(),
            device=next_unique_device(), email_domain=email_domain,
            phone_voip=rng.random() < 0.30, age=claimed_age, ssn_issue_year=ssn_issue_year,
            birth_year=birth_year, bureau_hit=bureau_hit, thin_file=thin_file,
            file_age_years=file_age_years, income=round(income, 2),
            requested_limit=round(requested_limit, 2),
            address_tenure_months=int(np.clip(rng.exponential(14), 0, 60)),  # newer than legit on average, but wide overlap
            inquiry_count_90d=int(np.clip(rng.poisson(1.9), 0, 20)),
            time_to_fill_seconds=int(np.clip(rng.normal(150, 80), 15, 1800)),  # somewhat faster than legit, with overlap
            label=1,
        ))
        app_id_counter += 1

    # =========================================================
    # 3) RING-BASED SYNTHETIC IDENTITY FRAUD
    #    Operator reuses a small pool of addresses/devices/emails/phones
    #    across multiple fabricated identities -> creates connected
    #    components in the link graph. Ring size varies (2-9) to avoid
    #    the model learning "component size > threshold = fraud" trivially.
    # =========================================================
    fraud_remaining = n_ring_fraud
    ring_phone_counter = [0]

    def next_ring_phone_prefix():
        ring_phone_counter[0] += 1
        return f"RINGPHONE_{ring_phone_counter[0]:05d}"

    while fraud_remaining > 0:
        ring_size = int(rng.integers(2, 10))
        ring_size = min(ring_size, fraud_remaining)
        # Each ring gets its OWN freshly-unique shared elements (not drawn from
        # a small shared pool across rings) so that DIFFERENT rings never
        # accidentally collide with each other and merge into one mega-component.
        # A ring shares 1-3 of its identity element types among members (not
        # all 5), keeping the signal realistic rather than a perfect giveaway.
        shared_addr = next_unique_addr() if rng.random() < 0.75 else None
        shared_device = next_unique_device() if rng.random() < 0.65 else None
        shared_phone_prefix = next_ring_phone_prefix() if rng.random() < 0.5 else None
        # Guarantee every ring has at least ONE shared element -- otherwise
        # it isn't actually a detectable ring in the graph at all, which
        # would silently shrink our intended ring-fraud share.
        if shared_addr is None and shared_device is None and shared_phone_prefix is None:
            shared_addr = next_unique_addr()

        for _ in range(ring_size):
            claimed_age = int(np.clip(rng.normal(34, 9), 21, 65))
            birth_year = 2026 - claimed_age
            if rng.random() < 0.15:
                ssn_issue_year = birth_year + rng.integers(0, 4)
            else:
                ssn_issue_year = birth_year + rng.integers(10, 35)
            bureau_hit = rng.random() < 0.45
            if bureau_hit and rng.random() < 0.30:
                file_age_years = round(rng.uniform(1.5, 5.0), 1)
            else:
                file_age_years = round(rng.uniform(0, 1.4), 1) if bureau_hit else 0
            thin_file = (not bureau_hit) or file_age_years < 1.5

            income = max(20000, rng.normal(60000, 22000))
            requested_limit = max(1000, rng.normal(0.27, 0.09) * income)

            email_domain = rng.choice(email_domain_pool_risky if rng.random() < 0.50 else email_domain_pool_legit)
            rows.append(_row(
                app_id_counter, rng,
                ssn=next_unique_ssn(),
                address=shared_addr if shared_addr is not None else next_unique_addr(),
                device=shared_device if shared_device is not None else next_unique_device(),
                email_domain=email_domain,
                phone_voip=rng.random() < 0.35,
                phone_prefix_override=shared_phone_prefix,
                age=claimed_age, ssn_issue_year=ssn_issue_year, birth_year=birth_year,
                bureau_hit=bureau_hit, thin_file=thin_file, file_age_years=file_age_years,
                income=round(income, 2), requested_limit=round(requested_limit, 2),
                address_tenure_months=int(np.clip(rng.exponential(10), 0, 48)),
                inquiry_count_90d=int(np.clip(rng.poisson(2.2), 0, 25)),
                time_to_fill_seconds=int(np.clip(rng.normal(85, 35), 15, 1800)),
                label=1,
            ))
            app_id_counter += 1

        fraud_remaining -= ring_size

    df = pd.DataFrame(rows)
    df = _inject_realistic_overlap_noise(df, rng)
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)  # shuffle rows
    df["application_id"] = [f"APP_{i:07d}" for i in range(len(df))]
    cols = ["application_id"] + [c for c in df.columns if c != "application_id"]
    return df[cols]


def _inject_realistic_overlap_noise(df: pd.DataFrame, rng: np.random.Generator,
                                     blend_fraction: float = 0.55) -> pd.DataFrame:
    """
    Caps feature separability at a believable real-world level.

    PROBLEM THIS SOLVES: generating fraud and legit rows from two cleanly
    different distributions (as the population blocks above do) produces
    means that barely overlap at all once averaged across thousands of
    rows -- e.g. credit_file_age_years of ~21 years for legit vs ~0.8 for
    fraud, with almost no shared range. Real bureau/application data is
    never this clean; legitimate thin-file applicants and well-aged
    synthetic identities genuinely overlap on most single features. Without
    correcting this, every feature's Information Value (IV) comes out far
    above the 0.3-0.5 ceiling real scorecard features ever reach, which
    would read as obvious data leakage to anyone reviewing this portfolio
    project with scorecard experience.

    FIX: for a random `blend_fraction` of rows in EACH class, swap that
    row's value on a handful of the strongest individual features for a
    value resampled from the OPPOSITE class's empirical distribution. This
    directly creates the missing overlap in the tails without touching the
    network/graph structure (which should remain a clean, deliberate
    signal, since real consortium-style graph signals genuinely are
    strong when available -- the realism gap here is specifically in the
    single-feature attribute signals, not the graph signal).
    """
def _inject_realistic_overlap_noise(df: pd.DataFrame, rng: np.random.Generator,
                                     target_iv_low: float = 0.15,
                                     target_iv_high: float = 0.45) -> pd.DataFrame:
    """
    Caps feature separability at a believable real-world level by
    calibrating a PER-COLUMN blend strength so each feature's resulting
    Information Value (IV) lands in [target_iv_low, target_iv_high] --
    the range real scorecard features actually occupy (medium-to-strong,
    per standard credit-risk IV convention) -- rather than applying one
    flat blend percentage to every column and hoping it lands close enough.

    PROBLEM THIS SOLVES: generating fraud and legit rows from two cleanly
    different distributions produces near-zero overlap once averaged
    across thousands of rows, which inflates IV far past anything seen in
    real bureau/application data (where IV > 0.5 is itself a red flag for
    leakage). A single flat blend fraction overcorrects some columns while
    undercorrecting others, since different features start from different
    levels of separability. Calibrating per column avoids both failure modes.

    METHOD: for each column, binary-search the blend fraction (the percent
    of each class's rows that get their value replaced with a value
    resampled from the OPPOSITE class's empirical distribution) until the
    resulting single-feature IV falls inside the target band.
    """
    try:
        from src.feature_engineering import compute_woe_iv
    except ModuleNotFoundError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.feature_engineering import compute_woe_iv

    df = df.copy()
    legit_idx = df.index[df["is_fraud"] == 0]
    fraud_idx = df.index[df["is_fraud"] == 1]

    blend_cols_numeric = [
        "credit_file_age_years", "ssn_dob_gap_years", "time_to_fill_form_seconds",
        "address_tenure_months", "bureau_inquiries_90d", "limit_to_income_ratio",
        "requested_credit_limit", "claimed_age",
    ]
    blend_cols_categorical = ["bureau_hit", "phone_is_voip", "email_domain"]

    original_values = {col: df[col].copy() for col in blend_cols_numeric + blend_cols_categorical}

    def _blend_column(col, frac, is_numeric):
        """Apply a given blend fraction to ONE column, starting fresh from the original values."""
        df[col] = original_values[col]  # reset before reapplying at a new strength
        fraud_pool = original_values[col].loc[fraud_idx].values
        legit_pool = original_values[col].loc[legit_idx].values

        n_legit_blend = int(len(legit_idx) * frac)
        n_fraud_blend = int(len(fraud_idx) * frac)
        sel_legit = rng.choice(legit_idx, size=n_legit_blend, replace=False)
        sel_fraud = rng.choice(fraud_idx, size=n_fraud_blend, replace=False)

        df.loc[sel_legit, col] = rng.choice(fraud_pool, size=len(sel_legit), replace=True)
        df.loc[sel_fraud, col] = rng.choice(legit_pool, size=len(sel_fraud), replace=True)

    def _iv_for_column(col, is_numeric):
        table = compute_woe_iv(df, col, "is_fraud", is_numeric=is_numeric)
        return table["iv_contribution"].sum()

    for col in blend_cols_numeric + blend_cols_categorical:
        is_numeric = col in blend_cols_numeric
        # Binary search blend fraction in [0, 0.97] for ~8 iterations -- cheap,
        # and avoids hand-tuning ~10 separate constants by trial and error.
        lo, hi = 0.0, 0.97
        best_frac = 0.0
        for _ in range(8):
            mid = (lo + hi) / 2
            _blend_column(col, mid, is_numeric)
            iv = _iv_for_column(col, is_numeric)
            if iv > target_iv_high:
                lo = mid  # need MORE blending to reduce IV further
            elif iv < target_iv_low:
                hi = mid  # need LESS blending, too washed out
            else:
                best_frac = mid
                break
            best_frac = mid
        _blend_column(col, best_frac, is_numeric)  # leave column set at final chosen strength

    # Recompute downstream-derived fields that depend on blended raw values,
    # so the dataset stays internally consistent (e.g. thin_file_flag must
    # still match the (possibly now-blended) credit_file_age_years/bureau_hit,
    # and ssn_issue_year must still match the blended ssn_dob_gap_years).
    df["thin_file_flag"] = (~df["bureau_hit"]) | (df["credit_file_age_years"] < 1.5)
    df["limit_to_income_ratio"] = (df["requested_credit_limit"] / df["annual_income"]).round(3)
    df["ssn_issue_year"] = df["claimed_birth_year"] + df["ssn_dob_gap_years"]

    return df


def _row(app_id_counter, rng, ssn, address, device, email_domain, phone_voip, age,
         ssn_issue_year, birth_year, bureau_hit, thin_file, file_age_years, income,
         requested_limit, address_tenure_months, inquiry_count_90d, time_to_fill_seconds,
         label, phone_prefix_override=None):
    phone_prefix = phone_prefix_override if phone_prefix_override is not None else f"P_{rng.integers(200, 999)}"
    email_local = f"user{rng.integers(10000, 999999)}"
    return {
        "row_id": app_id_counter,
        "ssn": ssn,
        "address": address,
        "device_fingerprint": device,
        "email": f"{email_local}@{email_domain}",
        "email_domain": email_domain,
        "phone_number": f"{phone_prefix}_{rng.integers(1000,9999)}",
        "phone_is_voip": bool(phone_voip),
        "claimed_age": age,
        "claimed_birth_year": birth_year,
        "ssn_issue_year": ssn_issue_year,
        "ssn_dob_gap_years": ssn_issue_year - birth_year,  # core synthetic-ID tell
        "bureau_hit": bool(bureau_hit),
        "thin_file_flag": bool(thin_file),
        "credit_file_age_years": file_age_years,
        "annual_income": income,
        "requested_credit_limit": requested_limit,
        "limit_to_income_ratio": round(requested_limit / income, 3),
        "address_tenure_months": address_tenure_months,
        "bureau_inquiries_90d": inquiry_count_90d,
        "time_to_fill_form_seconds": time_to_fill_seconds,
        "is_fraud": label,
    }


if __name__ == "__main__":
    df = generate_applications()
    print(df.shape)
    print(df["is_fraud"].value_counts(normalize=True))
    df.to_csv("/home/claude/synthetic_identity_fraud/data/applications_raw.csv", index=False)
    print("Saved to data/applications_raw.csv")
