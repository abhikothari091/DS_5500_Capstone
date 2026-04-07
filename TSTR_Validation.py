# =============================================================================
# SYNTHETIC DATA VALIDATION — LAYER 4
# 4A: Archetype Fidelity  |  4B: TSTR  |  4C: SHAP Feature Importance
# =============================================================================
# Target variable: is_significant_overrun = total genuine delay > 60 days
# Genuine delay = TCO shift > 14 days between consecutive TRUE rebaseline events
# Real data:      platinum.models.* (239 projects, ~32.6% positive)
# Synthetic data: analysts.self_managed.* (1,000 projects)
#
# NOTE: Real schedule_baseline is a monthly P6 snapshot table, not an
# event-driven rebaseline log. We deduplicate to true rebaseline events
# (rows where dt_tco actually changed vs the prior row) before computing
# overrun labels to ensure apples-to-apples comparison with synthetic data.
# =============================================================================
!pip install shap
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     RandomizedSearchCV, train_test_split)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed — 4C skipped. Run: %pip install shap")

# ── Tables ───────────────────────────────────────────────────
REAL_PROJECT  = "platinum.models.project"
REAL_SCHEDULE = "platinum.models.schedule_baseline"
SYN_PROJECT   = "analysts.self_managed.synthetic_project_meta_data_profiled"
SYN_SCHEDULE  = "analysts.self_managed.synthetic_schedule_baseline_v1"

CONTRACT_MIN  = 25_000_000
START_YEAR    = 2019
RANDOM_STATE  = 42
N_CV_FOLDS    = 5
N_BOOTSTRAP   = 1_000  # for confidence intervals

print("=" * 70)
print("LAYER 4: ARCHETYPE FIDELITY + TSTR + SHAP")
print("=" * 70)

# =============================================================================
# SECTION 1: Build feature matrices
# =============================================================================
print("\n[1/6] Building feature matrices...")

real_proj_sql = f"""
    SELECT
        p.ids_project_number,
        p.cat_sbu,
        p.cat_main_asset_class,
        p.amt_contract,
        DATEDIFF(p.dt_construction_end, p.dt_construction_start) AS proj_duration_days
    FROM {REAL_PROJECT} p
    WHERE p.amt_contract             >= {CONTRACT_MIN}
      AND YEAR(p.dt_construction_start) >= {START_YEAR}
      AND p.cat_sbu                  IS NOT NULL
      AND p.is_latest                = true
"""

# KEY FIX: deduplicate monthly snapshots to true rebaseline events only.
# A true rebaseline event is a row where dt_tco differs from the previous row.
# Without this, month-to-month noise accumulates spurious delay_days and
# inflates the real overrun rate, making it incomparable to synthetic data.
real_schedule_sql = f"""
    WITH ordered_snapshots AS (
        SELECT
            s.ids_project_number,
            s.dt_tco,
            s.dt_effective_from,
            LAG(s.dt_tco) OVER (
                PARTITION BY s.ids_project_number
                ORDER BY s.dt_effective_from
            ) AS prev_tco
        FROM {REAL_SCHEDULE} s
        JOIN {REAL_PROJECT} p
            ON s.ids_project_number = p.ids_project_number
        WHERE p.amt_contract             >= {CONTRACT_MIN}
          AND YEAR(p.dt_construction_start) >= {START_YEAR}
          AND p.cat_sbu                  IS NOT NULL
          AND p.is_latest                = true
    ),
    true_rebaseline_events AS (
        -- Only rows where dt_tco actually changed from the prior snapshot
        SELECT
            ids_project_number,
            DATEDIFF(dt_tco, prev_tco) AS tco_shift_days
        FROM ordered_snapshots
        WHERE prev_tco IS NOT NULL
          AND dt_tco != prev_tco
    )
    SELECT
        ids_project_number,
        COUNT(*)                                                           AS n_genuine_rebaselines,
        SUM(CASE WHEN tco_shift_days > 14 THEN tco_shift_days ELSE 0 END) AS total_delay_days
    FROM true_rebaseline_events
    GROUP BY ids_project_number
"""

syn_proj_sql = f"""
    SELECT
        p.ids_project_number_synth AS ids_project_number,
        p.cat_sbu,
        p.cat_main_asset_class,
        p.amt_contract,
        DATEDIFF(p.dt_construction_end, p.dt_construction_start) AS proj_duration_days
    FROM {SYN_PROJECT} p
"""

syn_schedule_sql = f"""
    WITH tco_changes AS (
        SELECT
            s.ids_project_number_synth AS ids_project_number,
            DATEDIFF(s.dt_tco,
                LAG(s.dt_tco) OVER (
                    PARTITION BY s.ids_project_number_synth
                    ORDER BY s.dt_effective_from
                )
            ) AS delay_days
        FROM {SYN_SCHEDULE} s
    )
    SELECT
        ids_project_number,
        COUNT(*)        AS n_genuine_rebaselines,
        SUM(delay_days) AS total_delay_days
    FROM tco_changes
    WHERE delay_days > 14
    GROUP BY ids_project_number
"""

real_proj_pd  = spark.sql(real_proj_sql).toPandas()
real_sched_pd = spark.sql(real_schedule_sql).toPandas()
syn_proj_pd   = spark.sql(syn_proj_sql).toPandas()
syn_sched_pd  = spark.sql(syn_schedule_sql).toPandas()

def build_df(proj, sched, source):
    df = proj.merge(sched, on="ids_project_number", how="left")
    df = df.loc[:, ~df.columns.duplicated()]
    df["n_genuine_rebaselines"] = df["n_genuine_rebaselines"].fillna(0).astype(int)
    df["total_delay_days"]      = df["total_delay_days"].fillna(0).astype(float)
    df["is_significant_overrun"] = (df["total_delay_days"] > 60).astype(int)
    df["source"] = source
    return df

real_df = build_df(real_proj_pd, real_sched_pd, "real")
syn_df  = build_df(syn_proj_pd,  syn_sched_pd,  "synthetic")

print(f"  Real:      {len(real_df):,} projects | "
      f"{real_df['is_significant_overrun'].mean():.1%} overrun rate")
print(f"  Synthetic: {len(syn_df):,} projects | "
      f"{syn_df['is_significant_overrun'].mean():.1%} overrun rate")

# =============================================================================
# SECTION 2: Feature engineering
# =============================================================================
print("\n[2/6] Engineering features...")

CONTINUOUS_FEATURES  = ["amt_contract_log", "proj_duration_days"]
CATEGORICAL_FEATURES = ["cat_sbu", "cat_main_asset_class"]
MODEL_FEATURES_BASE  = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

def engineer_features(df):
    d = df.copy()
    d["amt_contract_log"]   = np.log1p(d["amt_contract"].clip(lower=0))
    d["proj_duration_days"] = d["proj_duration_days"].fillna(0).clip(lower=0)
    for col in CATEGORICAL_FEATURES:
        d[col] = d[col].fillna("unknown").str.lower().str.strip()
    return d

real_df = engineer_features(real_df)
syn_df  = engineer_features(syn_df)

combined = pd.concat([real_df, syn_df], ignore_index=True)
encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    le.fit(combined[col])
    encoders[col] = le
    real_df[col + "_enc"] = le.transform(real_df[col])
    syn_df[col  + "_enc"] = le.transform(syn_df[col])

MODEL_FEATURES = CONTINUOUS_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]

X_real = real_df[MODEL_FEATURES].values
y_real = real_df["is_significant_overrun"].values
X_syn  = syn_df[MODEL_FEATURES].values
y_syn  = syn_df["is_significant_overrun"].values

print(f"  Features: {MODEL_FEATURES}")

# =============================================================================
# SECTION 3: 4A — Archetype Fidelity with effect sizes
# =============================================================================
print("\n[3/6] 4A — Archetype Fidelity...")
print("-" * 60)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-9)

results_4a = {}

print("  Continuous features (KS test + Cohen's d effect size):")
print(f"  {'Feature':<25} {'KS':>6} {'p':>8} {'Cohen d':>9} {'Status'}")
print(f"  {'-'*60}")
for feat in CONTINUOUS_FEATURES:
    r_vals = real_df[feat].dropna().values
    s_vals = syn_df[feat].dropna().values
    ks_stat, p_val = stats.ks_2samp(r_vals, s_vals)
    d = cohens_d(r_vals, s_vals)
    effect = "small" if abs(d) < 0.2 else "medium" if abs(d) < 0.5 else "large"
    status = "✅" if abs(d) < 0.5 else "⚠️ "
    results_4a[feat] = {"ks_stat": round(ks_stat, 4), "p_value": round(p_val, 4),
                        "cohens_d": round(d, 3), "effect": effect}
    print(f"  {feat:<25} {ks_stat:>6.3f} {p_val:>8.4f} {d:>9.3f} "
          f"({effect})  {status}")

print(f"\n  Categorical features (chi-squared + Cramér's V effect size):")
print(f"  {'Feature':<25} {'chi2':>10} {'p':>8} {'Cramér V':>10} {'Status'}")
print(f"  {'-'*60}")
for feat in CATEGORICAL_FEATURES:
    r_counts = real_df[feat].value_counts()
    s_counts = syn_df[feat].value_counts()
    all_cats  = sorted(r_counts.index)
    r_freq    = np.array([r_counts.get(c, 0) for c in all_cats], dtype=float)
    s_freq    = np.array([s_counts.get(c, 0) for c in all_cats], dtype=float)
    s_freq    = s_freq / s_freq.sum() * r_freq.sum()
    s_freq    = np.where(s_freq == 0, 0.5, s_freq)
    s_freq    = s_freq * (r_freq.sum() / s_freq.sum())
    mask      = r_freq > 0
    chi2, p_val = stats.chisquare(r_freq[mask], s_freq[mask])
    n   = r_freq.sum()
    k   = mask.sum()
    cramer_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0
    effect = "small" if cramer_v < 0.1 else "medium" if cramer_v < 0.3 else "large"
    status = "✅" if cramer_v < 0.3 else "⚠️ "
    results_4a[feat] = {"chi2": round(chi2, 2), "p_value": round(p_val, 4),
                        "cramers_v": round(cramer_v, 3), "effect": effect}
    print(f"  {feat:<25} {chi2:>10.2f} {p_val:>8.4f} {cramer_v:>10.3f} "
          f"({effect})  {status}")

print(f"\n  Overrun rate — real: {real_df['is_significant_overrun'].mean():.1%} | "
      f"synthetic: {syn_df['is_significant_overrun'].mean():.1%}")
print(f"  Note: p-values are inflated by sample size; effect sizes are the "
      f"meaningful metric.")

# =============================================================================
# SECTION 4: 4B — TSTR with proper ML flow
# =============================================================================
print("\n[4/6] 4B — TSTR (proper ML flow)...")
print("-" * 60)

param_dist = {
    "n_estimators":     randint(100, 500),
    "max_depth":        randint(2, 6),
    "learning_rate":    uniform(0.01, 0.2),
    "subsample":        uniform(0.6, 0.4),
    "min_samples_leaf": randint(5, 30),
}

cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_outer = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── TRTR: nested CV on real data ─────────────────────────────
print("  TRTR — nested CV on real data (tune + evaluate)...")
trtr_scores = []
for fold, (tr_idx, te_idx) in enumerate(cv_outer.split(X_real, y_real)):
    X_tr, X_te = X_real[tr_idx], X_real[te_idx]
    y_tr, y_te = y_real[tr_idx], y_real[te_idx]
    sw_tr = compute_sample_weight("balanced", y_tr)

    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_dist, n_iter=30, cv=cv_inner,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_tr, y_tr, sample_weight=sw_tr)
    y_prob = search.best_estimator_.predict_proba(X_te)[:, 1]
    fold_auc = roc_auc_score(y_te, y_prob)
    trtr_scores.append(fold_auc)
    print(f"    Fold {fold+1}: AUC={fold_auc:.4f}  "
          f"best params: n_est={search.best_params_['n_estimators']}, "
          f"depth={search.best_params_['max_depth']}, "
          f"lr={search.best_params_['learning_rate']:.3f}")

trtr_auc  = np.mean(trtr_scores)
trtr_std  = np.std(trtr_scores)
trtr_ci95 = 1.96 * trtr_std / np.sqrt(N_CV_FOLDS)
print(f"\n  TRTR AUC: {trtr_auc:.4f} ± {trtr_std:.4f}  "
      f"95% CI: [{trtr_auc-trtr_ci95:.4f}, {trtr_auc+trtr_ci95:.4f}]")

# ── TSTR: tune on synthetic, evaluate on all real ────────────
print(f"\n  TSTR — tune on synthetic, evaluate on full real test set...")
sw_syn = compute_sample_weight("balanced", y_syn)

search_syn = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_dist, n_iter=50, cv=cv_inner,
    scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE
)
search_syn.fit(X_syn, y_syn, sample_weight=sw_syn)
best_syn_clf = search_syn.best_estimator_

print(f"  Best synthetic params: {search_syn.best_params_}")

y_prob_tstr = best_syn_clf.predict_proba(X_real)[:, 1]
tstr_auc    = roc_auc_score(y_real, y_prob_tstr)

# Bootstrap CI
rng = np.random.RandomState(RANDOM_STATE)
boot_aucs = []
for _ in range(N_BOOTSTRAP):
    idx = rng.choice(len(y_real), size=len(y_real), replace=True)
    if len(np.unique(y_real[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_real[idx], y_prob_tstr[idx]))

tstr_ci_lo = np.percentile(boot_aucs, 2.5)
tstr_ci_hi = np.percentile(boot_aucs, 97.5)

print(f"\n  TSTR AUC: {tstr_auc:.4f}  "
      f"95% CI: [{tstr_ci_lo:.4f}, {tstr_ci_hi:.4f}]")

utility_ratio = tstr_auc / trtr_auc
status_tstr   = "✅ PASS" if utility_ratio >= 0.75 else "⚠️  BELOW TARGET"
print(f"  Utility ratio (TSTR/TRTR): {utility_ratio:.3f}  {status_tstr}")
print(f"  Interpretation: synthetic data retains "
      f"{utility_ratio*100:.1f}% of real data's predictive signal")

# TSTR by SBU
print(f"\n  TSTR AUC by SBU:")
for sbu in sorted(real_df["cat_sbu"].unique()):
    mask = real_df["cat_sbu"].values == sbu
    if mask.sum() < 10 or y_real[mask].sum() < 2:
        continue
    sbu_auc = roc_auc_score(y_real[mask], y_prob_tstr[mask])
    n_pos   = y_real[mask].sum()
    print(f"    {sbu:<22} n={mask.sum():>4}  pos={n_pos:>3}  AUC={sbu_auc:.3f}")

# =============================================================================
# SECTION 5: 4C — SHAP Feature Importance
# =============================================================================
print("\n[5/6] 4C — SHAP Feature Importance Preservation...")
print("-" * 60)

if not SHAP_AVAILABLE:
    print("  SKIPPED — install shap first")
else:
    search_real_final = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_dist, n_iter=50, cv=cv_inner,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE
    )
    search_real_final.fit(X_real, y_real,
                          sample_weight=compute_sample_weight("balanced", y_real))
    clf_real_final = search_real_final.best_estimator_

    explainer_real = shap.TreeExplainer(clf_real_final)
    explainer_syn  = shap.TreeExplainer(best_syn_clf)

    shap_real = np.abs(explainer_real.shap_values(X_real)).mean(axis=0)
    shap_syn  = np.abs(explainer_syn.shap_values(X_syn)).mean(axis=0)

    spearman_r, spearman_p = stats.spearmanr(shap_real, shap_syn)
    status_shap = "✅ PASS" if spearman_r >= 0.70 else "⚠️  LOW CORRELATION"

    print(f"  Spearman rank correlation: {spearman_r:.4f}  {status_shap}")
    print(f"  Target: ≥ 0.70")
    print()
    feat_labels = MODEL_FEATURES
    real_ranks  = pd.Series(shap_real, index=feat_labels).rank(ascending=False)
    syn_ranks   = pd.Series(shap_syn,  index=feat_labels).rank(ascending=False)

    print(f"  {'Feature':<30} {'Real rank':>10} {'Syn rank':>10} "
          f"{'Real SHAP':>12} {'Syn SHAP':>12}")
    print(f"  {'-'*76}")
    for i, feat in enumerate(feat_labels):
        print(f"  {feat:<30} {int(real_ranks[feat]):>10} {int(syn_ranks[feat]):>10} "
              f"{shap_real[i]:>12.4f} {shap_syn[i]:>12.4f}")

# =============================================================================
# SECTION 6: Summary
# =============================================================================
print("\n[6/6] Layer 4 Summary")
print("=" * 70)

print("\n  4A — Archetype Fidelity (effect size is the meaningful metric):")
for feat, res in results_4a.items():
    if "cohens_d" in res:
        status = "✅" if abs(res["cohens_d"]) < 0.5 else "⚠️ "
        print(f"    {status} {feat}: Cohen's d={res['cohens_d']} ({res['effect']})")
    else:
        status = "✅" if res["cramers_v"] < 0.3 else "⚠️ "
        print(f"    {status} {feat}: Cramér's V={res['cramers_v']} ({res['effect']})")

print(f"\n  4B — TSTR:")
print(f"    TRTR AUC: {trtr_auc:.4f} ± {trtr_std:.4f} "
      f"[{trtr_auc-trtr_ci95:.4f}, {trtr_auc+trtr_ci95:.4f}]")
print(f"    TSTR AUC: {tstr_auc:.4f} "
      f"[{tstr_ci_lo:.4f}, {tstr_ci_hi:.4f}]")
print(f"    Utility:  {utility_ratio:.3f} "
      f"({'✅ ≥0.75' if utility_ratio >= 0.75 else '⚠️ <0.75'})")

if SHAP_AVAILABLE:
    print(f"\n  4C — SHAP:")
    print(f"    Spearman r: {spearman_r:.4f} "
          f"({'✅ ≥0.70' if spearman_r >= 0.70 else '⚠️ <0.70'})")

print("\n  Layer 4 complete.")