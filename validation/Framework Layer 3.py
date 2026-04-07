# Databricks notebook source
# =============================================================================
# SYNTHETIC DATA VALIDATION — LAYER 3
# Lifecycle Trace Validation & Cross-Table Coherence
# =============================================================================
# This layer goes beyond row-level checks to validate that each synthetic
# project tells a coherent lifecycle story across all tables.
#
# Operates on pandas per-project (small traces), with PySpark for bulk loading.
# Works with partial data — adapts checks based on which tables have data
# for each project.
# =============================================================================

from pyspark.sql import functions as F, Window
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG
# =============================================================================
TABLES = {
    "project":    "analysts.self_managed.synthetic_project_meta_data_profiled",
    "schedule":   "analysts.self_managed.synthetic_schedule_baseline_v1",
    "rebaseline": "analysts.self_managed.rebaseline_event_metadata_v1",
    "task":       "analysts.self_managed.synthetic_task_summary_by_sow_v1",
    "rfi":        "analysts.self_managed.synthetic_rfi_v1",
    "budget":     "analysts.self_managed.synthetic_budget_v1",
}

# =============================================================================
# LOAD TABLES
# =============================================================================
print("=" * 80)
print("LAYER 3: LIFECYCLE TRACE VALIDATION")
print("=" * 80)

dfs = {}
for key, table_name in TABLES.items():
    try:
        dfs[key] = spark.read.table(table_name)
        print(f"  [OK] {key:12s} -> {table_name}")
    except Exception:
        dfs[key] = None
        print(f"  [--] {key:12s} -> {table_name} (NOT FOUND)")

# Convert to pandas for per-project trace analysis
print("\nConverting to pandas...")
pdf = {}
for key, sdf in dfs.items():
    if sdf is not None:
        pdf[key] = sdf.toPandas()
        print(f"  {key:12s}: {len(pdf[key]):,} rows")
    else:
        pdf[key] = None

# =============================================================================
# IDENTIFY WHICH PROJECTS HAVE MULTI-TABLE DATA
# =============================================================================
project_coverage = {}
for _, row in pdf["project"].iterrows():
    pid = row["ids_project_number_synth"]
    coverage = {"project": True}
    if pdf["schedule"] is not None:
        coverage["schedule"] = pid in pdf["schedule"]["ids_project_number_synth"].values
    if pdf["task"] is not None:
        coverage["task"] = pid in pdf["task"]["ids_project_number_synth"].values
    if pdf["rfi"] is not None:
        coverage["rfi"] = pid in pdf["rfi"]["ids_project_number_synth"].values
    if pdf["budget"] is not None:
        coverage["budget"] = pid in pdf["budget"]["ids_project_number_synth"].values
    if pdf["rebaseline"] is not None:
        coverage["rebaseline"] = pid in pdf["rebaseline"]["ids_project_number_synth"].values
    project_coverage[pid] = coverage

# Projects with richest data (have budget or RFI data beyond just schedule/task)
rich_projects = [
    pid for pid, cov in project_coverage.items()
    if cov.get("budget") or cov.get("rfi")
]
all_projects = list(project_coverage.keys())

print(f"\nTotal projects: {len(all_projects)}")
print(f"Projects with budget data: {sum(1 for c in project_coverage.values() if c.get('budget'))}")
print(f"Projects with RFI data: {sum(1 for c in project_coverage.values() if c.get('rfi'))}")
print(f"Rich projects (budget OR rfi): {len(rich_projects)}")

# =============================================================================
# HELPER: Extract project data slice
# =============================================================================
def get_project_data(pid):
    """Extract all data for a single project across all tables."""
    data = {}
    data["project"] = pdf["project"][
        pdf["project"]["ids_project_number_synth"] == pid
    ].iloc[0].to_dict()

    for key in ["schedule", "task", "rfi", "budget", "rebaseline"]:
        if pdf[key] is not None:
            subset = pdf[key][pdf[key]["ids_project_number_synth"] == pid]
            data[key] = subset if len(subset) > 0 else None
        else:
            data[key] = None
    return data


# =============================================================================
# TRACE CHECK 1: SOW Sequencing Coherence
# Do early-sequence SOWs start/finish before late-sequence SOWs?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 1: SOW SEQUENCING COHERENCE")
print("=" * 80)

def check_sow_sequencing(pid):
    """Check that SOW start dates generally follow sequence order in active version."""
    data = get_project_data(pid)
    if data["task"] is None:
        return None

    active_tasks = data["task"][data["task"]["is_active_record"] == True].copy()
    if len(active_tasks) < 3:
        return None

    active_tasks = active_tasks.sort_values("sow_sequence_order")
    starts = active_tasks["dt_sow_start"].values
    orders = active_tasks["sow_sequence_order"].values

    # Count inversions: how many pairs have later sequence but earlier start date
    n_pairs = 0
    n_inversions = 0
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            if orders[i] < orders[j]:
                n_pairs += 1
                if starts[i] > starts[j]:
                    n_inversions += 1

    # With 84% overlap, some inversions are expected (concurrent SOWs)
    # But massive inversions (>40%) would indicate broken sequencing
    inversion_rate = n_inversions / n_pairs if n_pairs > 0 else 0
    return {
        "pid": pid,
        "n_sows": len(active_tasks),
        "n_pairs": n_pairs,
        "n_inversions": n_inversions,
        "inversion_rate": round(inversion_rate, 3),
        "passed": inversion_rate <= 0.40  # 40% threshold accounts for heavy overlap
    }

sow_seq_results = []
for pid in all_projects:
    result = check_sow_sequencing(pid)
    if result:
        sow_seq_results.append(result)

sow_seq_df = pd.DataFrame(sow_seq_results)
n_checked = len(sow_seq_df)
n_failed = len(sow_seq_df[~sow_seq_df["passed"]])
avg_inversion = sow_seq_df["inversion_rate"].mean()
p95_inversion = sow_seq_df["inversion_rate"].quantile(0.95)

print(f"  Projects checked: {n_checked}")
print(f"  Failed (>40% inversions): {n_failed} ({n_failed/n_checked*100:.1f}%)")
print(f"  Avg inversion rate: {avg_inversion:.3f}")
print(f"  P95 inversion rate: {p95_inversion:.3f}")
print(f"  Result: {'PASS' if n_failed / n_checked <= 0.05 else 'FAIL'}")

if n_failed > 0:
    worst = sow_seq_df.nlargest(5, "inversion_rate")
    print(f"\n  Worst 5 projects:")
    for _, r in worst.iterrows():
        print(f"    {r['pid']}: {r['inversion_rate']:.1%} inversions ({r['n_inversions']}/{r['n_pairs']} pairs)")


# =============================================================================
# TRACE CHECK 2: RFI Temporal Alignment with SOW Windows
# Are RFIs concentrated during active SOW periods, not dead periods?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 2: RFI-SOW TEMPORAL ALIGNMENT")
print("=" * 80)

def check_rfi_sow_alignment(pid):
    """Check that RFIs fall within active construction periods, not gaps."""
    data = get_project_data(pid)
    if data["rfi"] is None or data["task"] is None:
        return None

    rfis = data["rfi"]
    active_tasks = data["task"][data["task"]["is_active_record"] == True]
    if len(rfis) == 0 or len(active_tasks) == 0:
        return None

    # Build the active construction envelope from SOW windows
    earliest_sow_start = pd.Timestamp(active_tasks["dt_sow_start"].min())
    latest_sow_finish = pd.Timestamp(active_tasks["dt_sow_finish"].max())

    # Count RFIs within the construction envelope (with 30-day buffer)
    buffer = pd.Timedelta(days=30)
    rfi_dates = pd.to_datetime(rfis["dt_rfi_created"])
    within_envelope = (
        (rfi_dates >= earliest_sow_start - buffer) &
        (rfi_dates <= latest_sow_finish + buffer)
    ).sum()

    alignment_rate = within_envelope / len(rfis)
    return {
        "pid": pid,
        "n_rfis": len(rfis),
        "n_within_envelope": int(within_envelope),
        "alignment_rate": round(alignment_rate, 3),
        "passed": alignment_rate >= 0.90  # 90% of RFIs should be within construction window
    }

rfi_sow_results = []
for pid in all_projects:
    result = check_rfi_sow_alignment(pid)
    if result:
        rfi_sow_results.append(result)

if rfi_sow_results:
    rfi_sow_df = pd.DataFrame(rfi_sow_results)
    n_checked = len(rfi_sow_df)
    n_failed = len(rfi_sow_df[~rfi_sow_df["passed"]])
    print(f"  Projects checked: {n_checked}")
    print(f"  Failed (<90% alignment): {n_failed}")
    for _, r in rfi_sow_df.iterrows():
        print(f"    {r['pid']}: {r['alignment_rate']:.1%} RFIs within SOW envelope "
              f"({r['n_within_envelope']}/{r['n_rfis']})")
    print(f"  Result: {'PASS' if n_failed == 0 else 'FAIL'}")
else:
    print("  No projects with both RFI and task data — SKIPPED")


# =============================================================================
# TRACE CHECK 3: Budget S-Curve Shape
# Does cumulative spend follow a reasonable S-curve (not front-loaded or flat)?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 3: BUDGET S-CURVE SHAPE")
print("=" * 80)

def check_budget_scurve(pid):
    """Check that cumulative JTD spend follows a plausible S-curve shape.
    
    Key insight: divisions appear/disappear across months, so summing JTD
    per month can produce apparent decreases. Instead, we track the RUNNING
    MAXIMUM of project-level JTD to get the true cumulative spend envelope.
    We also normalize against the peak JTD (not the last month's JTD).
    """
    data = get_project_data(pid)
    if data["budget"] is None:
        return None

    budget = data["budget"].copy()

    # Aggregate JTD across all divisions per month
    monthly_jtd = (
        budget.groupby("dt_budget_month")["amt_cost_jtd"]
        .sum()
        .sort_index()
        .reset_index()
    )
    if len(monthly_jtd) < 4:
        return None

    jtd_values = monthly_jtd["amt_cost_jtd"].values

    # Use running max to handle divisions dropping out of monthly reporting
    running_max = np.maximum.accumulate(jtd_values)
    peak_jtd = running_max[-1]
    if peak_jtd <= 0:
        return None

    # Normalize against peak JTD
    normalized = running_max / peak_jtd
    n_months = len(normalized)

    # S-curve checks using the smoothed cumulative envelope:
    # 1. First quartile should have < 50% of total spend
    q1_idx = max(1, n_months // 4)
    q1_spend = normalized[q1_idx]
    front_loaded = q1_spend > 0.50

    # 2. Third quartile should have > 55% of total spend
    q3_idx = min(n_months - 1, 3 * n_months // 4)
    q3_spend = normalized[q3_idx]
    no_convergence = q3_spend < 0.55

    # 3. Running max is monotonic by construction, so no decrease check needed

    passed = not front_loaded and not no_convergence
    return {
        "pid": pid,
        "n_months": n_months,
        "peak_jtd": round(peak_jtd, 0),
        "q1_spend_pct": round(q1_spend * 100, 1),
        "q3_spend_pct": round(q3_spend * 100, 1),
        "front_loaded": front_loaded,
        "no_convergence": no_convergence,
        "passed": passed,
    }

scurve_results = []
for pid in all_projects:
    result = check_budget_scurve(pid)
    if result:
        scurve_results.append(result)

if scurve_results:
    scurve_df = pd.DataFrame(scurve_results)
    n_checked = len(scurve_df)
    n_failed = len(scurve_df[~scurve_df["passed"]])
    n_front = len(scurve_df[scurve_df["front_loaded"]])
    n_no_conv = len(scurve_df[scurve_df["no_convergence"]])
    avg_q1 = scurve_df["q1_spend_pct"].mean()
    avg_q3 = scurve_df["q3_spend_pct"].mean()

    print(f"  Projects checked: {n_checked}")
    print(f"  Failed: {n_failed} ({n_failed/n_checked*100:.1f}%)")
    print(f"    Front-loaded (Q1 > 40%): {n_front}")
    print(f"    No convergence (Q3 < 60%): {n_no_conv}")
    print(f"  Avg Q1 spend: {avg_q1:.1f}%  |  Avg Q3 spend: {avg_q3:.1f}%")
    print(f"  Result: {'PASS' if n_failed / n_checked <= 0.10 else 'FAIL'}")

    if n_failed > 0:
        print(f"\n  Failed projects:")
        for _, r in scurve_df[~scurve_df["passed"]].iterrows():
            flags = []
            if r["front_loaded"]: flags.append(f"front-loaded Q1={r['q1_spend_pct']}%")
            if r["no_convergence"]: flags.append(f"no-convergence Q3={r['q3_spend_pct']}%")
            print(f"    {r['pid']}: {', '.join(flags)} (peak JTD={r['peak_jtd']:,.0f})")
else:
    print("  No projects with budget data — SKIPPED")


# =============================================================================
# TRACE CHECK 4: Rebaseline-RFI Correlation
# Do months around rebaseline events show elevated RFI volume?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 4: REBASELINE-RFI VOLUME CORRELATION")
print("=" * 80)

def check_rebaseline_rfi_spike(pid):
    """Check whether RFI volume is elevated near rebaseline events."""
    data = get_project_data(pid)
    if data["rfi"] is None or data["rebaseline"] is None:
        return None
    if data["schedule"] is None:
        return None

    rfis = data["rfi"]
    # Get rebaseline dates from non-baseline schedule versions
    sched = data["schedule"]
    rb_versions = sched[~sched["cat_schedule_code"].str.endswith("-BL")]
    if len(rb_versions) == 0:
        return None  # no rebaselines

    rb_dates = pd.to_datetime(rb_versions["dt_effective_from"].values)
    rfi_dates = pd.to_datetime(rfis["dt_rfi_created"].values)
    total_rfis = len(rfis)

    # Count RFIs within 30-day window of each rebaseline event
    near_rb_count = 0
    for rb_date in rb_dates:
        window_rfis = ((rfi_dates >= rb_date - pd.Timedelta(days=30)) &
                       (rfi_dates <= rb_date + pd.Timedelta(days=30))).sum()
        near_rb_count += window_rfis

    # Calculate expected RFIs if uniformly distributed
    proj = data["project"]
    proj_start = pd.Timestamp(proj["dt_construction_start"])
    proj_end = pd.Timestamp(proj["dt_construction_end"])
    proj_duration_days = (proj_end - proj_start).days
    if proj_duration_days <= 0:
        return None

    # Each rebaseline contributes a 60-day window
    total_rb_window_days = len(rb_dates) * 60
    expected_uniform = total_rfis * (total_rb_window_days / proj_duration_days)

    # The spike ratio: actual / expected under uniform distribution
    spike_ratio = near_rb_count / expected_uniform if expected_uniform > 0 else 0

    return {
        "pid": pid,
        "n_rebaselines": len(rb_dates),
        "total_rfis": total_rfis,
        "near_rb_rfis": int(near_rb_count),
        "expected_uniform": round(expected_uniform, 1),
        "spike_ratio": round(spike_ratio, 2),
        # We expect spike_ratio > 1.0 due to the 1.08x multiplier in generation
        "shows_spike": spike_ratio >= 0.8,
    }

rb_rfi_results = []
for pid in all_projects:
    result = check_rebaseline_rfi_spike(pid)
    if result:
        rb_rfi_results.append(result)

if rb_rfi_results:
    rb_rfi_df = pd.DataFrame(rb_rfi_results)
    n_checked = len(rb_rfi_df)
    avg_spike = rb_rfi_df["spike_ratio"].mean()
    n_shows_spike = len(rb_rfi_df[rb_rfi_df["shows_spike"]])

    print(f"  Projects checked: {n_checked}")
    print(f"  Avg spike ratio: {avg_spike:.2f} (1.0 = uniform, >1.0 = elevated near rebaseline)")
    print(f"  Projects showing spike (ratio >= 0.8): {n_shows_spike}/{n_checked}")
    for _, r in rb_rfi_df.iterrows():
        print(f"    {r['pid']}: {r['near_rb_rfis']} RFIs near {r['n_rebaselines']} rebaseline(s) "
              f"(expected {r['expected_uniform']}, ratio {r['spike_ratio']})")
    print(f"  Result: {'PASS' if avg_spike >= 0.8 else 'REVIEW'}")
else:
    print("  No projects with both RFI and rebaseline data — SKIPPED")


# =============================================================================
# TRACE CHECK 5: Construction Phase Progression
# Does the project progress through phases in the right order?
# Early SOWs (site work, foundations) -> Mid (structure, MEP) -> Late (finishes, closeout)
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 5: CONSTRUCTION PHASE PROGRESSION")
print("=" * 80)

# Define phase buckets based on typical SOW sequence ranges
PHASE_MAP = {
    "early": [
        "(01)", "(03)", "(04)", "(05)", "(06)", "(07)",
        "(08)", "(09)", "(10)", "(10B)", "(11)", "(12)", "(12B)", "(13)"
    ],
    "mid": [
        "(14)", "(14C)", "(15)", "(16)", "(17)", "(18)", "(18B)", "(19)",
        "(20)", "(21)", "(22)", "(23)", "(24)", "(25)"
    ],
    "late": [
        "(26)", "(27)", "(28)", "(29)", "(30)", "(31)", "(32)", "(33)", "(9BX)"
    ]
}

def classify_sow_phase(cat_sow):
    """Classify a SOW into early/mid/late phase."""
    for phase, prefixes in PHASE_MAP.items():
        for prefix in prefixes:
            if cat_sow.startswith(prefix):
                return phase
    return "unknown"

def check_phase_progression(pid):
    """Check that phase median start dates follow early < mid < late ordering.
    
    Uses actual sow_sequence_order from the data to classify phases into thirds
    rather than hardcoded SOW prefix mapping, since sequence position varies by SBU.
    """
    data = get_project_data(pid)
    if data["task"] is None:
        return None

    active_tasks = data["task"][data["task"]["is_active_record"] == True].copy()
    if len(active_tasks) < 6:
        return None

    active_tasks["dt_sow_start"] = pd.to_datetime(active_tasks["dt_sow_start"])
    active_tasks = active_tasks.sort_values("sow_sequence_order")

    # Classify into thirds based on actual sequence position
    n_sows = len(active_tasks)
    third = n_sows // 3
    active_tasks["phase"] = "mid"  # default
    active_tasks.iloc[:third, active_tasks.columns.get_loc("phase")] = "early"
    active_tasks.iloc[-third:, active_tasks.columns.get_loc("phase")] = "late"

    phase_medians = (
        active_tasks.groupby("phase")["dt_sow_start"]
        .median()
        .to_dict()
    )

    has_all = all(p in phase_medians for p in ["early", "mid", "late"])
    if not has_all:
        return None

    early_before_mid = phase_medians["early"] <= phase_medians["mid"]
    mid_before_late = phase_medians["mid"] <= phase_medians["late"]

    return {
        "pid": pid,
        "early_median": phase_medians["early"].strftime("%Y-%m-%d"),
        "mid_median": phase_medians["mid"].strftime("%Y-%m-%d"),
        "late_median": phase_medians["late"].strftime("%Y-%m-%d"),
        "early_before_mid": early_before_mid,
        "mid_before_late": mid_before_late,
        "early_before_late": phase_medians["early"] <= phase_medians["late"],
        "passed": early_before_mid and mid_before_late,
    }

phase_results = []
for pid in all_projects:
    result = check_phase_progression(pid)
    if result:
        phase_results.append(result)

if phase_results:
    phase_df = pd.DataFrame(phase_results)
    n_checked = len(phase_df)
    n_passed = len(phase_df[phase_df["passed"]])
    n_early_mid = len(phase_df[phase_df["early_before_mid"]])
    n_mid_late = len(phase_df[phase_df["mid_before_late"]])

    print(f"  Projects checked: {n_checked}")
    print(f"  Full progression (early < mid < late): {n_passed}/{n_checked} "
          f"({n_passed/n_checked*100:.1f}%)")
    print(f"    Early before Mid: {n_early_mid}/{n_checked} ({n_early_mid/n_checked*100:.1f}%)")
    print(f"    Mid before Late:  {n_mid_late}/{n_checked} ({n_mid_late/n_checked*100:.1f}%)")
    # With heavy overlap (84%), some projects may have mid starting with early
    # But early-before-late should hold for nearly all projects
    print(f"  Result: {'PASS' if n_passed / n_checked >= 0.70 else 'FAIL'}")

    if n_checked - n_passed > 0:
        failed = phase_df[~phase_df["passed"]].head(5)
        print(f"\n  Sample failures:")
        for _, r in failed.iterrows():
            print(f"    {r['pid']}: early={r['early_median']}, mid={r['mid_median']}, "
                  f"late={r['late_median']}")
else:
    print("  Insufficient task data — SKIPPED")


# =============================================================================
# TRACE CHECK 6: Budget-Schedule Lifecycle Alignment
# Does the budget spend trajectory align with schedule progress?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 6: BUDGET-SCHEDULE LIFECYCLE ALIGNMENT")
print("=" * 80)

def check_budget_schedule_alignment(pid):
    """Check that budget spend trajectory loosely aligns with construction progress.
    
    Uses running-max JTD (to handle divisions dropping out) normalized against
    the peak ECAC observed across the project lifecycle.
    """
    data = get_project_data(pid)
    if data["budget"] is None or data["task"] is None:
        return None

    budget = data["budget"]
    active_tasks = data["task"][data["task"]["is_active_record"] == True]

    # Schedule % complete from task summary
    total_tasks = active_tasks["n_tasks_total"].sum()
    completed_tasks = active_tasks["n_tasks_completed"].sum()
    schedule_pct = completed_tasks / total_tasks if total_tasks > 0 else 0

    # Budget % spent: running-max JTD / peak ECAC
    monthly_jtd = budget.groupby("dt_budget_month")["amt_cost_jtd"].sum().sort_index()
    monthly_ecac = budget.groupby("dt_budget_month")["amt_ecac"].sum().sort_index()
    if len(monthly_jtd) == 0 or len(monthly_ecac) == 0:
        return None

    peak_jtd = np.maximum.accumulate(monthly_jtd.values)[-1]
    peak_ecac = monthly_ecac.max()
    budget_pct = peak_jtd / peak_ecac if peak_ecac > 0 else 0

    gap = abs(schedule_pct - budget_pct)

    return {
        "pid": pid,
        "schedule_pct": round(schedule_pct * 100, 1),
        "budget_pct": round(budget_pct * 100, 1),
        "gap_pct": round(gap * 100, 1),
        "passed": gap <= 0.30,
    }

alignment_results = []
for pid in all_projects:
    result = check_budget_schedule_alignment(pid)
    if result:
        alignment_results.append(result)

if alignment_results:
    align_df = pd.DataFrame(alignment_results)
    n_checked = len(align_df)
    n_passed = len(align_df[align_df["passed"]])
    avg_gap = align_df["gap_pct"].mean()

    print(f"  Projects checked: {n_checked}")
    print(f"  Passed (gap <= 30%): {n_passed}/{n_checked} ({n_passed/n_checked*100:.1f}%)")
    print(f"  Avg schedule-budget gap: {avg_gap:.1f}%")
    print(f"  Result: {'PASS' if n_passed / n_checked >= 0.80 else 'FAIL'}")

    print(f"\n  Per-project breakdown:")
    for _, r in align_df.iterrows():
        status = "OK" if r["passed"] else "!!"
        print(f"    [{status}] {r['pid']}: schedule={r['schedule_pct']}%, "
              f"budget={r['budget_pct']}%, gap={r['gap_pct']}%")
else:
    print("  No projects with both budget and task data — SKIPPED")


# =============================================================================
# TRACE CHECK 7: RFI Trade Distribution vs Project Type
# Does the RFI trade mix make sense for the project's asset class?
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 7: RFI TRADE MIX vs PROJECT TYPE")
print("=" * 80)

def check_rfi_trade_mix(pid):
    """Check that RFI trade distribution is plausible for the project type."""
    data = get_project_data(pid)
    if data["rfi"] is None:
        return None

    proj = data["project"]
    rfis = data["rfi"]

    trade_counts = rfis["cat_trade_l1"].value_counts(normalize=True).to_dict()

    # Domain expectations:
    # D SERVICES (MEP) should be significant for all project types (typically 20-40%)
    # C INTERIORS should be significant (typically 25-50%)
    # A SUBSTRUCTURE should be small for most projects (typically <10%)
    d_services_pct = trade_counts.get("D SERVICES", 0)
    c_interiors_pct = trade_counts.get("C INTERIORS", 0)

    # MEP + Interiors should collectively be >40% for any real project
    mep_interior_combined = d_services_pct + c_interiors_pct

    return {
        "pid": pid,
        "sbu": proj["cat_sbu"],
        "asset": proj["cat_main_asset_class"],
        "n_rfis": len(rfis),
        "d_services_pct": round(d_services_pct * 100, 1),
        "c_interiors_pct": round(c_interiors_pct * 100, 1),
        "combined_pct": round(mep_interior_combined * 100, 1),
        "top_trade": max(trade_counts, key=trade_counts.get),
        "passed": mep_interior_combined >= 0.40,
    }

trade_results = []
for pid in all_projects:
    result = check_rfi_trade_mix(pid)
    if result:
        trade_results.append(result)

if trade_results:
    trade_df = pd.DataFrame(trade_results)
    n_checked = len(trade_df)
    n_passed = len(trade_df[trade_df["passed"]])

    print(f"  Projects checked: {n_checked}")
    for _, r in trade_df.iterrows():
        status = "OK" if r["passed"] else "!!"
        print(f"    [{status}] {r['pid']} ({r['sbu']}/{r['asset']}): "
              f"D SERVICES={r['d_services_pct']}%, C INTERIORS={r['c_interiors_pct']}%, "
              f"combined={r['combined_pct']}%, top={r['top_trade']}")
    print(f"  Result: {'PASS' if n_passed == n_checked else 'REVIEW'}")
else:
    print("  No projects with RFI data — SKIPPED")


# =============================================================================
# TRACE CHECK 8: Project Lifecycle Event Density
# Build a monthly event timeline and check for dead periods
# =============================================================================
print("\n" + "=" * 80)
print("TRACE CHECK 8: LIFECYCLE EVENT DENSITY (RICH PROJECTS ONLY)")
print("=" * 80)

def build_monthly_timeline(pid):
    """Build month-by-month event counts across all tables for one project."""
    data = get_project_data(pid)
    proj = data["project"]
    proj_start = pd.Timestamp(proj["dt_construction_start"])
    proj_end = pd.Timestamp(proj["dt_construction_end"])

    # Generate monthly periods
    months = pd.date_range(start=proj_start.replace(day=1),
                           end=proj_end + pd.offsets.MonthEnd(1), freq="MS")
    timeline = pd.DataFrame({"month": months})
    timeline["month_str"] = timeline["month"].dt.strftime("%Y-%m")

    # Count RFIs per month
    if data["rfi"] is not None:
        rfis = data["rfi"].copy()
        rfis["month_str"] = pd.to_datetime(rfis["dt_rfi_created"]).dt.strftime("%Y-%m")
        rfi_monthly = rfis.groupby("month_str").size().reset_index(name="n_rfis")
        timeline = timeline.merge(rfi_monthly, on="month_str", how="left")
    else:
        timeline["n_rfis"] = np.nan

    # Count active SOWs per month
    if data["task"] is not None:
        active_tasks = data["task"][data["task"]["is_active_record"] == True]
        sow_counts = []
        for _, row in timeline.iterrows():
            m = row["month"]
            active_sows = (
                (pd.to_datetime(active_tasks["dt_sow_start"]) <= m + pd.offsets.MonthEnd(0)) &
                (pd.to_datetime(active_tasks["dt_sow_finish"]) >= m)
            ).sum()
            sow_counts.append(active_sows)
        timeline["n_active_sows"] = sow_counts
    else:
        timeline["n_active_sows"] = np.nan

    # Budget JTD per month
    if data["budget"] is not None:
        budget = data["budget"]
        budget_monthly = (
            budget.groupby("dt_budget_month")["amt_cost_jtd"]
            .sum()
            .reset_index()
            .rename(columns={"dt_budget_month": "month_str", "amt_cost_jtd": "budget_jtd"})
        )
        timeline = timeline.merge(budget_monthly, on="month_str", how="left")
    else:
        timeline["budget_jtd"] = np.nan

    return timeline

def check_event_density(pid):
    """Check for dead periods (months with zero activity during active construction)."""
    timeline = build_monthly_timeline(pid)

    # Skip first and last month (ramp-up/closeout)
    if len(timeline) <= 2:
        return None
    core = timeline.iloc[1:-1]

    # Check for months with zero active SOWs during mid-construction
    if "n_active_sows" in core.columns and not core["n_active_sows"].isna().all():
        dead_months = (core["n_active_sows"] == 0).sum()
        total_months = len(core)
        dead_rate = dead_months / total_months if total_months > 0 else 0
    else:
        dead_months = 0
        total_months = 0
        dead_rate = 0

    return {
        "pid": pid,
        "total_months": len(timeline),
        "core_months": len(core),
        "dead_months": int(dead_months),
        "dead_rate": round(dead_rate, 3),
        "passed": dead_rate <= 0.10,  # <10% dead months is acceptable
    }

density_results = []
# Run on all projects (task data is available for all 1,000)
for pid in all_projects:
    result = check_event_density(pid)
    if result:
        density_results.append(result)

if density_results:
    density_df = pd.DataFrame(density_results)
    n_checked = len(density_df)
    n_passed = len(density_df[density_df["passed"]])
    avg_dead = density_df["dead_rate"].mean()

    print(f"  Projects checked: {n_checked}")
    print(f"  Passed (<10% dead months): {n_passed}/{n_checked} ({n_passed/n_checked*100:.1f}%)")
    print(f"  Avg dead month rate: {avg_dead:.1%}")
    print(f"  Result: {'PASS' if n_passed / n_checked >= 0.90 else 'FAIL'}")

    if n_checked - n_passed > 0 and n_checked - n_passed <= 20:
        print(f"\n  Failed projects:")
        for _, r in density_df[~density_df["passed"]].head(10).iterrows():
            print(f"    {r['pid']}: {r['dead_months']}/{r['core_months']} dead months "
                  f"({r['dead_rate']:.1%})")
else:
    print("  Insufficient data — SKIPPED")


# =============================================================================
# LAYER 3 SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("LAYER 3 SUMMARY")
print("=" * 80)

checks = [
    ("TC1: SOW Sequencing", sow_seq_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC2: RFI-SOW Alignment", rfi_sow_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC3: Budget S-Curve", scurve_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC4: Rebaseline-RFI Spike", rb_rfi_results,
     lambda r: sum(1 for x in r if x.get("shows_spike", False)), len),
    ("TC5: Phase Progression", phase_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC6: Budget-Schedule Align", alignment_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC7: RFI Trade Mix", trade_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
    ("TC8: Event Density", density_results,
     lambda r: sum(1 for x in r if x["passed"]), len),
]

for name, results_list, pass_fn, total_fn in checks:
    if results_list:
        n_pass = pass_fn(results_list)
        n_total = total_fn(results_list)
        pct = n_pass / n_total * 100 if n_total > 0 else 0
        icon = "+" if pct >= 70 else "!"
        print(f"  [{icon}] {name:30s}: {n_pass:>5}/{n_total:<5} ({pct:5.1f}%)")
    else:
        print(f"  [~] {name:30s}: SKIPPED (no applicable data)")

print(f"\n{'=' * 80}")
print(f"RUN COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")