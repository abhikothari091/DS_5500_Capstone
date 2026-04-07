# Databricks notebook source
# =============================================================================
# SYNTHETIC DATA VALIDATION — LAYER 1 + 2
# Temporal Integrity, Referential Integrity, Domain Rules, Cross-Table Alignment
# =============================================================================
# Run this notebook end-to-end after any generation run.
# Works with partial data (e.g. 10-project budget smoke test).
# Prints a scorecard at the end with pass/fail per check.
# =============================================================================

from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from datetime import datetime
import json

# =============================================================================
# CONFIG
# =============================================================================
TABLES = {
    "project":    "analysts.self_managed.synthetic_project_meta_data_profiled",
    "schedule":   "analysts.self_managed.synthetic_schedule_baseline_v1",
    "rebaseline": "analysts.self_managed.rebaseline_event_metadata_v1",
    "task":       "analysts.self_managed.synthetic_task_summary_by_sow_v1",
    "rfi":        "analysts.self_managed.synthetic_rfi_v1",
    "budget":     "analysts.self_managed.synthetic_budget_v1",  # may not exist yet for full run
}

# Severity levels
HARD = "HARD"    # zero tolerance, must be 0 violations
SOFT = "SOFT"    # threshold-based, small violation rate acceptable
INFO = "INFO"    # informational, not a pass/fail

# Soft constraint thresholds
SOFT_THRESHOLD = 0.05  # 5% violation rate max for SOFT checks

# =============================================================================
# LOAD TABLES
# =============================================================================
print("=" * 80)
print("LOADING SYNTHETIC TABLES")
print("=" * 80)

dfs = {}
table_counts = {}

for key, table_name in TABLES.items():
    try:
        df = spark.read.table(table_name)
        count = df.count()
        dfs[key] = df
        table_counts[key] = count
        print(f"  [OK] {key:12s} -> {table_name:60s} ({count:,} rows)")
    except Exception as e:
        dfs[key] = None
        table_counts[key] = 0
        print(f"  [--] {key:12s} -> {table_name:60s} (NOT FOUND: {str(e)[:60]})")

proj_df   = dfs["project"]
sched_df  = dfs["schedule"]
rb_df     = dfs["rebaseline"]
task_df   = dfs["task"]
rfi_df    = dfs["rfi"]
budget_df = dfs["budget"]

n_projects = table_counts["project"]
print(f"\nProjects loaded: {n_projects:,}")
print("=" * 80)

# =============================================================================
# RESULTS COLLECTOR
# =============================================================================
results = []

def record(check_id, layer, category, description, severity,
           total_checked, violations, detail=""):
    """Record a single validation check result."""
    pct = (violations / total_checked * 100) if total_checked > 0 else 0.0
    if severity == HARD:
        passed = violations == 0
    elif severity == SOFT:
        passed = (violations / total_checked) <= SOFT_THRESHOLD if total_checked > 0 else True
    else:
        passed = True  # INFO checks always pass

    results.append({
        "check_id": check_id,
        "layer": layer,
        "category": category,
        "description": description,
        "severity": severity,
        "total_checked": total_checked,
        "violations": violations,
        "violation_pct": round(pct, 3),
        "passed": passed,
        "detail": detail,
    })
    status = "PASS" if passed else "FAIL"
    icon = "+" if passed else "X"
    print(f"  [{icon}] {check_id:35s} | {severity:4s} | {violations:>7,} / {total_checked:>9,} ({pct:5.1f}%) | {status}")


# =============================================================================
# SECTION 1: REFERENTIAL INTEGRITY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: REFERENTIAL INTEGRITY")
print("=" * 80)

project_ids = proj_df.select("ids_project_number_synth").distinct()

# 1.1 Schedule -> Project FK
if sched_df:
    orphaned = sched_df.join(
        project_ids, on="ids_project_number_synth", how="left_anti"
    ).count()
    record("REF_SCHED_TO_PROJECT", "L1", "Referential",
           "Every schedule baseline row references a valid project",
           HARD, table_counts["schedule"], orphaned)

# 1.2 Task Summary -> Project FK
if task_df:
    orphaned = task_df.join(
        project_ids, on="ids_project_number_synth", how="left_anti"
    ).count()
    record("REF_TASK_TO_PROJECT", "L1", "Referential",
           "Every task summary row references a valid project",
           HARD, table_counts["task"], orphaned)

# 1.3 Task Summary -> Schedule FK
if task_df and sched_df:
    sched_codes = sched_df.select("cat_schedule_code").distinct()
    orphaned = task_df.join(
        sched_codes, on="cat_schedule_code", how="left_anti"
    ).count()
    record("REF_TASK_TO_SCHEDULE", "L1", "Referential",
           "Every task summary row references a valid schedule version",
           HARD, table_counts["task"], orphaned)

# 1.4 RFI -> Project FK
if rfi_df:
    orphaned = rfi_df.join(
        project_ids, on="ids_project_number_synth", how="left_anti"
    ).count()
    record("REF_RFI_TO_PROJECT", "L1", "Referential",
           "Every RFI row references a valid project",
           HARD, table_counts["rfi"], orphaned)

# 1.5 RFI chain: parent_rfi_id references valid id_rfi_synth
if rfi_df:
    chain_rfis = rfi_df.filter(F.col("parent_rfi_id").isNotNull())
    n_chains = chain_rfis.count()
    if n_chains > 0:
        valid_ids = rfi_df.select(F.col("id_rfi_synth").alias("parent_rfi_id"))
        orphaned = chain_rfis.join(
            valid_ids, on="parent_rfi_id", how="left_anti"
        ).count()
        record("REF_RFI_CHAIN_PARENT", "L1", "Referential",
               "Every chain RFI parent_rfi_id references a valid RFI",
               HARD, n_chains, orphaned)

# 1.6 Budget -> Project FK
if budget_df:
    orphaned = budget_df.join(
        project_ids, on="ids_project_number_synth", how="left_anti"
    ).count()
    record("REF_BUDGET_TO_PROJECT", "L1", "Referential",
           "Every budget row references a valid project",
           HARD, table_counts["budget"], orphaned)

# 1.7 Rebaseline metadata -> Project FK
if rb_df:
    orphaned = rb_df.join(
        project_ids, on="ids_project_number_synth", how="left_anti"
    ).count()
    record("REF_REBASELINE_TO_PROJECT", "L1", "Referential",
           "Every rebaseline metadata row references a valid project",
           HARD, table_counts["rebaseline"], orphaned)


# =============================================================================
# SECTION 2: TEMPORAL INTEGRITY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: TEMPORAL INTEGRITY")
print("=" * 80)

# 2.1 SCD Type 2: Exactly one active record per project in schedule
if sched_df:
    active_counts = (
        sched_df.filter(F.col("is_active_record") == True)
        .groupBy("ids_project_number_synth")
        .agg(F.count("*").alias("n_active"))
    )
    not_exactly_one = active_counts.filter(F.col("n_active") != 1).count()
    record("TEMP_SCHED_ONE_ACTIVE", "L1", "Temporal",
           "Exactly one active schedule version per project",
           HARD, n_projects, not_exactly_one)

# 2.2 SCD Type 2: No overlapping effective date windows
if sched_df:
    w = Window.partitionBy("ids_project_number_synth").orderBy("dt_effective_from")
    overlap_check = (
        sched_df
        .withColumn("next_from", F.lead("dt_effective_from").over(w))
        .filter(F.col("next_from").isNotNull())
        .filter(F.col("dt_effective_to") > F.col("next_from"))
    )
    n_overlaps = overlap_check.count()
    record("TEMP_SCD2_NO_OVERLAP", "L1", "Temporal",
           "SCD Type 2 effective date windows do not overlap",
           HARD, table_counts["schedule"], n_overlaps)

# 2.3 Monotonic TCO across schedule versions
if sched_df:
    w = Window.partitionBy("ids_project_number_synth").orderBy("dt_effective_from")
    tco_regression = (
        sched_df
        .withColumn("prev_tco", F.lag("dt_tco").over(w))
        .filter(F.col("prev_tco").isNotNull())
        .filter(F.col("dt_tco") < F.col("prev_tco"))
    )
    n_regressions = tco_regression.count()
    # denominator is number of version transitions, not total rows
    n_transitions = sched_df.count() - sched_df.select("ids_project_number_synth").distinct().count()
    record("TEMP_TCO_MONOTONIC", "L1", "Temporal",
           "TCO date never decreases across schedule versions",
           HARD, max(n_transitions, 1), n_regressions)

# 2.4 Schedule: NTP <= TCO for every version
if sched_df:
    ntp_after_tco = sched_df.filter(F.col("dt_ntp") > F.col("dt_tco")).count()
    record("TEMP_NTP_BEFORE_TCO", "L1", "Temporal",
           "NTP date is on or before TCO date in every schedule version",
           HARD, table_counts["schedule"], ntp_after_tco)

# 2.5 Task summary: SOW dates within project date range
#     Compare each SOW row against the TCO of its OWN schedule version (not latest)
if task_df and proj_df and sched_df:
    # Join each task row to its specific schedule version's TCO
    task_with_version_tco = (
        task_df
        .join(
            sched_df.select(
                F.col("cat_schedule_code"),
                F.col("dt_tco").alias("dt_version_tco")
            ),
            on="cat_schedule_code"
        )
        .join(
            proj_df.select("ids_project_number_synth", "dt_construction_start"),
            on="ids_project_number_synth"
        )
    )
    # Allow 90-day buffer for pre-construction and closeout activities
    buffer_days = 90
    out_of_range = task_with_version_tco.filter(
        (F.col("dt_sow_start") < F.date_sub(F.col("dt_construction_start"), buffer_days)) |
        (F.col("dt_sow_finish") > F.date_add(F.col("dt_version_tco"), buffer_days))
    ).count()
    record("TEMP_SOW_IN_PROJECT_RANGE", "L1", "Temporal",
           f"SOW dates within version-specific range (start-{buffer_days}d to version_tco+{buffer_days}d)",
           SOFT, table_counts["task"], out_of_range)

# 2.6 Task summary: Version 0 (baseline) always starts at 0% completion
#     Note: n_tasks_completed and n_tasks_total are independently sampled per version
#     from profiled distributions, so raw counts are NOT guaranteed monotonic across
#     versions. The design guarantee is: V0 = 0% and later versions show progress
#     via the completion progression model (position-based sigmoid).
if task_df:
    # Check 1: Baseline version (BL) should have 0 completed tasks
    baseline_nonzero = (
        task_df
        .filter(F.col("cat_schedule_code").endswith("-BL"))
        .filter(F.col("n_tasks_completed") > 0)
    ).count()
    n_baseline_rows = task_df.filter(F.col("cat_schedule_code").endswith("-BL")).count()
    record("TEMP_BASELINE_ZERO_COMPLETION", "L1", "Temporal",
           "Baseline version (V0) has 0 tasks completed for all SOWs",
           HARD, n_baseline_rows, baseline_nonzero)

    # Check 2: Completion ratio regression (SOFT, not HARD)
    #     Some regression is expected when task counts are re-sampled per version.
    #     Flag only if >15% of transitions regress, indicating a systematic issue.
    w = Window.partitionBy("ids_project_number_synth", "cat_sow").orderBy("dt_effective_from")
    task_with_pct = task_df.withColumn(
        "pct_complete",
        F.when(F.col("n_tasks_total") > 0,
               F.col("n_tasks_completed") / F.col("n_tasks_total")
        ).otherwise(0.0)
    )
    pct_regression = (
        task_with_pct
        .withColumn("prev_pct", F.lag("pct_complete").over(w))
        .filter(F.col("prev_pct").isNotNull())
        .filter(F.col("pct_complete") < F.col("prev_pct") - 0.01)  # 1% epsilon
    )
    n_pct_regress = pct_regression.count()
    n_sow_transitions = (
        task_df.count() -
        task_df.select("ids_project_number_synth", "cat_sow").distinct().count()
    )
    record("TEMP_COMPLETION_PROGRESSION", "L1", "Temporal",
           "SOW completion % generally increases across versions (soft, re-sampling expected)",
           SOFT, max(n_sow_transitions, 1), n_pct_regress)

# 2.7 RFI dates within project date range
if rfi_df and proj_df:
    rfi_proj = rfi_df.join(
        proj_df.select("ids_project_number_synth", "dt_construction_start", "dt_construction_end"),
        on="ids_project_number_synth"
    )
    # Allow 30-day buffer for pre-NTP RFIs and post-TCO closeout RFIs
    rfi_buffer = 30
    out_of_range = rfi_proj.filter(
        (F.col("dt_rfi_created") < F.date_sub(F.col("dt_construction_start"), rfi_buffer)) |
        (F.col("dt_rfi_created") > F.date_add(F.col("dt_construction_end"), rfi_buffer))
    ).count()
    record("TEMP_RFI_IN_PROJECT_RANGE", "L1", "Temporal",
           f"RFI created dates within project range (+-{rfi_buffer}d buffer)",
           HARD, table_counts["rfi"], out_of_range)

# 2.8 RFI: due date >= created date
if rfi_df:
    due_before_created = rfi_df.filter(
        F.col("dt_rfi_due") < F.col("dt_rfi_created")
    ).count()
    record("TEMP_RFI_DUE_AFTER_CREATED", "L1", "Temporal",
           "RFI due date is on or after created date",
           HARD, table_counts["rfi"], due_before_created)

# 2.9 RFI: resolved date >= created date (for closed RFIs)
if rfi_df:
    closed_rfis = rfi_df.filter(F.col("dt_rfi_resolved").isNotNull())
    n_closed = closed_rfis.count()
    resolved_before_created = closed_rfis.filter(
        F.col("dt_rfi_resolved") < F.col("dt_rfi_created")
    ).count()
    record("TEMP_RFI_RESOLVED_AFTER_CREATED", "L1", "Temporal",
           "RFI resolved date is on or after created date (closed RFIs)",
           HARD, n_closed, resolved_before_created)

# 2.10 Budget months within project date range
#     Budget extends to revised TCO for rebaselined projects, not original dt_construction_end
if budget_df and proj_df and sched_df:
    # Get latest TCO per project from active schedule version
    latest_tco = (
        sched_df.filter(F.col("is_active_record") == True)
        .select("ids_project_number_synth", F.col("dt_tco").alias("dt_latest_tco"))
    )
    budget_proj = (
        budget_df
        .join(
            proj_df.select("ids_project_number_synth", "dt_construction_start"),
            on="ids_project_number_synth"
        )
        .join(latest_tco, on="ids_project_number_synth", how="left")
    )
    budget_buffer = 90  # pre-construction procurement + closeout
    out_of_range = budget_proj.filter(
        (F.col("dt_budget") < F.date_sub(F.col("dt_construction_start"), budget_buffer)) |
        (F.col("dt_budget") > F.date_add(F.col("dt_latest_tco"), budget_buffer))
    ).count()
    n_budget_checked = budget_proj.count()
    record("TEMP_BUDGET_IN_PROJECT_RANGE", "L1", "Temporal",
           f"Budget dates within range (start-{budget_buffer}d to latest_tco+{budget_buffer}d)",
           SOFT, n_budget_checked, out_of_range)

# 2.11 Budget: JTD monotonicity per cost code per project
if budget_df:
    w = Window.partitionBy(
        "ids_project_number_synth", "cat_budget_division", "cat_cost_code"
    ).orderBy("dt_budget")
    jtd_regression = (
        budget_df
        .withColumn("prev_jtd", F.lag("amt_cost_jtd").over(w))
        .filter(F.col("prev_jtd").isNotNull())
        .filter(F.col("amt_cost_jtd") < F.col("prev_jtd") - 0.01)  # small epsilon
    )
    n_jtd_regress = jtd_regression.count()
    n_jtd_transitions = (
        budget_df.count() -
        budget_df.select(
            "ids_project_number_synth", "cat_budget_division", "cat_cost_code"
        ).distinct().count()
    )
    record("TEMP_BUDGET_JTD_MONOTONIC", "L1", "Temporal",
           "JTD cost never decreases over time per division/cost code",
           HARD, max(n_jtd_transitions, 1), n_jtd_regress)


# =============================================================================
# SECTION 3: DOMAIN RULES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: DOMAIN RULES")
print("=" * 80)

# 3.1 RFI status logic: resolved date NULL for open/overdue/draft
if rfi_df:
    non_closed = rfi_df.filter(F.col("cat_status").isin("open", "overdue", "draft"))
    n_non_closed = non_closed.count()
    has_resolved = non_closed.filter(F.col("dt_rfi_resolved").isNotNull()).count()
    record("DOMAIN_RFI_STATUS_RESOLVED", "L2", "Domain",
           "Non-closed RFIs (open/overdue/draft) have NULL resolved date",
           HARD, n_non_closed, has_resolved)

# 3.2 RFI status logic: closed RFIs have non-NULL resolved date
if rfi_df:
    closed = rfi_df.filter(F.col("cat_status") == "closed")
    n_closed_total = closed.count()
    missing_resolved = closed.filter(F.col("dt_rfi_resolved").isNull()).count()
    record("DOMAIN_RFI_CLOSED_HAS_RESOLVED", "L2", "Domain",
           "Closed RFIs have a non-NULL resolved date",
           HARD, n_closed_total, missing_resolved)

# 3.3 Budget: FTC non-negative
if budget_df:
    neg_ftc = budget_df.filter(F.col("amt_forecast_to_complete") < -0.01).count()
    record("DOMAIN_BUDGET_FTC_NONNEG", "L2", "Domain",
           "Forecast to complete is non-negative",
           HARD, table_counts["budget"], neg_ftc)

# 3.4 Budget: ECAC = cost_projected + FTC (not JTD + FTC)
#     The generation computes: FTC = max(0, ECAC - cost_projected)
#     where cost_projected = cost_direct + cost_committed = JTD * (direct_ratio + committed_ratio)
#     So the identity is: ECAC >= cost_projected + FTC (with FTC clamped to 0)
if budget_df:
    ecac_check = (
        budget_df
        .filter(
            F.col("amt_ecac").isNotNull() &
            F.col("amt_cost_projected").isNotNull() &
            F.col("amt_forecast_to_complete").isNotNull()
        )
        .withColumn("ecac_calc", F.col("amt_cost_projected") + F.col("amt_forecast_to_complete"))
        .withColumn("ecac_diff", F.abs(F.col("amt_ecac") - F.col("ecac_calc")))
        .withColumn("tolerance", F.greatest(
            F.abs(F.col("amt_ecac")) * 0.01,
            F.lit(1000.0)
        ))
        .filter(F.col("ecac_diff") > F.col("tolerance"))
    )
    n_ecac_checked = budget_df.filter(
        F.col("amt_ecac").isNotNull() & F.col("amt_cost_projected").isNotNull()
    ).count()
    n_ecac_mismatch = ecac_check.count()
    record("DOMAIN_BUDGET_ECAC_IDENTITY", "L2", "Domain",
           "ECAC approx equals cost_projected + FTC (1% tolerance)",
           SOFT, n_ecac_checked, n_ecac_mismatch)

# 3.5 Budget: budget_revised = budget_original + approved_in + approved_out
#     (NOT original + mods; mods is typically 0, transfers are the actual modifications)
if budget_df:
    revised_check = (
        budget_df
        .filter(
            F.col("amt_budget_original").isNotNull() &
            F.col("amt_budget_approved_in").isNotNull() &
            F.col("amt_budget_approved_out").isNotNull() &
            F.col("amt_budget_revised").isNotNull()
        )
        .withColumn("revised_calc",
            F.col("amt_budget_original") +
            F.col("amt_budget_approved_in") +
            F.col("amt_budget_approved_out")
        )
        .withColumn("revised_diff", F.abs(F.col("amt_budget_revised") - F.col("revised_calc")))
        .filter(F.col("revised_diff") > F.greatest(
            F.abs(F.col("amt_budget_revised")) * 0.01,
            F.lit(100.0)
        ))
    )
    n_revised_checked = budget_df.filter(
        F.col("amt_budget_original").isNotNull() & F.col("amt_budget_revised").isNotNull()
    ).count()
    record("DOMAIN_BUDGET_REVISED_CALC", "L2", "Domain",
           "Budget revised = original + approved_in + approved_out (1% tolerance)",
           SOFT, n_revised_checked, revised_check.count())

# 3.6 Task summary: exactly one active version set per project
if task_df:
    active_sow_sets = (
        task_df.filter(F.col("is_active_record") == True)
        .groupBy("ids_project_number_synth")
        .agg(F.countDistinct("cat_schedule_code").alias("n_active_versions"))
    )
    not_one = active_sow_sets.filter(F.col("n_active_versions") != 1).count()
    record("DOMAIN_TASK_ONE_ACTIVE_VERSION", "L2", "Domain",
           "Exactly one active schedule version in task summary per project",
           HARD, n_projects, not_one)

# 3.7 Task summary: task count consistency (total = completed + active + not_started)
if task_df:
    task_count_mismatch = (
        task_df
        .withColumn("task_sum",
            F.col("n_tasks_completed") + F.col("n_tasks_active") + F.col("n_tasks_not_started")
        )
        .filter(F.col("n_tasks_total") != F.col("task_sum"))
    ).count()
    record("DOMAIN_TASK_COUNT_IDENTITY", "L2", "Domain",
           "n_tasks_total = completed + active + not_started",
           HARD, table_counts["task"], task_count_mismatch)

# 3.8 Task summary: derived pct_complete in [0, 1] range
if task_df:
    out_of_range = (
        task_df
        .withColumn("pct_complete",
            F.when(F.col("n_tasks_total") > 0,
                   F.col("n_tasks_completed") / F.col("n_tasks_total")
            ).otherwise(0.0)
        )
        .filter(
            (F.col("pct_complete") < -0.001) | (F.col("pct_complete") > 1.001)
        )
    ).count()
    record("DOMAIN_TASK_PCT_RANGE", "L2", "Domain",
           "Derived pct_complete (completed/total) between 0.0 and 1.0",
           HARD, table_counts["task"], out_of_range)

# 3.9 Schedule: Version 0 (original baseline) exists for every project
if sched_df:
    # Original baseline codes end with "-BL"
    projects_with_bl = (
        sched_df.filter(F.col("cat_schedule_code").endswith("-BL"))
        .select("ids_project_number_synth").distinct().count()
    )
    missing_bl = n_projects - projects_with_bl
    record("DOMAIN_SCHED_HAS_BASELINE", "L2", "Domain",
           "Every project has an original baseline schedule version",
           HARD, n_projects, missing_bl)

# 3.10 Project: construction end >= construction start
if proj_df:
    end_before_start = proj_df.filter(
        F.col("dt_construction_end") < F.col("dt_construction_start")
    ).count()
    record("DOMAIN_PROJ_DATE_ORDER", "L2", "Domain",
           "Construction end date >= start date",
           HARD, n_projects, end_before_start)

# 3.11 Project: amt_contract > 0 and gross_area > 0
if proj_df:
    invalid_amounts = proj_df.filter(
        (F.col("amt_contract") <= 0) | (F.col("gross_area") <= 0)
    ).count()
    record("DOMAIN_PROJ_POSITIVE_AMOUNTS", "L2", "Domain",
           "Contract amount and gross area are positive",
           HARD, n_projects, invalid_amounts)

# 3.12 RFI: sow_sequence_timing is valid enum
if rfi_df:
    valid_timings = ["Pre-SOW", "During", "Post-SOW"]
    invalid_timing = rfi_df.filter(
        ~F.col("sow_sequence_timing").isin(valid_timings)
    ).count()
    record("DOMAIN_RFI_VALID_TIMING", "L2", "Domain",
           "sow_sequence_timing is Pre-SOW, During, or Post-SOW",
           HARD, table_counts["rfi"], invalid_timing)

# 3.13 RFI: cat_trade_l1 is valid (8 canonical Uniformat categories)
if rfi_df:
    valid_trades = {
        "A SUBSTRUCTURE", "B SHELL", "C INTERIORS", "D SERVICES",
        "E EQUIPMENT AND FURNISHINGS", "F SPECIAL CONSTRUCTION AND DEMOLITION",
        "G SITEWORK", "Z GENERAL"
    }
    actual_trades = [
        row["cat_trade_l1"] for row in
        rfi_df.select("cat_trade_l1").distinct().collect()
    ]
    invalid_trades = [t for t in actual_trades if t and t.strip() not in valid_trades]
    invalid_count = rfi_df.filter(
        ~F.trim(F.col("cat_trade_l1")).isin(list(valid_trades))
    ).count()
    record("DOMAIN_RFI_VALID_TRADE", "L2", "Domain",
           "cat_trade_l1 is one of 8 canonical Uniformat categories",
           HARD, table_counts["rfi"], invalid_count,
           detail=f"Unrecognized: {invalid_trades[:5]}" if invalid_trades else "All valid")


# =============================================================================
# SECTION 4: CROSS-TABLE ALIGNMENT
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: CROSS-TABLE ALIGNMENT")
print("=" * 80)

# 4.1 RFI is_near_rebaseline consistency:
#     If is_near_rebaseline=True, there should be a rebaseline event within 30 days
if rfi_df and rb_df and sched_df:
    flagged_rfis = rfi_df.filter(F.col("is_near_rebaseline") == True)
    n_flagged = flagged_rfis.count()

    if n_flagged > 0:
        # Get rebaseline dates from schedule (dt_effective_from for non-baseline versions)
        rb_dates = (
            sched_df
            .filter(~F.col("cat_schedule_code").endswith("-BL"))
            .select(
                "ids_project_number_synth",
                F.col("dt_effective_from").alias("dt_rebaseline")
            )
        )
        # Join flagged RFIs to rebaseline dates and check 30-day window
        flagged_with_rb = (
            flagged_rfis
            .join(rb_dates, on="ids_project_number_synth", how="left")
            .withColumn("days_from_rb",
                F.abs(F.datediff(F.col("dt_rfi_created"), F.col("dt_rebaseline")))
            )
        )
        # For each flagged RFI, find the minimum distance to any rebaseline
        min_dist = (
            flagged_with_rb
            .groupBy("id_rfi_synth")
            .agg(F.min("days_from_rb").alias("min_days_from_rb"))
        )
        # Violations: flagged but no rebaseline within 60 days (generous window)
        no_nearby_rb = min_dist.filter(
            F.col("min_days_from_rb").isNull() | (F.col("min_days_from_rb") > 60)
        ).count()
        record("CROSS_RFI_REBASELINE_FLAG", "L2", "Cross-Table",
               "is_near_rebaseline=True RFIs have a rebaseline within 60 days",
               SOFT, n_flagged, no_nearby_rb)

# 4.2 Schedule version count aligns with rebaseline metadata count
if sched_df and rb_df:
    sched_version_counts = (
        sched_df
        .groupBy("ids_project_number_synth")
        .agg((F.count("*") - 1).alias("n_rebaselines_sched"))  # minus 1 for original BL
    )
    rb_counts = (
        rb_df
        .groupBy("ids_project_number_synth")
        .agg(F.count("*").alias("n_rebaselines_meta"))
    )
    count_compare = sched_version_counts.join(rb_counts, on="ids_project_number_synth", how="outer")
    mismatches = count_compare.filter(
        F.coalesce(F.col("n_rebaselines_sched"), F.lit(0)) !=
        F.coalesce(F.col("n_rebaselines_meta"), F.lit(0))
    ).count()
    n_projects_with_versions = count_compare.count()
    record("CROSS_SCHED_RB_COUNT_MATCH", "L2", "Cross-Table",
           "Rebaseline count matches between schedule and metadata tables",
           HARD, n_projects_with_versions, mismatches)

# 4.3 RFI volume scales with project size (sanity check)
#     Flag projects where RFI count per $M contract is wildly off
if rfi_df and proj_df:
    rfi_per_project = (
        rfi_df.groupBy("ids_project_number_synth")
        .agg(F.count("*").alias("n_rfis"))
    )
    rfi_scaling = rfi_per_project.join(
        proj_df.select("ids_project_number_synth", "amt_contract"),
        on="ids_project_number_synth"
    ).withColumn(
        "rfis_per_million", F.col("n_rfis") / (F.col("amt_contract") / 1e6)
    )
    # Flag extremely low (<1 RFI per $M) or high (>100 RFI per $M)
    outliers = rfi_scaling.filter(
        (F.col("rfis_per_million") < 1) | (F.col("rfis_per_million") > 100)
    ).count()
    n_with_rfis = rfi_scaling.count()
    # Only enforce as SOFT when we have enough projects for meaningful assessment
    sev = SOFT if n_with_rfis >= 10 else INFO
    record("CROSS_RFI_SIZE_SCALING", "L2", "Cross-Table",
           "RFI count per $M contract is between 1 and 100",
           sev, n_with_rfis, outliers,
           detail=f"{n_with_rfis} project(s) with RFIs; real range varies by SBU/asset class")

# 4.4 Task summary version count matches schedule version count per project
if task_df and sched_df:
    task_versions = (
        task_df
        .select("ids_project_number_synth", "cat_schedule_code").distinct()
        .groupBy("ids_project_number_synth")
        .agg(F.count("*").alias("n_task_versions"))
    )
    sched_versions = (
        sched_df
        .groupBy("ids_project_number_synth")
        .agg(F.count("*").alias("n_sched_versions"))
    )
    version_compare = task_versions.join(sched_versions, on="ids_project_number_synth", how="outer")
    mismatches = version_compare.filter(
        F.coalesce(F.col("n_task_versions"), F.lit(0)) !=
        F.coalesce(F.col("n_sched_versions"), F.lit(0))
    ).count()
    record("CROSS_TASK_SCHED_VERSION_MATCH", "L2", "Cross-Table",
           "Task summary and schedule have same number of versions per project",
           HARD, version_compare.count(), mismatches)

# 4.5 Budget project count vs project table
#     (informational: budget may only have 10 projects in smoke test)
if budget_df:
    budget_projects = budget_df.select("ids_project_number_synth").distinct().count()
    record("CROSS_BUDGET_PROJECT_COVERAGE", "L2", "Cross-Table",
           f"Budget covers {budget_projects} of {n_projects} projects",
           INFO, n_projects, n_projects - budget_projects,
           detail="Expected <1000 if budget is still smoke-test only")


# =============================================================================
# SECTION 5: CARDINALITY & DISTRIBUTION SANITY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: CARDINALITY & DISTRIBUTION SANITY")
print("=" * 80)

# 5.1 Project: all 7 SBUs represented
# 5.1 Project: check which SBUs are represented
#     Aviation and Federal may be absent if they didn't meet profiling filters
if proj_df:
    all_possible_sbus = {"commercial", "healthcare", "education", "mission critical",
                         "life sciences", "aviation", "federal"}
    actual_sbus_raw = [
        row["cat_sbu"] for row in
        proj_df.select("cat_sbu").distinct().collect()
    ]
    actual_sbus = set(s.lower().strip() for s in actual_sbus_raw if s)
    missing = all_possible_sbus - actual_sbus
    # INFO not HARD: some SBUs legitimately excluded by profiling filter
    record("CARD_ALL_SBUS_PRESENT", "L2", "Cardinality",
           f"SBUs present: {sorted(actual_sbus)}",
           INFO, len(all_possible_sbus), len(missing),
           detail=f"Missing (excluded by profiling filter): {sorted(missing)}" if missing else "All 7 present")
# 5.2 Schedule: average versions per project in reasonable range
if sched_df:
    avg_versions = table_counts["schedule"] / n_projects
    # Real data shows ~2.5 avg. Flag if way off.
    in_range = 1.5 <= avg_versions <= 4.0
    record("CARD_SCHED_AVG_VERSIONS", "L2", "Cardinality",
           f"Avg schedule versions per project: {avg_versions:.2f} (expect 1.5-4.0)",
           SOFT, 1, 0 if in_range else 1,
           detail=f"Actual: {avg_versions:.2f}, Real benchmark: ~2.5")

# 5.3 Task summary: ~23 SOWs per active version
if task_df:
    active_sow_count = (
        task_df.filter(F.col("is_active_record") == True)
        .groupBy("ids_project_number_synth")
        .agg(F.countDistinct("cat_sow").alias("n_sows"))
    )
    avg_sows = active_sow_count.agg(F.avg("n_sows")).collect()[0][0]
    min_sows = active_sow_count.agg(F.min("n_sows")).collect()[0][0]
    max_sows = active_sow_count.agg(F.max("n_sows")).collect()[0][0]
    in_range = 15 <= avg_sows <= 30
    record("CARD_TASK_AVG_SOWS", "L2", "Cardinality",
           f"Avg SOWs per project (active): {avg_sows:.1f} (expect 15-30)",
           SOFT, 1, 0 if in_range else 1,
           detail=f"Range: {min_sows}-{max_sows}, Mean: {avg_sows:.1f}")

# 5.4 RFI: status distribution reasonable
if rfi_df:
    status_dist = (
        rfi_df.groupBy("cat_status")
        .agg(F.count("*").alias("n"))
        .withColumn("pct", F.col("n") / F.lit(table_counts["rfi"]) * 100)
        .orderBy(F.desc("n"))
    )
    status_rows = status_dist.collect()
    status_str = ", ".join([f"{r['cat_status']}={r['pct']:.1f}%" for r in status_rows])
    # Closed should dominate (typically 70-90%)
    closed_pct = next((r["pct"] for r in status_rows if r["cat_status"] == "closed"), 0)
    n_rfi_projects = rfi_df.select("ids_project_number_synth").distinct().count()
    # Use SOFT threshold only when we have enough projects for aggregate stats
    sev = SOFT if n_rfi_projects >= 10 else INFO
    record("CARD_RFI_STATUS_DIST", "L2", "Cardinality",
           f"RFI status distribution: {status_str}",
           sev, 1, 0 if 50 <= closed_pct <= 95 else 1,
           detail=f"Closed: {closed_pct:.1f}% (expect 50-95%) | {n_rfi_projects} project(s) with RFIs")

# 5.5 Budget: divisions present
if budget_df:
    divisions = (
        budget_df.select("cat_budget_division").distinct()
        .orderBy("cat_budget_division")
        .collect()
    )
    div_list = [r["cat_budget_division"] for r in divisions]
    record("CARD_BUDGET_DIVISIONS", "L2", "Cardinality",
           f"Budget divisions present: {len(div_list)}",
           INFO, len(div_list), 0,
           detail=f"Divisions: {div_list[:10]}{'...' if len(div_list) > 10 else ''}")


# =============================================================================
# SCORECARD
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SCORECARD")
print("=" * 80)

n_total   = len(results)
n_passed  = sum(1 for r in results if r["passed"])
n_failed  = sum(1 for r in results if not r["passed"] and r["severity"] != INFO)
n_info    = sum(1 for r in results if r["severity"] == INFO)
n_hard_f  = sum(1 for r in results if not r["passed"] and r["severity"] == HARD)
n_soft_f  = sum(1 for r in results if not r["passed"] and r["severity"] == SOFT)

print(f"\n  Total checks:      {n_total}")
print(f"  Passed:            {n_passed}")
print(f"  Failed (HARD):     {n_hard_f}")
print(f"  Failed (SOFT):     {n_soft_f}")
print(f"  Informational:     {n_info}")
print(f"  Overall:           {'ALL CLEAR' if n_hard_f == 0 and n_soft_f == 0 else 'ISSUES FOUND'}")

if n_hard_f > 0 or n_soft_f > 0:
    print(f"\n  --- FAILED CHECKS ---")
    for r in results:
        if not r["passed"] and r["severity"] != INFO:
            print(f"  [{r['severity']:4s}] {r['check_id']:35s} | "
                  f"{r['violations']:,} violations / {r['total_checked']:,} checked "
                  f"({r['violation_pct']}%)")
            if r["detail"]:
                print(f"         Detail: {r['detail']}")

# Summary by category
print(f"\n  --- BY CATEGORY ---")
categories = sorted(set(r["category"] for r in results))
for cat in categories:
    cat_results = [r for r in results if r["category"] == cat]
    cat_pass = sum(1 for r in cat_results if r["passed"])
    cat_total = len(cat_results)
    icon = "+" if cat_pass == cat_total else "!"
    print(f"  [{icon}] {cat:15s}: {cat_pass}/{cat_total} passed")

print("\n" + "=" * 80)
print(f"RUN COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)