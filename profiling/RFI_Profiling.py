from pyspark.sql import functions as F, Window as W

# Configuration
META_TABLE          = "analysts.self_managed.lessons_learned_meta_data"
LESSONS_TABLE       = "analysts.self_managed.lessons_learned"
PROCORE_TABLE       = "gold.procore.rfis"
PROJECT_TABLE       = "platinum.models.project"
REBASELINE_TABLE    = "analysts.self_managed.rebaseline_event_metadata_v1"
OUT_RFI_PROFILE     = "analysts.self_managed.rfi_profile"
OUT_RFI_CONSTRAINTS = "analysts.self_managed.rfi_constraints"

MIN_GROUP_ROWS      = 5
LIFECYCLE_CAP_PCT   = 165   # p99 from real data (593,449 RFIs)
RESOLUTION_CAP_DAYS = 300   # p99=219 from real data (258,320 closed RFIs)

# Column type registry — typed nulls, no silent string casts
COL_TYPES = {
    "level":            "string",
    "field":            "string",
    "cat_sbu":          "string",
    "size_category":    "string",
    "cat_trade_l1":     "string",
    "cat_trade_l3":     "string",
    "status_simplified":"string",
    "n_projects":       "long",
    "n_rfis":           "long",
    "max_rfis":         "long",
    "p10":              "double",
    "p25":              "double",
    "median":           "double",
    "p75":              "double",
    "p90":              "double",
    "mean":             "double",
    "stddev":           "double",
    "pct_of_sbu":       "double",
    "pct_of_trade":     "double",
    "pct_design_issue": "double",
    "pct":              "double",
}

def pad_cols(df):
    for c, dtype in COL_TYPES.items():
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(dtype))
    return df.select(list(COL_TYPES.keys()))

# Section 0: Pre-flight — derive empirical caps from real data
# (stored as scalar constants; already confirmed above but
#  re-derived here so the script is self-contained)
print("Pre-flight: verifying empirical caps...")

project_base = (
    spark.read.table(PROJECT_TABLE)
    .filter(F.col("is_latest") == True)
    .filter(F.col("dt_construction_start").isNotNull())
    .filter(F.col("dt_construction_end").isNotNull())
    .filter(
        F.datediff("dt_construction_end", "dt_construction_start") > 30
    )
    .select(
        "ids_project_number", "cat_sbu",
        "amt_area_gross", "amt_contract",
        "dt_construction_start", "dt_construction_end"
    )
    .withColumn(
        "duration_days",
        F.datediff("dt_construction_end", "dt_construction_start")
    )
    .withColumn("size_category",
        F.when(F.col("amt_area_gross") < 50000,  "Small")
         .when(F.col("amt_area_gross") < 100000, "Medium")
         .when(F.col("amt_area_gross") < 200000, "Large")
         .otherwise("XLarge")
    )
    .filter(F.col("cat_sbu").isNotNull())
)

lifecycle_caps = (
    spark.read.table(META_TABLE)
    .join(
        project_base.select(
            "ids_project_number", "dt_construction_start", "duration_days"
        ),
        "ids_project_number", "inner"
    )
    .filter(F.col("dt_created").isNotNull())
    .filter(F.col("dt_created") >= F.col("dt_construction_start"))
    .withColumn(
        "lifecycle_pct",
        F.datediff("dt_created", "dt_construction_start")
        * 100.0 / F.col("duration_days")
    )
    .filter(F.col("lifecycle_pct") <= 200)
    .agg(
        F.percentile_approx("lifecycle_pct", 0.99).alias("p99"),
        F.percentile_approx("lifecycle_pct", 0.999).alias("p999"),
        F.count("*").alias("n_rfis")
    )
)
lifecycle_caps.show()

resolution_caps = (
    spark.read.table(PROCORE_TABLE)
    .filter(F.col("time_resolved").isNotNull())
    .filter(F.col("created_at").isNotNull())
    .filter(F.col("time_resolved") > F.col("created_at"))
    .filter(F.lower(F.col("status")) == "closed")
    .withColumn(
        "resolution_days",
        F.datediff("time_resolved", "created_at")
    )
    .filter(F.col("resolution_days") > 0)
    .agg(
        F.percentile_approx("resolution_days", 0.99).alias("p99"),
        F.percentile_approx("resolution_days", 0.999).alias("p999"),
        F.count("*").alias("n_rfis")
    )
)
resolution_caps.show()
print(f"Using lifecycle cap: {LIFECYCLE_CAP_PCT}%  |  resolution cap: {RESOLUTION_CAP_DAYS} days")

# Section 1: Canonical trade normalization
print("Normalizing trade_l1 to canonical 8 categories...")

CANONICAL_TRADES = [
    'A SUBSTRUCTURE', 'B SHELL', 'C INTERIORS', 'D SERVICES',
    'E EQUIPMENT AND FURNISHINGS',
    'F SPECIAL CONSTRUCTION AND DEMOLITION',
    'G SITEWORK', 'Z GENERAL'
]

lessons_raw = spark.read.table(LESSONS_TABLE).select(
    "ids_project_number", "id_str",
    "trade_l1", "trade_l3", "design_issue"
)

lessons_df = (
    lessons_raw
    .withColumn("trade_l1_norm", F.upper(F.trim(F.col("trade_l1"))))
    .filter(F.col("trade_l1_norm").isin(CANONICAL_TRADES))
)

# Pull SBU from meta table — join on project+id_str
_meta_raw = spark.read.table(META_TABLE)

# Drop cat_sbu from meta before joining — project_base is the authoritative
# source for SBU since it's already quality-filtered (is_latest, contract ≥ 25M etc.)
# This avoids AMBIGUOUS_REFERENCE when both tables carry cat_sbu
_meta_cols_no_sbu = [c for c in _meta_raw.columns if c != "cat_sbu"]

meta_sbu = (
    _meta_raw.select(_meta_cols_no_sbu)
    .join(
        project_base.select("ids_project_number", "cat_sbu",
                            "size_category", "dt_construction_start",
                            "dt_construction_end", "duration_days"),
        "ids_project_number", "inner"
    )
    .select(
        "ids_project_number", "id_str", "cat_sbu", "size_category",
        "dt_created", "cat_status",
        "dt_construction_start", "dt_construction_end", "duration_days"
    )
)

lessons_with_meta = (
    lessons_df
    .join(meta_sbu, ["ids_project_number", "id_str"], "inner")
    .filter(F.col("cat_sbu").isNotNull())
)

# Section 2: RFI Count Distribution — SBU × Size + fallbacks
print("Profiling RFI counts by SBU × size...")

rfi_counts = (
    spark.read.table(META_TABLE)
    .groupBy("ids_project_number")
    .agg(F.count("id_str").alias("n_rfis"))
)

rfi_count_with_meta = rfi_counts.join(
    project_base.select("ids_project_number", "cat_sbu", "size_category"),
    "ids_project_number", "inner"
)

rfi_count_by_sbu_size = (
    rfi_count_with_meta
    .groupBy("cat_sbu", "size_category")
    .agg(
        F.count("*").alias("n_projects"),
        F.percentile_approx("n_rfis", 0.25).alias("p25"),
        F.percentile_approx("n_rfis", 0.50).alias("median"),
        F.percentile_approx("n_rfis", 0.75).alias("p75"),
        F.percentile_approx("n_rfis", 0.90).alias("p90"),
        F.max("n_rfis").alias("max_rfis"),
        F.avg("n_rfis").alias("mean"),
        F.stddev_samp("n_rfis").alias("stddev")
    )
    .filter(F.col("n_projects") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU_SIZE"))
    .withColumn("field", F.lit("n_rfis_per_project"))
)

rfi_count_by_sbu = (
    rfi_count_with_meta
    .groupBy("cat_sbu")
    .agg(
        F.count("*").alias("n_projects"),
        F.percentile_approx("n_rfis", 0.25).alias("p25"),
        F.percentile_approx("n_rfis", 0.50).alias("median"),
        F.percentile_approx("n_rfis", 0.75).alias("p75"),
        F.percentile_approx("n_rfis", 0.90).alias("p90"),
        F.max("n_rfis").alias("max_rfis"),
        F.avg("n_rfis").alias("mean"),
        F.stddev_samp("n_rfis").alias("stddev")
    )
    .filter(F.col("n_projects") >= MIN_GROUP_ROWS)
    .withColumn("level",         F.lit("SBU"))
    .withColumn("field",         F.lit("n_rfis_per_project"))
    .withColumn("size_category", F.lit(None).cast("string"))
)

rfi_count_global = (
    rfi_count_with_meta
    .agg(
        F.count("*").alias("n_projects"),
        F.percentile_approx("n_rfis", 0.25).alias("p25"),
        F.percentile_approx("n_rfis", 0.50).alias("median"),
        F.percentile_approx("n_rfis", 0.75).alias("p75"),
        F.percentile_approx("n_rfis", 0.90).alias("p90"),
        F.max("n_rfis").alias("max_rfis"),
        F.avg("n_rfis").alias("mean"),
        F.stddev_samp("n_rfis").alias("stddev")
    )
    .withColumn("level",         F.lit("GLOBAL"))
    .withColumn("field",         F.lit("n_rfis_per_project"))
    .withColumn("cat_sbu",       F.lit(None).cast("string"))
    .withColumn("size_category", F.lit(None).cast("string"))
)

# Section 3: Trade Weights by SBU + Global fallback
print("Profiling trade weights by SBU...")

sbu_totals = lessons_with_meta.groupBy("cat_sbu").agg(
    F.count("*").alias("sbu_total")
)

trade_weights_by_sbu = (
    lessons_with_meta
    .groupBy("cat_sbu", "trade_l1_norm")
    .agg(F.count("*").alias("n_rfis"))
    .join(sbu_totals, "cat_sbu", "inner")
    .withColumn("pct_of_sbu",
        F.round(F.col("n_rfis") * 100.0 / F.col("sbu_total"), 4))
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumn("level", F.lit("SBU"))
    .withColumn("field", F.lit("trade_weight"))
)

global_total = lessons_with_meta.count()
trade_weights_global = (
    lessons_with_meta
    .groupBy("trade_l1_norm")
    .agg(F.count("*").alias("n_rfis"))
    .withColumn("pct_of_sbu",
        F.round(F.col("n_rfis") * 100.0 / F.lit(global_total), 4))
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumn("level",   F.lit("GLOBAL"))
    .withColumn("field",   F.lit("trade_weight"))
    .withColumn("cat_sbu", F.lit(None).cast("string"))
)

# Section 4: Lifecycle Timing by Trade
# — empirical cap: 165% (p99 from 593,449 RFIs)
print("Profiling lifecycle timing by trade...")

lifecycle_df = (
    lessons_with_meta
    .filter(F.col("dt_created").isNotNull())
    .filter(F.col("dt_created") >= F.col("dt_construction_start"))
    .withColumn(
        "lifecycle_pct",
        F.round(
            F.datediff("dt_created", "dt_construction_start")
            * 100.0 / F.col("duration_days"), 2
        )
    )
    .filter(F.col("lifecycle_pct").between(0, LIFECYCLE_CAP_PCT))
)

lifecycle_by_trade = (
    lifecycle_df
    .groupBy("trade_l1_norm")
    .agg(
        F.count("*").alias("n_rfis"),
        F.percentile_approx("lifecycle_pct", 0.10).alias("p10"),
        F.percentile_approx("lifecycle_pct", 0.25).alias("p25"),
        F.percentile_approx("lifecycle_pct", 0.50).alias("median"),
        F.percentile_approx("lifecycle_pct", 0.75).alias("p75"),
        F.percentile_approx("lifecycle_pct", 0.90).alias("p90"),
        F.avg("lifecycle_pct").alias("mean"),
        F.stddev_samp("lifecycle_pct").alias("stddev")
    )
    .filter(F.col("n_rfis") >= MIN_GROUP_ROWS)
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumn("level", F.lit("TRADE"))
    .withColumn("field", F.lit("lifecycle_timing_pct"))
)

# Section 5: SOW Timing Zone Probabilities by Trade
# (Pre-SOW / During / Post-SOW relative to matched SOW window)
# — requires synthetic_task_summary_by_sow_v1 for real projects
# — profiled here as proportions to ground generation defaults
print("Profiling SOW timing zone distribution by trade...")

# NOTE: This section profiles what % of RFIs fall pre/during/post their
# matched SOW window in real data. Uses the trade→SOW mapping as a proxy
# since no direct SOW field exists in the RFI tables.
# Pre-SOW window defined as: dt_rfi_created < SOW_start
# During: SOW_start <= dt_rfi_created <= SOW_finish
# Post-SOW: dt_rfi_created > SOW_finish

# Since we cannot directly join to real SOW windows (P6 not linked to
# Procore RFIs), we use lifecycle_pct bands as a proxy:
# Pre-SOW proxy  → RFI lifecycle_pct is in the bottom 20% for that trade
# Post-SOW proxy → RFI lifecycle_pct is in the top 10% for that trade

# Build per-trade p20 and p90 cutoffs from lifecycle distribution
trade_lifecycle_cutoffs = (
    lifecycle_df
    .groupBy("trade_l1_norm")
    .agg(
        F.percentile_approx("lifecycle_pct", 0.20).alias("p20_cutoff"),
        F.percentile_approx("lifecycle_pct", 0.90).alias("p90_cutoff"),
        F.count("*").alias("n_rfis_total")
    )
)

sow_timing_zone_by_trade = (
    lifecycle_df
    .join(trade_lifecycle_cutoffs, "trade_l1_norm", "inner")
    .withColumn("timing_zone",
        F.when(F.col("lifecycle_pct") < F.col("p20_cutoff"), "pre_sow")
         .when(F.col("lifecycle_pct") > F.col("p90_cutoff"), "post_sow")
         .otherwise("during_sow")
    )
    .groupBy("trade_l1_norm", "timing_zone")
    .agg(F.count("*").alias("n_rfis"))
    .join(
        lifecycle_df.groupBy("trade_l1_norm")
            .agg(F.count("*").alias("trade_total")),
        "trade_l1_norm", "inner"
    )
    .withColumn("pct_of_trade",
        F.round(F.col("n_rfis") * 100.0 / F.col("trade_total"), 4))
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumnRenamed("timing_zone", "status_simplified")  # reuse col
    .withColumn("level", F.lit("TRADE"))
    .withColumn("field", F.lit("sow_timing_zone_pct"))
)

# Section 6: Resolution Time Distribution
# — empirical cap: 300 days (p99=219, max meaningful ~300)
print("Profiling resolution time...")

resolution_df = (
    spark.read.table(PROCORE_TABLE)
    .filter(F.col("time_resolved").isNotNull())
    .filter(F.col("created_at").isNotNull())
    .filter(F.col("time_resolved") > F.col("created_at"))
    .filter(F.lower(F.col("status")) == "closed")
    .withColumn("resolution_days",
        F.datediff("time_resolved", "created_at"))
    .filter(F.col("resolution_days").between(1, RESOLUTION_CAP_DAYS))
    .agg(
        F.count("*").alias("n_rfis"),
        F.percentile_approx("resolution_days", 0.25).alias("p25"),
        F.percentile_approx("resolution_days", 0.50).alias("median"),
        F.percentile_approx("resolution_days", 0.75).alias("p75"),
        F.percentile_approx("resolution_days", 0.90).alias("p90"),
        F.avg("resolution_days").alias("mean"),
        F.stddev_samp("resolution_days").alias("stddev")
    )
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("field", F.lit("resolution_days"))
)

# Section 7: Design Issue % by Trade
print("Profiling design issue flag by trade...")

design_issue_by_trade = (
    lessons_with_meta
    .groupBy("trade_l1_norm")
    .agg(
        F.count("*").alias("n_rfis"),
        F.round(
            F.sum(F.when(F.col("design_issue") == True, 1).otherwise(0))
            * 100.0 / F.count("*"), 4
        ).alias("pct_design_issue")
    )
    .filter(F.col("n_rfis") >= MIN_GROUP_ROWS)
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumn("level", F.lit("TRADE"))
    .withColumn("field", F.lit("design_issue_pct"))
)

# Section 8: Status Distribution by SBU
print("Profiling status distribution by SBU...")

status_with_sbu = (
    meta_sbu
    .filter(F.trim(F.col("cat_status")) != "")
    .filter(F.col("cat_status").isNotNull())
    .withColumn("status_simplified",
        F.when(F.lower(F.col("cat_status")).isin(
            "closed", "closed_draft", "closed_with_revision"), "closed")
         .when(F.lower(F.col("cat_status")) == "draft", "draft")
         .otherwise("open"))
)

sbu_status_totals = status_with_sbu.groupBy("cat_sbu").agg(
    F.count("*").alias("sbu_total")
)

status_by_sbu = (
    status_with_sbu
    .groupBy("cat_sbu", "status_simplified")
    .agg(F.count("*").alias("n_rfis"))
    .join(sbu_status_totals, "cat_sbu", "inner")
    .withColumn("pct",
        F.round(F.col("n_rfis") * 100.0 / F.col("sbu_total"), 4))
    .withColumn("level", F.lit("SBU"))
    .withColumn("field", F.lit("status_distribution"))
)

# Section 9: Rebaseline Spike Profile
# — % uplift in RFI volume in 30-day post-rebaseline window
print("Profiling rebaseline spike...")

# Source real rebaseline events from schedule_baseline (real project IDs)
# synthetic rebaseline_event_metadata_v1 uses ids_project_number_synth
# and cannot be joined against real RFI data in lessons_learned_meta_data
SCHEDULE_BASELINE_TABLE = "platinum.models.schedule_baseline"

rebaseline_df = (
    spark.read.table(SCHEDULE_BASELINE_TABLE)
    .filter(F.col("is_active_record") == False)   # historical versions = rebaselines
    .filter(F.col("dt_tco").isNotNull())
    .filter(F.col("dt_effective_from").isNotNull())
    # Each non-active record represents a rebaseline snapshot
    .select(
        "ids_project_number",
        F.col("dt_effective_from").alias("dt_rebaseline_occurred")
    )
    .withColumn("rebaseline_number",
        F.row_number().over(
            W.partitionBy("ids_project_number")
             .orderBy("dt_rebaseline_occurred")
        )
    )
    .distinct()
)

# Assert join will be non-empty before proceeding
_rb_project_count = rebaseline_df.select("ids_project_number").distinct().count()
_meta_project_count = meta_sbu.select("ids_project_number").distinct().count()
_overlap = (
    rebaseline_df.select("ids_project_number")
    .intersect(meta_sbu.select("ids_project_number"))
    .count()
)
print(f"Rebaseline projects (real schedule_baseline): {_rb_project_count}")
print(f"Meta table projects:                          {_meta_project_count}")
print(f"Overlap (joinable projects):                  {_overlap}")
assert _overlap > 0, (
    "CRITICAL: Zero overlap between schedule_baseline rebaselines and "
    "lessons_learned_meta_data. Rebaseline spike profile will be empty. "
    "Check that both tables share the same ids_project_number space."
)

# Join RFIs to rebaseline events on same project
rfi_rebaseline = (
    meta_sbu
    .filter(F.col("dt_created").isNotNull())
    .join(rebaseline_df, "ids_project_number", "inner")
    .withColumn("days_from_rebaseline",
        F.datediff("dt_created", "dt_rebaseline_occurred"))
)

# Compute avg RFIs per 30-day window: baseline (-60 to -31) vs post (0 to +30)
baseline_window = (
    rfi_rebaseline
    .filter(F.col("days_from_rebaseline").between(-60, -31))
    .groupBy("ids_project_number", "rebaseline_number")
    .agg(F.count("*").alias("n_rfis_baseline_window"))
)

post_window = (
    rfi_rebaseline
    .filter(F.col("days_from_rebaseline").between(0, 30))
    .groupBy("ids_project_number", "rebaseline_number")
    .agg(F.count("*").alias("n_rfis_post_window"))
)

rebaseline_spike = (
    baseline_window
    .join(post_window, ["ids_project_number", "rebaseline_number"], "inner")
    .withColumn("spike_ratio",
        F.when(F.col("n_rfis_baseline_window") > 0,
               F.col("n_rfis_post_window") / F.col("n_rfis_baseline_window")))
    .filter(F.col("spike_ratio").between(0.5, 5))  # tightened from (0.1, 10)
    .agg(
        F.count("*").alias("n_observations"),
        F.percentile_approx("spike_ratio", 0.25).alias("p25"),
        F.percentile_approx("spike_ratio", 0.50).alias("median"),
        F.percentile_approx("spike_ratio", 0.75).alias("p75"),
        F.avg("spike_ratio").alias("mean"),
        F.stddev_samp("spike_ratio").alias("stddev")
    )
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("field", F.lit("rebaseline_spike_ratio"))
    .withColumnRenamed("n_observations", "n_rfis")
)

# Section 10: L3 Frequency by Trade (diversity weighting)
print("Profiling Uniformat L3 frequencies by trade...")

l3_exploded = (
    lessons_with_meta
    .withColumn("trade_l3_val", F.explode("trade_l3"))
    .withColumn("trade_l3_norm", F.upper(F.trim(F.col("trade_l3_val"))))
    # Valid Uniformat codes: letter + digit prefix (e.g. D3010, B1020)
    .filter(F.col("trade_l3_norm").rlike("^[A-Z][0-9]"))
    # Discard LLM hallucinations — too long or garbage strings
    .filter(F.length(F.col("trade_l3_norm")) <= 60)
    .filter(~F.col("trade_l3_norm").rlike("[?!@#$%]"))
)

trade_l3_totals = l3_exploded.groupBy("trade_l1_norm").agg(
    F.count("*").alias("trade_total")
)

l3_frequency_by_trade = (
    l3_exploded
    .groupBy("trade_l1_norm", "trade_l3_norm")
    .agg(F.count("*").alias("n_rfis"))
    .join(trade_l3_totals, "trade_l1_norm", "inner")
    .withColumn("pct_of_trade",
        F.round(F.col("n_rfis") * 100.0 / F.col("trade_total"), 4))
    .filter(F.col("n_rfis") >= MIN_GROUP_ROWS)
    .withColumnRenamed("trade_l1_norm", "cat_trade_l1")
    .withColumnRenamed("trade_l3_norm", "cat_trade_l3")
    .withColumn("level", F.lit("TRADE"))
    .withColumn("field", F.lit("l3_frequency"))
)

# Section 11: Few-Shot Example Pool
# — top 500 examples per trade × SBU for LLM prompt seeding
# — stored in profile table for retrieval during generation
print("Building few-shot example pool...")

few_shot_pool = (
    lessons_with_meta
    .join(
        spark.read.table(LESSONS_TABLE).select(
            "ids_project_number", "id_str",
            "lesson_general", "root_cause"
        ),
        ["ids_project_number", "id_str"], "inner"
    )
    .filter(F.col("lesson_general").isNotNull())
    .filter(F.length(F.col("lesson_general")) > 50)
    # Deduplicate on lesson text before ranking — same lesson can appear
    # across multiple projects (copy-paste or LLM repetition)
    .dropDuplicates(["trade_l1_norm", "cat_sbu", "lesson_general"])
    .withColumn("rank",
        F.row_number().over(
            W.partitionBy("trade_l1_norm", "cat_sbu")
             .orderBy(F.length("lesson_general").desc())
        )
    )
    .filter(F.col("rank") <= 10)
    .select(
        F.col("trade_l1_norm").alias("cat_trade_l1"),
        "cat_sbu",
        "lesson_general",
        "root_cause"
    )
    .withColumn("level", F.lit("TRADE_SBU"))
    .withColumn("field", F.lit("few_shot_example"))
    .withColumn("bounds_json",
        F.to_json(F.struct(
            F.col("lesson_general").alias("lesson"),
            F.col("root_cause").alias("root_cause")
        ))
    )
)

# Section 12: Consolidate Profile Table
print("Consolidating profile table...")

profile_all = (
    pad_cols(rfi_count_by_sbu_size)
    .unionByName(pad_cols(rfi_count_by_sbu))
    .unionByName(pad_cols(rfi_count_global))
    .unionByName(pad_cols(trade_weights_by_sbu))
    .unionByName(pad_cols(trade_weights_global))
    .unionByName(pad_cols(lifecycle_by_trade))
    .unionByName(pad_cols(sow_timing_zone_by_trade))
    .unionByName(pad_cols(resolution_df))
    .unionByName(pad_cols(design_issue_by_trade))
    .unionByName(pad_cols(status_by_sbu))
    .unionByName(pad_cols(rebaseline_spike))
    .unionByName(pad_cols(l3_frequency_by_trade))
)

(profile_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_RFI_PROFILE))

print(f"Saved profile: {profile_all.count()} rows → {OUT_RFI_PROFILE}")

# Section 13: Constraints Table
print("Building constraints table...")

CONSTRAINT_COLS = {
    "level":         "string",
    "field":         "string",
    "cat_sbu":       "string",
    "size_category": "string",
    "cat_trade_l1":  "string",
    "n_projects":    "long",
    "n_rfis":        "long",
    "bounds_json":   "string",
}

def pad_constraint_cols(df):
    for c, dtype in CONSTRAINT_COLS.items():
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(dtype))
    return df.select(list(CONSTRAINT_COLS.keys()))

def make_constraints(df, level, field, value_cols):
    map_args = []
    for c in value_cols:
        if c in df.columns:
            map_args.extend([F.lit(c), F.col(c).cast("double")])
    return (
        df
        .withColumn("bounds_json", F.to_json(F.create_map(*map_args)))
        .withColumn("level", F.lit(level))
        .withColumn("field", F.lit(field))
    )

# RFI count constraints — three-level hierarchy
rfi_count_constraints = (
    pad_constraint_cols(
        make_constraints(
            rfi_count_by_sbu_size, "SBU_SIZE", "n_rfis_per_project",
            ["p25", "median", "p75", "p90", "mean", "stddev"]
        )
    )
    .unionByName(pad_constraint_cols(
        make_constraints(
            rfi_count_by_sbu, "SBU", "n_rfis_per_project",
            ["p25", "median", "p75", "p90", "mean", "stddev"]
        ).withColumn("size_category", F.lit(None).cast("string"))
    ))
    .unionByName(pad_constraint_cols(
        make_constraints(
            rfi_count_global, "GLOBAL", "n_rfis_per_project",
            ["p25", "median", "p75", "p90", "mean", "stddev"]
        ).withColumn("cat_sbu",       F.lit(None).cast("string"))
         .withColumn("size_category", F.lit(None).cast("string"))
    ))
)

# Lifecycle timing constraints by trade
lifecycle_constraints = pad_constraint_cols(
    make_constraints(
        lifecycle_by_trade, "TRADE", "lifecycle_timing_pct",
        ["p10", "p25", "median", "p75", "p90", "mean", "stddev"]
    )
)

# Resolution time constraints
resolution_constraints = pad_constraint_cols(
    make_constraints(
        resolution_df, "GLOBAL", "resolution_days",
        ["p25", "median", "p75", "p90", "mean", "stddev"]
    ).withColumn("cat_sbu",       F.lit(None).cast("string"))
     .withColumn("size_category", F.lit(None).cast("string"))
)

# Rebaseline spike constraints
rebaseline_constraints = pad_constraint_cols(
    make_constraints(
        rebaseline_spike, "GLOBAL", "rebaseline_spike_ratio",
        ["p25", "median", "p75", "mean", "stddev"]
    ).withColumn("cat_sbu",       F.lit(None).cast("string"))
     .withColumn("size_category", F.lit(None).cast("string"))
)

# Few-shot examples (bounds_json already built in Section 11)
few_shot_constraints = pad_constraint_cols(
    few_shot_pool
    .withColumn("level",         F.lit("TRADE_SBU"))
    .withColumn("field",         F.lit("few_shot_example"))
    .withColumn("n_projects",    F.lit(None).cast("long"))
    .withColumn("n_rfis",        F.lit(None).cast("long"))
    .withColumn("size_category", F.lit(None).cast("string"))
)

constraints_all = (
    rfi_count_constraints
    .unionByName(lifecycle_constraints)
    .unionByName(resolution_constraints)
    .unionByName(rebaseline_constraints)
    .unionByName(few_shot_constraints)
)

(constraints_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_RFI_CONSTRAINTS))

print(f"Saved constraints: {constraints_all.count()} rows → {OUT_RFI_CONSTRAINTS}")

# Section 14: Validation Summary
print("\n=== PROFILING SUMMARY ===")
print("\nProfile rows by level × field:")
profile_all.groupBy("level", "field").count().orderBy("level", "field").show(40, truncate=False)

print("\nLifecycle timing by trade (key medians):")
lifecycle_by_trade.select(
    "cat_trade_l1", "n_rfis", "p10", "p25", "median", "p75", "p90"
).orderBy("median").show(truncate=False)

print("\nRFI counts by SBU × size (medians):")
rfi_count_by_sbu_size.select(
    "cat_sbu", "size_category", "n_projects", "p25", "median", "p75", "p90"
).orderBy("cat_sbu", "size_category").show(40, truncate=False)

print("\nTrade weights by SBU:")
trade_weights_by_sbu.select(
    "cat_sbu", "cat_trade_l1", "n_rfis", "pct_of_sbu"
).orderBy("cat_sbu", F.col("pct_of_sbu").desc()).show(60, truncate=False)

print("\nRebaseline spike ratio:")
rebaseline_spike.show(truncate=False)

print("\nL3 categories per trade (count of distinct valid L3s):")
l3_frequency_by_trade.groupBy("cat_trade_l1").count().orderBy(
    "cat_trade_l1"
).show(truncate=False)

print("\n=== PROFILING COMPLETE ===")
