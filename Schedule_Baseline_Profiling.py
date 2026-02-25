from pyspark.sql import functions as F, Window as W

# Configuration
SOURCE_SCHEDULE = "platinum.models.schedule_baseline"
SOURCE_PROJECT = "platinum.models.project"
OUT_PROFILE = "analysts.self_managed.schedule_baseline_profile"
OUT_CONSTRAINTS = "analysts.self_managed.schedule_baseline_constraints"
MIN_GROUP_ROWS = 5  # Minimum projects for group profiling

# Data Loading & Cleaning
schedule_df = spark.read.table(SOURCE_SCHEDULE).select(
    "ids_project_number",
    "cat_schedule_code",
    "dt_ntp",
    "dt_tco",
    "dt_next_tco",
    "dt_co",
    "cat_source_tco_date",
    "dt_effective_from",
    "dt_effective_to",
    "is_first_record_for_new_snapshot",
    "is_active_record"
)

# Load project characteristics for grouping
project_df = spark.read.table(SOURCE_PROJECT).select(
    "ids_project_number",
    "cat_sbu",
    "cat_region",
    "cat_main_asset_class",
    "amt_area_gross",
    "amt_contract",
    "dt_construction_start",
    "dt_construction_end"
)

# Apply data quality filters (matching Stage 1 approach)
project_df = (
    project_df
    .filter(F.col("amt_contract") >= 25000000)  # Contract value ≥ $25M
    .filter(F.year("dt_construction_start") >= 2019)  # Projects starting 2019+
    .filter(F.col("dt_construction_start").isNotNull())
    .filter(F.col("cat_sbu").isNotNull())
)

# Join schedule with filtered projects
df = (
    schedule_df
    .join(project_df, "ids_project_number", "inner")
)

# Standardize dates
df = (
    df
    .withColumn("dt_ntp", F.to_date("dt_ntp"))
    .withColumn("dt_tco", F.to_date("dt_tco"))
    .withColumn("dt_next_tco", F.to_date("dt_next_tco"))
    .withColumn("dt_co", F.to_date("dt_co"))
    .withColumn("dt_effective_from", F.to_date("dt_effective_from"))
    .withColumn("dt_effective_to", F.to_date("dt_effective_to"))
    .withColumn("dt_construction_start", F.to_date("dt_construction_start"))
    .withColumn("dt_construction_end", F.to_date("dt_construction_end"))
)

# Create size buckets for profiling
df = df.withColumn(
    "size_category",
    F.when(F.col("amt_area_gross") < 50000, "Small")
     .when(F.col("amt_area_gross") < 100000, "Medium")
     .when(F.col("amt_area_gross") < 200000, "Large")
     .otherwise("XLarge")
)

# Create contract value buckets for profiling 
df = df.withColumn(
    "contract_bucket",
    F.when(F.col("amt_contract") < 50000000, "Medium")  # $25-50M
     .when(F.col("amt_contract") < 100000000, "Large")   # $50-100M
     .when(F.col("amt_contract") < 200000000, "XLarge")  # $100-200M
     .otherwise("XXLarge")  # > $200M
)

# Identify Distinct Baseline Versions
# Collapse monthly snapshots to distinct baseline versions
# Keep first occurrence of each unique dt_tco per project
distinct_baselines = (
    df
    .filter(F.col("dt_tco").isNotNull())
    .filter(F.col("is_first_record_for_new_snapshot") == True)
    .groupBy("ids_project_number", "dt_tco")
    .agg(
        F.first("cat_schedule_code").alias("cat_schedule_code"),
        F.first("dt_ntp").alias("dt_ntp"),
        F.first("dt_next_tco").alias("dt_next_tco"),
        F.first("dt_co").alias("dt_co"),
        F.first("cat_source_tco_date").alias("cat_source_tco_date"),
        F.min("dt_effective_from").alias("dt_effective_from"),
        F.max("dt_effective_to").alias("dt_effective_to"),
        F.first("is_active_record").alias("is_active_record"),
        F.first("cat_sbu").alias("cat_sbu"),
        F.first("cat_region").alias("cat_region"),
        F.first("cat_main_asset_class").alias("cat_main_asset_class"),
        F.first("size_category").alias("size_category"),
        F.first("contract_bucket").alias("contract_bucket"),
        F.first("dt_construction_start").alias("dt_construction_start"),
        F.first("dt_construction_end").alias("dt_construction_end"),
        F.first("amt_area_gross").alias("amt_area_gross"),
        F.first("amt_contract").alias("amt_contract")
    )
)

# Add version sequence number per project
w_proj = W.partitionBy("ids_project_number").orderBy("dt_effective_from")
distinct_baselines = distinct_baselines.withColumn(
    "version_seq",
    F.row_number().over(w_proj)
)

# Calculate rebaseline count per project
rebaseline_counts = (
    distinct_baselines
    .groupBy("ids_project_number")
    .agg(
        F.max("version_seq").alias("n_versions"),
        F.first("cat_sbu").alias("cat_sbu"),
        F.first("cat_region").alias("cat_region"),
        F.first("cat_main_asset_class").alias("cat_main_asset_class"),
        F.first("size_category").alias("size_category"),
        F.first("contract_bucket").alias("contract_bucket")
    )
    .withColumn("n_rebaselines", F.col("n_versions") - 1)
)

# Rebaseline Count Profiling
print("Profiling rebaseline counts...")

# Overall distribution
rebaseline_overall = (
    rebaseline_counts
    .groupBy("n_rebaselines")
    .agg(F.count("*").alias("n_projects"))
    .withColumn("pct_projects", 
                F.round(F.col("n_projects") * 100.0 / F.sum("n_projects").over(W.orderBy(F.lit(1))), 2))
    .orderBy("n_rebaselines")
)

print("Overall rebaseline distribution:")
rebaseline_overall.show(20, truncate=False)

# By SBU only
rebaseline_by_sbu = (
    rebaseline_counts
    .groupBy("cat_sbu")
    .agg(
        F.count("*").alias("n_projects"),
        F.avg("n_rebaselines").alias("avg_rebaselines"),
        F.percentile_approx("n_rebaselines", 0.01).alias("p1"),
        F.percentile_approx("n_rebaselines", 0.05).alias("p5"),
        F.percentile_approx("n_rebaselines", 0.25).alias("p25"),
        F.percentile_approx("n_rebaselines", 0.50).alias("p50"),
        F.percentile_approx("n_rebaselines", 0.75).alias("p75"),
        F.percentile_approx("n_rebaselines", 0.95).alias("p95"),
        F.percentile_approx("n_rebaselines", 0.99).alias("p99"),
        F.max("n_rebaselines").alias("max_rebaselines")
    )
    .withColumn("level", F.lit("SBU"))
    .select("level", "cat_sbu", F.lit(None).cast("string").alias("cat_main_asset_class"), 
            F.lit(None).cast("string").alias("size_category"),
            F.lit(None).cast("string").alias("contract_bucket"),
            "n_projects", "avg_rebaselines", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max_rebaselines")
)

# By SBU × Size
rebaseline_by_sbu_size = (
    rebaseline_counts
    .filter(F.col("size_category").isNotNull())
    .groupBy("cat_sbu", "size_category")
    .agg(
        F.count("*").alias("n_projects"),
        F.avg("n_rebaselines").alias("avg_rebaselines"),
        F.percentile_approx("n_rebaselines", 0.01).alias("p1"),
        F.percentile_approx("n_rebaselines", 0.05).alias("p5"),
        F.percentile_approx("n_rebaselines", 0.25).alias("p25"),
        F.percentile_approx("n_rebaselines", 0.50).alias("p50"),
        F.percentile_approx("n_rebaselines", 0.75).alias("p75"),
        F.percentile_approx("n_rebaselines", 0.95).alias("p95"),
        F.percentile_approx("n_rebaselines", 0.99).alias("p99"),
        F.max("n_rebaselines").alias("max_rebaselines")
    )
    .filter(F.col("n_projects") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU_SIZE"))
    .select("level", "cat_sbu", F.lit(None).cast("string").alias("cat_main_asset_class"), 
            "size_category", F.lit(None).cast("string").alias("contract_bucket"),
            "n_projects", "avg_rebaselines", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max_rebaselines")
)

# By SBU × Contract Bucket
rebaseline_by_sbu_contract = (
    rebaseline_counts
    .filter(F.col("contract_bucket").isNotNull())
    .groupBy("cat_sbu", "contract_bucket")
    .agg(
        F.count("*").alias("n_projects"),
        F.avg("n_rebaselines").alias("avg_rebaselines"),
        F.percentile_approx("n_rebaselines", 0.01).alias("p1"),
        F.percentile_approx("n_rebaselines", 0.05).alias("p5"),
        F.percentile_approx("n_rebaselines", 0.25).alias("p25"),
        F.percentile_approx("n_rebaselines", 0.50).alias("p50"),
        F.percentile_approx("n_rebaselines", 0.75).alias("p75"),
        F.percentile_approx("n_rebaselines", 0.95).alias("p95"),
        F.percentile_approx("n_rebaselines", 0.99).alias("p99"),
        F.max("n_rebaselines").alias("max_rebaselines")
    )
    .filter(F.col("n_projects") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU_CONTRACT"))
    .select("level", "cat_sbu", F.lit(None).cast("string").alias("cat_main_asset_class"), 
            F.lit(None).cast("string").alias("size_category"), "contract_bucket",
            "n_projects", "avg_rebaselines", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max_rebaselines")
)

# By SBU × Asset (if sufficient data)
rebaseline_by_sbu_asset = (
    rebaseline_counts
    .filter(F.col("cat_main_asset_class").isNotNull())
    .groupBy("cat_sbu", "cat_main_asset_class")
    .agg(
        F.count("*").alias("n_projects"),
        F.avg("n_rebaselines").alias("avg_rebaselines"),
        F.percentile_approx("n_rebaselines", 0.01).alias("p1"),
        F.percentile_approx("n_rebaselines", 0.05).alias("p5"),
        F.percentile_approx("n_rebaselines", 0.25).alias("p25"),
        F.percentile_approx("n_rebaselines", 0.50).alias("p50"),
        F.percentile_approx("n_rebaselines", 0.75).alias("p75"),
        F.percentile_approx("n_rebaselines", 0.95).alias("p95"),
        F.percentile_approx("n_rebaselines", 0.99).alias("p99"),
        F.max("n_rebaselines").alias("max_rebaselines")
    )
    .filter(F.col("n_projects") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU_ASSET"))
    .select("level", "cat_sbu", "cat_main_asset_class", 
            F.lit(None).cast("string").alias("size_category"),
            F.lit(None).cast("string").alias("contract_bucket"),
            "n_projects", "avg_rebaselines", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max_rebaselines")
)

# Combine rebaseline count profiles
rebaseline_count_profile = (
    rebaseline_by_sbu
    .unionByName(rebaseline_by_sbu_size, allowMissingColumns=True)
    .unionByName(rebaseline_by_sbu_contract, allowMissingColumns=True)
    .unionByName(rebaseline_by_sbu_asset, allowMissingColumns=True)
)

# Delay Magnitude Profiling
print("Profiling delay magnitudes...")

# Calculate delays between consecutive baseline versions
w_version = W.partitionBy("ids_project_number").orderBy("version_seq")
version_delays = (
    distinct_baselines
    .withColumn("prev_tco", F.lag("dt_tco").over(w_version))
    .withColumn("delay_days", F.datediff("dt_tco", "prev_tco"))
    .filter(F.col("delay_days").isNotNull())
    .filter(F.col("delay_days") > 0)  # Only delays, exclude accelerations
)

# Add delay category
version_delays = version_delays.withColumn(
    "delay_category",
    F.when(F.col("delay_days") < 7, "minimal")
     .when(F.col("delay_days") < 30, "small")
     .when(F.col("delay_days") < 60, "moderate")
     .when(F.col("delay_days") < 120, "major")
     .when(F.col("delay_days") < 180, "large")
     .otherwise("severe")
)

# Profile delays by SBU
delay_by_sbu = (
    version_delays
    .groupBy("cat_sbu", "delay_category")
    .agg(
        F.count("*").alias("n_delays"),
        F.percentile_approx("delay_days", 0.01).alias("p1"),
        F.percentile_approx("delay_days", 0.05).alias("p5"),
        F.percentile_approx("delay_days", 0.25).alias("p25"),
        F.percentile_approx("delay_days", 0.50).alias("p50"),
        F.percentile_approx("delay_days", 0.75).alias("p75"),
        F.percentile_approx("delay_days", 0.95).alias("p95"),
        F.percentile_approx("delay_days", 0.99).alias("p99"),
        F.min("delay_days").alias("min_delay"),
        F.max("delay_days").alias("max_delay")
    )
    .filter(F.col("n_delays") >= 3)
    .withColumn("level", F.lit("SBU"))
)

# Profile delays by SBU × Size
delay_by_sbu_size = (
    version_delays
    .filter(F.col("size_category").isNotNull())
    .groupBy("cat_sbu", "size_category", "delay_category")
    .agg(
        F.count("*").alias("n_delays"),
        F.percentile_approx("delay_days", 0.01).alias("p1"),
        F.percentile_approx("delay_days", 0.05).alias("p5"),
        F.percentile_approx("delay_days", 0.25).alias("p25"),
        F.percentile_approx("delay_days", 0.50).alias("p50"),
        F.percentile_approx("delay_days", 0.75).alias("p75"),
        F.percentile_approx("delay_days", 0.95).alias("p95"),
        F.percentile_approx("delay_days", 0.99).alias("p99"),
        F.min("delay_days").alias("min_delay"),
        F.max("delay_days").alias("max_delay")
    )
    .filter(F.col("n_delays") >= 3)
    .withColumn("level", F.lit("SBU_SIZE"))
)

# Profile delays by SBU × Contract
delay_by_sbu_contract = (
    version_delays
    .filter(F.col("contract_bucket").isNotNull())
    .groupBy("cat_sbu", "contract_bucket", "delay_category")
    .agg(
        F.count("*").alias("n_delays"),
        F.percentile_approx("delay_days", 0.01).alias("p1"),
        F.percentile_approx("delay_days", 0.05).alias("p5"),
        F.percentile_approx("delay_days", 0.25).alias("p25"),
        F.percentile_approx("delay_days", 0.50).alias("p50"),
        F.percentile_approx("delay_days", 0.75).alias("p75"),
        F.percentile_approx("delay_days", 0.95).alias("p95"),
        F.percentile_approx("delay_days", 0.99).alias("p99"),
        F.min("delay_days").alias("min_delay"),
        F.max("delay_days").alias("max_delay")
    )
    .filter(F.col("n_delays") >= 3)
    .withColumn("level", F.lit("SBU_CONTRACT"))
)

# Global delay distribution (fallback)
delay_global = (
    version_delays
    .groupBy("delay_category")
    .agg(
        F.count("*").alias("n_delays"),
        F.percentile_approx("delay_days", 0.01).alias("p1"),
        F.percentile_approx("delay_days", 0.05).alias("p5"),
        F.percentile_approx("delay_days", 0.25).alias("p25"),
        F.percentile_approx("delay_days", 0.50).alias("p50"),
        F.percentile_approx("delay_days", 0.75).alias("p75"),
        F.percentile_approx("delay_days", 0.95).alias("p95"),
        F.percentile_approx("delay_days", 0.99).alias("p99"),
        F.min("delay_days").alias("min_delay"),
        F.max("delay_days").alias("max_delay")
    )
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("cat_sbu", F.lit(None).cast("string"))
    .withColumn("cat_main_asset_class", F.lit(None).cast("string"))
    .withColumn("size_category", F.lit(None).cast("string"))
    .withColumn("contract_bucket", F.lit(None).cast("string"))
)

# Rebaseline Timing Profiling
print("Profiling rebaseline timing...")

# Clarifying comment about timing semantics:
# normalized_timing = (rebaseline_date - project_start) / (project_end - project_start)
# Interpretation: 0.0 = project start, 0.5 = midpoint, 1.0 = project end
# This tells us "when in the project lifecycle" rebaselines typically occur
# Example: median 0.486 for 1st rebaseline = happens about halfway through project
rebaseline_timing = (
    version_delays
    .filter(F.col("dt_construction_start").isNotNull())
    .filter(F.col("dt_construction_end").isNotNull())
    .withColumn("project_duration_days", 
                F.datediff("dt_construction_end", "dt_construction_start"))
    .filter(F.col("project_duration_days") > 0)
    .withColumn("days_from_start", 
                F.datediff("dt_effective_from", "dt_construction_start"))
    .withColumn("normalized_timing",
                F.col("days_from_start") / F.col("project_duration_days"))
)

# Profile timing by rebaseline sequence (1st, 2nd, 3rd rebaseline)
timing_by_sequence = (
    rebaseline_timing
    .groupBy("version_seq")
    .agg(
        F.count("*").alias("n_rebaselines"),
        F.avg("normalized_timing").alias("avg_timing"),
        F.percentile_approx("normalized_timing", 0.01).alias("p1"),
        F.percentile_approx("normalized_timing", 0.05).alias("p5"),
        F.percentile_approx("normalized_timing", 0.25).alias("p25"),
        F.percentile_approx("normalized_timing", 0.50).alias("p50"),
        F.percentile_approx("normalized_timing", 0.75).alias("p75"),
        F.percentile_approx("normalized_timing", 0.95).alias("p95"),
        F.percentile_approx("normalized_timing", 0.99).alias("p99")
    )
    .filter(F.col("version_seq") > 1)  # Exclude original baseline (seq=1)
    .withColumn("rebaseline_number", F.col("version_seq") - 1)
    .filter(F.col("rebaseline_number") <= 5)  # Focus on first 5 rebaselines
    .withColumn("level", F.lit("TIMING"))
)

# Milestone Date Patterns
print("Profiling milestone date patterns...")

# Use only active baseline versions
active_baselines = (
    distinct_baselines
    .filter(F.col("is_active_record") == True)
    .filter(F.col("dt_ntp").isNotNull())
    .filter(F.col("dt_tco").isNotNull())
)

# Calculate all durations (describe reality - including invalid values)
active_baselines = active_baselines.withColumn(
    "ntp_to_tco_days",
    F.datediff("dt_tco", "dt_ntp")
)

active_baselines = active_baselines.withColumn(
    "tco_to_co_days",
    F.when(F.col("dt_co").isNotNull(), F.datediff("dt_co", "dt_tco"))
)

active_baselines = active_baselines.withColumn(
    "tco_to_next_tco_days",
    F.when(F.col("dt_next_tco").isNotNull(), F.datediff("dt_next_tco", "dt_tco"))
)

# Flag validity (constraints decide what we allow to exist)
active_baselines = active_baselines.withColumn(
    "is_valid_ntp_tco",
    (F.col("ntp_to_tco_days").isNotNull()) & (F.col("ntp_to_tco_days") >= 30) 
)

active_baselines = active_baselines.withColumn(
    "is_valid_tco_co",
    (F.col("tco_to_co_days").isNotNull()) & (F.col("tco_to_co_days") >= 14) 
)

# Log invalidity statistics (describe reality)
total_active = active_baselines.count()
invalid_ntp_tco = active_baselines.filter(~F.col("is_valid_ntp_tco")).count()
invalid_tco_co = active_baselines.filter(~F.col("is_valid_tco_co")).count()

print(f"\n=== DATA QUALITY ASSESSMENT ===")
print(f"Total active baselines: {total_active}")
print(f"Invalid NTP→TCO (< 30 days or negative): {invalid_ntp_tco} ({invalid_ntp_tco/total_active*100:.1f}%)")
print(f"Invalid TCO→CO (< 14 days or negative): {invalid_tco_co} ({invalid_tco_co/total_active*100:.1f}%)")

# Show raw distribution (reality as-is, including invalid)
print("\nRaw NTP→TCO distribution (all data, including invalid):")
active_baselines.agg(
    F.min("ntp_to_tco_days").alias("min"),
    F.percentile_approx("ntp_to_tco_days", 0.50).alias("median"),
    F.max("ntp_to_tco_days").alias("max")
).show(truncate=False)

# Profile valid subset only (constraints-compliant data for generation)
valid_for_profiling = active_baselines.filter(F.col("is_valid_ntp_tco"))

print(f"Profiling milestone offsets from {valid_for_profiling.count()} valid baselines (out of {total_active} total)...\n")

# Profile milestone offsets by SBU (from valid data only)
milestone_fields = ["ntp_to_tco_days", "tco_to_co_days", "tco_to_next_tco_days"]

# Updated function to exclude negative values explicitly
def build_milestone_profile(df_in, level_label, group_cols):
    """Profile milestone offset patterns at given aggregation level"""
    pieces = []
    for field in milestone_fields:
        stats = (
            df_in
            .filter(F.col(field).isNotNull())
            .filter(F.col(field) >= 0)  # Exclude negative values (invalid date ordering)
            .groupBy(*group_cols)
            .agg(
                F.count(field).alias("n_nonnull"),
                F.min(field).alias("min"),
                F.percentile_approx(field, 0.01).alias("p1"),
                F.percentile_approx(field, 0.05).alias("p5"),
                F.percentile_approx(field, 0.25).alias("p25"),
                F.percentile_approx(field, 0.50).alias("median"),
                F.percentile_approx(field, 0.75).alias("p75"),
                F.percentile_approx(field, 0.95).alias("p95"),
                F.percentile_approx(field, 0.99).alias("p99"),
                F.max(field).alias("max"),
                F.avg(field).alias("mean"),
                F.stddev_samp(field).alias("stddev")
            )
            .withColumn("field", F.lit(field))
            .withColumn("level", F.lit(level_label))
        )
        pieces.append(stats)
    
    result = pieces[0]
    for p in pieces[1:]:
        result = result.unionByName(p, allowMissingColumns=True)
    
    return result

# Profile at SBU level (from valid data)
milestone_sbu = build_milestone_profile(valid_for_profiling, "SBU", ["cat_sbu"])

# Profile at SBU × Asset level (from valid data)
valid_sa = (
    valid_for_profiling
    .filter(F.col("cat_main_asset_class").isNotNull())
    .groupBy("cat_sbu", "cat_main_asset_class")
    .agg(F.count("*").alias("n_group"))
    .filter(F.col("n_group") >= MIN_GROUP_ROWS)
    .select("cat_sbu", "cat_main_asset_class")
)

milestone_sa = build_milestone_profile(
    valid_for_profiling.join(valid_sa, ["cat_sbu", "cat_main_asset_class"], "inner"),
    "SBU_ASSET",
    ["cat_sbu", "cat_main_asset_class"]
)

# Consolidate Profile Table
print("Consolidating profile table...")

# Rebaseline count profiles (convert to field-based format with consistent p1-p99 range)
rebaseline_profile_formatted = (
    rebaseline_by_sbu
    .withColumn("field", F.lit("n_rebaselines"))
    .select(
        "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
        "n_projects", 
        F.lit(None).cast("double").alias("pct_missing"),
        F.lit(None).cast("double").alias("min"),
        F.col("p1"),
        F.col("p5"),
        F.col("p25"),
        F.col("p50").alias("median"),
        F.col("p75"),
        F.col("p95"),
        F.col("p99"),
        F.col("max_rebaselines").alias("max"),
        F.col("avg_rebaselines").alias("mean"),
        F.lit(None).cast("double").alias("stddev")
    )
    .unionByName(
        rebaseline_by_sbu_size
        .withColumn("field", F.lit("n_rebaselines"))
        .select(
            "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
            "n_projects", 
            F.lit(None).cast("double").alias("pct_missing"),
            F.lit(None).cast("double").alias("min"),
            F.col("p1"),
            F.col("p5"),
            F.col("p25"),
            F.col("p50").alias("median"),
            F.col("p75"),
            F.col("p95"),
            F.col("p99"),
            F.col("max_rebaselines").alias("max"),
            F.col("avg_rebaselines").alias("mean"),
            F.lit(None).cast("double").alias("stddev")
        ),
        allowMissingColumns=True
    )
    .unionByName(
        rebaseline_by_sbu_contract
        .withColumn("field", F.lit("n_rebaselines"))
        .select(
            "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
            "n_projects", 
            F.lit(None).cast("double").alias("pct_missing"),
            F.lit(None).cast("double").alias("min"),
            F.col("p1"),
            F.col("p5"),
            F.col("p25"),
            F.col("p50").alias("median"),
            F.col("p75"),
            F.col("p95"),
            F.col("p99"),
            F.col("max_rebaselines").alias("max"),
            F.col("avg_rebaselines").alias("mean"),
            F.lit(None).cast("double").alias("stddev")
        ),
        allowMissingColumns=True
    )
    .unionByName(
        rebaseline_by_sbu_asset
        .withColumn("field", F.lit("n_rebaselines"))
        .select(
            "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
            "n_projects", 
            F.lit(None).cast("double").alias("pct_missing"),
            F.lit(None).cast("double").alias("min"),
            F.col("p1"),
            F.col("p5"),
            F.col("p25"),
            F.col("p50").alias("median"),
            F.col("p75"),
            F.col("p95"),
            F.col("p99"),
            F.col("max_rebaselines").alias("max"),
            F.col("avg_rebaselines").alias("mean"),
            F.lit(None).cast("double").alias("stddev")
        ),
        allowMissingColumns=True
    )
)

# Milestone offset profiles (now with consistent p1-p99 range)
milestone_profile_formatted = (
    milestone_sbu
    .withColumn("size_category", F.lit(None).cast("string"))
    .withColumn("cat_main_asset_class", F.lit(None).cast("string"))
    .withColumn("contract_bucket", F.lit(None).cast("string"))
    .withColumn("pct_missing", F.lit(0.0))
    .select(
        "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
        "n_nonnull", "pct_missing",
        F.round("min", 2).alias("min"),
        F.round("p1", 2).alias("p1"),
        F.round("p5", 2).alias("p5"),
        F.round("p25", 2).alias("p25"),
        F.round("median", 2).alias("median"),
        F.round("p75", 2).alias("p75"),
        F.round("p95", 2).alias("p95"),
        F.round("p99", 2).alias("p99"),
        F.round("max", 2).alias("max"),
        F.round("mean", 2).alias("mean"),
        F.round("stddev", 2).alias("stddev")
    )
    .withColumnRenamed("n_nonnull", "n_projects")
    .unionByName(
        milestone_sa
        .withColumn("size_category", F.lit(None).cast("string"))
        .withColumn("contract_bucket", F.lit(None).cast("string"))
        .withColumn("pct_missing", F.lit(0.0))
        .select(
            "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket", "field",
            "n_nonnull", "pct_missing",
            F.round("min", 2).alias("min"),
            F.round("p1", 2).alias("p1"),
            F.round("p5", 2).alias("p5"),
            F.round("p25", 2).alias("p25"),
            F.round("median", 2).alias("median"),
            F.round("p75", 2).alias("p75"),
            F.round("p95", 2).alias("p95"),
            F.round("p99", 2).alias("p99"),
            F.round("max", 2).alias("max"),
            F.round("mean", 2).alias("mean"),
            F.round("stddev", 2).alias("stddev")
        )
        .withColumnRenamed("n_nonnull", "n_projects"),
        allowMissingColumns=True
    )
)

# Combine all profiles
profile_all = (
    rebaseline_profile_formatted
    .unionByName(milestone_profile_formatted, allowMissingColumns=True)
)

# Save profile table
(profile_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_PROFILE))

print(f"Saved consolidated profile to: {OUT_PROFILE}")
print("Building constraints table...")

# Delay magnitude bounds (with full p1-p99 range)
delay_constraints = (
    delay_by_sbu
    .withColumn(
        "bounds_json",
        F.to_json(F.create_map(
            F.lit("p1"), F.round(F.col("p1"), 2),
            F.lit("p5"), F.round(F.col("p5"), 2),
            F.lit("p25"), F.round(F.col("p25"), 2),
            F.lit("p50"), F.round(F.col("p50"), 2),
            F.lit("p75"), F.round(F.col("p75"), 2),
            F.lit("p95"), F.round(F.col("p95"), 2),
            F.lit("p99"), F.round(F.col("p99"), 2),
            F.lit("min"), F.round(F.col("min_delay"), 2),
            F.lit("max"), F.round(F.col("max_delay"), 2)
        ))
    )
    .withColumn("field", F.concat(F.lit("delay_"), F.col("delay_category")))
    .select(
        "level", "cat_sbu", 
        F.lit(None).cast("string").alias("cat_main_asset_class"),
        F.lit(None).cast("string").alias("size_category"),
        F.lit(None).cast("string").alias("contract_bucket"),
        "field", "n_delays", "bounds_json"
    )
    .withColumnRenamed("n_delays", "n")
    .unionByName(
        delay_by_sbu_size
        .withColumn(
            "bounds_json",
            F.to_json(F.create_map(
                F.lit("p1"), F.round(F.col("p1"), 2),
                F.lit("p5"), F.round(F.col("p5"), 2),
                F.lit("p25"), F.round(F.col("p25"), 2),
                F.lit("p50"), F.round(F.col("p50"), 2),
                F.lit("p75"), F.round(F.col("p75"), 2),
                F.lit("p95"), F.round(F.col("p95"), 2),
                F.lit("p99"), F.round(F.col("p99"), 2),
                F.lit("min"), F.round(F.col("min_delay"), 2),
                F.lit("max"), F.round(F.col("max_delay"), 2)
            ))
        )
        .withColumn("field", F.concat(F.lit("delay_"), F.col("delay_category")))
        .select(
            "level", "cat_sbu", 
            F.lit(None).cast("string").alias("cat_main_asset_class"),
            "size_category", 
            F.lit(None).cast("string").alias("contract_bucket"),
            "field", "n_delays", "bounds_json"
        )
        .withColumnRenamed("n_delays", "n"),
        allowMissingColumns=True
    )
    .unionByName(
        delay_by_sbu_contract
        .withColumn(
            "bounds_json",
            F.to_json(F.create_map(
                F.lit("p1"), F.round(F.col("p1"), 2),
                F.lit("p5"), F.round(F.col("p5"), 2),
                F.lit("p25"), F.round(F.col("p25"), 2),
                F.lit("p50"), F.round(F.col("p50"), 2),
                F.lit("p75"), F.round(F.col("p75"), 2),
                F.lit("p95"), F.round(F.col("p95"), 2),
                F.lit("p99"), F.round(F.col("p99"), 2),
                F.lit("min"), F.round(F.col("min_delay"), 2),
                F.lit("max"), F.round(F.col("max_delay"), 2)
            ))
        )
        .withColumn("field", F.concat(F.lit("delay_"), F.col("delay_category")))
        .select(
            "level", "cat_sbu", 
            F.lit(None).cast("string").alias("cat_main_asset_class"),
            F.lit(None).cast("string").alias("size_category"),
            "contract_bucket",
            "field", "n_delays", "bounds_json"
        )
        .withColumnRenamed("n_delays", "n"),
        allowMissingColumns=True
    )
    .unionByName(
        delay_global
        .withColumn(
            "bounds_json",
            F.to_json(F.create_map(
                F.lit("p1"), F.round(F.col("p1"), 2),
                F.lit("p5"), F.round(F.col("p5"), 2),
                F.lit("p25"), F.round(F.col("p25"), 2),
                F.lit("p50"), F.round(F.col("p50"), 2),
                F.lit("p75"), F.round(F.col("p75"), 2),
                F.lit("p95"), F.round(F.col("p95"), 2),
                F.lit("p99"), F.round(F.col("p99"), 2),
                F.lit("min"), F.round(F.col("min_delay"), 2),
                F.lit("max"), F.round(F.col("max_delay"), 2)
            ))
        )
        .withColumn("field", F.concat(F.lit("delay_"), F.col("delay_category")))
        .select(
            "level", "cat_sbu", 
            F.lit(None).cast("string").alias("cat_main_asset_class"),
            F.lit(None).cast("string").alias("size_category"),
            F.lit(None).cast("string").alias("contract_bucket"),
            "field", "n_delays", "bounds_json"
        )
        .withColumnRenamed("n_delays", "n"),
        allowMissingColumns=True
    )
)
rebaseline_count_dist = (
    rebaseline_count_profile
    .withColumn(
        "dist_json",
        F.to_json(F.create_map(
            F.lit("p1"), F.col("p1"),
            F.lit("p5"), F.col("p5"),
            F.lit("p25"), F.col("p25"),
            F.lit("p50"), F.col("p50"),
            F.lit("p75"), F.col("p75"),
            F.lit("p95"), F.col("p95"),
            F.lit("p99"), F.col("p99"),
            F.lit("max"), F.col("max_rebaselines"),
            F.lit("mean"), F.round(F.col("avg_rebaselines"), 2)
        ))
    )
    .withColumn("field", F.lit("rebaseline_count_distribution"))
    .select(
        "level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket",
        "field", "n_projects", 
        F.col("dist_json").alias("bounds_json")
    )
    .withColumnRenamed("n_projects", "n")
)

# Rebaseline timing distributions (with full p1-p99 range)
timing_constraints = (
    timing_by_sequence
    .withColumn(
        "timing_json",
        F.to_json(F.create_map(
            F.lit("p1"), F.round(F.col("p1"), 3),
            F.lit("p5"), F.round(F.col("p5"), 3),
            F.lit("p25"), F.round(F.col("p25"), 3),
            F.lit("p50"), F.round(F.col("p50"), 3),
            F.lit("p75"), F.round(F.col("p75"), 3),
            F.lit("p95"), F.round(F.col("p95"), 3),
            F.lit("p99"), F.round(F.col("p99"), 3),
            F.lit("mean"), F.round(F.col("avg_timing"), 3)
        ))
    )
    .withColumn("field", F.concat(F.lit("timing_rebaseline_"), F.col("rebaseline_number").cast("string")))
    .select(
        "level",
        F.lit(None).cast("string").alias("cat_sbu"),
        F.lit(None).cast("string").alias("cat_main_asset_class"),
        F.lit(None).cast("string").alias("size_category"),
        F.lit(None).cast("string").alias("contract_bucket"),
        "field", "n_rebaselines",
        F.col("timing_json").alias("bounds_json")
    )
    .withColumnRenamed("n_rebaselines", "n")
)

# Milestone offset bounds (with full p1-p99 range)
milestone_bounds = (
    milestone_profile_formatted
    .withColumn(
        "bounds_json",
        F.to_json(F.create_map(
            F.lit("p1"), F.col("p1"),
            F.lit("p5"), F.col("p5"),
            F.lit("p25"), F.col("p25"),
            F.lit("median"), F.col("median"),
            F.lit("p75"), F.col("p75"),
            F.lit("p95"), F.col("p95"),
            F.lit("p99"), F.col("p99"),
            F.lit("min"), F.col("min"),
            F.lit("max"), F.col("max"),
            F.lit("mean"), F.col("mean")
        ))
    )
    .select("level", "cat_sbu", "cat_main_asset_class", "size_category", "contract_bucket",
            "field", "n_projects", "bounds_json")
    .withColumnRenamed("n_projects", "n")
)

# Combine all constraints
constraints_all = (
    delay_constraints
    .unionByName(rebaseline_count_dist, allowMissingColumns=True)
    .unionByName(timing_constraints, allowMissingColumns=True)
    .unionByName(milestone_bounds, allowMissingColumns=True)
)

# Save constraints table
(constraints_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_CONSTRAINTS))

print(f"Saved constraints to: {OUT_CONSTRAINTS}")
# Summary Statistics
print("\n=== PROFILING SUMMARY ===")
print(f"Total projects analyzed: {df.select('ids_project_number').distinct().count()}")
print(f"Total baseline versions: {distinct_baselines.count()}")
print(f"Projects with rebaselines: {rebaseline_counts.filter(F.col('n_rebaselines') > 0).count()}")
print(f"Total delay observations: {version_delays.count()}")
print(f"\nProfile table rows: {profile_all.count()}")
print(f"Constraints table rows: {constraints_all.count()}")
print("\nProfiling complete!")