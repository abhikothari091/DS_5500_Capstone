from pyspark.sql import functions as F, Window as W

# Configuration
SOURCE_TASK = "platinum.models.task_baseline"
SOURCE_PROJECT = "platinum.models.project"
OUT_PROFILE = "analysts.self_managed.task_summary_profile"
OUT_CONSTRAINTS = "analysts.self_managed.task_summary_constraints"
OUT_SOW_FREQUENCY = "analysts.self_managed.task_summary_sow_frequency"
OUT_SOW_OVERLAPS = "analysts.self_managed.task_summary_sow_overlaps"
OUT_COMPLETION_PROGRESSION = "analysts.self_managed.task_summary_completion_progression"
OUT_DATE_SHIFT_PROFILES = "analysts.self_managed.task_summary_date_shift_profiles"
MIN_GROUP_ROWS = 5
SOW_COVERAGE_THRESHOLD = 75.0

# Reusable Profiling Helpers
def profile_percentiles(df, group_cols, value_col, min_rows=None):
    """Standard percentile profiling used across all sections."""
    min_rows = min_rows or MIN_GROUP_ROWS
    result = (
        df.groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_projects"),
            F.percentile_approx(value_col, 0.01).alias("p1"),
            F.percentile_approx(value_col, 0.05).alias("p5"),
            F.percentile_approx(value_col, 0.10).alias("p10"),
            F.percentile_approx(value_col, 0.25).alias("p25"),
            F.percentile_approx(value_col, 0.50).alias("median"),
            F.percentile_approx(value_col, 0.75).alias("p75"),
            F.percentile_approx(value_col, 0.90).alias("p90"),
            F.percentile_approx(value_col, 0.95).alias("p95"),
            F.percentile_approx(value_col, 0.99).alias("p99"),
            F.min(value_col).alias("min"),
            F.max(value_col).alias("max"),
            F.avg(value_col).alias("mean"),
            F.stddev_samp(value_col).alias("stddev")
        )
        .filter(F.col("n_projects") >= min_rows)
    )
    return result

def profile_percentiles_short(df, group_cols, value_col, min_rows=None):
    """Shorter percentile set for secondary profiles."""
    min_rows = min_rows or MIN_GROUP_ROWS
    return (
        df.groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_observations"),
            F.percentile_approx(value_col, 0.05).alias("p5"),
            F.percentile_approx(value_col, 0.10).alias("p10"),
            F.percentile_approx(value_col, 0.25).alias("p25"),
            F.percentile_approx(value_col, 0.50).alias("median"),
            F.percentile_approx(value_col, 0.75).alias("p75"),
            F.percentile_approx(value_col, 0.90).alias("p90"),
            F.percentile_approx(value_col, 0.95).alias("p95"),
            F.avg(value_col).alias("mean"),
            F.min(value_col).alias("min"),
            F.max(value_col).alias("max")
        )
        .filter(F.col("n_observations") >= min_rows)
    )

# Data Loading & Quality Filters
print("Loading task baseline and project data...")

task_df = spark.read.table(SOURCE_TASK).select(
    "ids_project_number", "cat_task_code", "str_task_name",
    "cat_sow", "dt_task_start", "dt_task_end",
    "cat_task_start_source", "amt_duration_target",
    "amt_duration_remaining", "is_active_record",
    "dt_effective_from"
)

task_active_df = task_df.filter(F.col("is_active_record") == True)

project_df = spark.read.table(SOURCE_PROJECT).select(
    "ids_project_number", "is_latest", "cat_sbu", "cat_region",
    "cat_main_asset_class", "amt_area_gross", "amt_contract",
    "dt_construction_start", "dt_construction_end"
).filter(F.col("is_latest") == True)

project_df = (
    project_df
    .filter(F.col("amt_contract") >= 25000000)
    .filter(F.year("dt_construction_start") >= 2019)
    .filter(F.col("dt_construction_start").isNotNull())
    .filter(F.col("cat_sbu").isNotNull())
)

# SOW coverage filtering
project_sow_coverage = (
    task_active_df
    .join(project_df, "ids_project_number", "inner")
    .groupBy("ids_project_number")
    .agg(
        F.count("cat_task_code").alias("total_tasks"),
        F.sum(F.when(F.col("cat_sow").isNotNull(), 1).otherwise(0)).alias("tasks_with_sow"),
        F.countDistinct("cat_sow").alias("n_distinct_sows")
    )
    .withColumn("pct_sow_coverage",
        F.round(F.col("tasks_with_sow") * 100.0 / F.col("total_tasks"), 2))
)

valid_projects = (
    project_sow_coverage
    .filter(F.col("pct_sow_coverage") >= SOW_COVERAGE_THRESHOLD)
    .filter(F.col("total_tasks") >= 100)
    .filter(F.col("n_distinct_sows") >= 5)
    .select("ids_project_number")
)

valid_projects_count = valid_projects.count()
print(f"Projects meeting SOW coverage threshold (>={SOW_COVERAGE_THRESHOLD}%): {valid_projects_count}")

# Build main working dataframe (active records only for V1 profiling)
df = (
    task_active_df
    .join(valid_projects, "ids_project_number", "inner")
    .join(project_df, "ids_project_number", "inner")
    .withColumn("dt_task_start", F.to_date("dt_task_start"))
    .withColumn("dt_task_end", F.to_date("dt_task_end"))
    .withColumn("dt_construction_start", F.to_date("dt_construction_start"))
    .withColumn("dt_construction_end", F.to_date("dt_construction_end"))
    .withColumn("size_category",
        F.when(F.col("amt_area_gross") < 50000, "Small")
         .when(F.col("amt_area_gross") < 100000, "Medium")
         .when(F.col("amt_area_gross") < 200000, "Large")
         .otherwise("XLarge"))
    .withColumn("contract_bucket",
        F.when(F.col("amt_contract") < 50000000, "Medium")
         .when(F.col("amt_contract") < 100000000, "Large")
         .when(F.col("amt_contract") < 200000000, "XLarge")
         .otherwise("XXLarge"))
)

print(f"Total active tasks in profiling dataset: {df.count()}")
print(f"Active tasks with non-NULL SOW: {df.filter(F.col('cat_sow').isNotNull()).count()}")

# SOW Count Profiling
print("\nProfiling SOW counts per project...")

sow_counts = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.trim(F.col("cat_sow")) != "")
    .groupBy("ids_project_number", "cat_sbu", "cat_main_asset_class",
             "size_category", "contract_bucket")
    .agg(F.countDistinct("cat_sow").alias("n_sows"))
)

sow_count_by_sbu = (
    profile_percentiles(sow_counts, ["cat_sbu"], "n_sows")
    .withColumn("level", F.lit("SBU"))
    .withColumn("field", F.lit("n_sows_per_project"))
)

sow_count_by_sbu_size = (
    profile_percentiles(
        sow_counts.filter(F.col("size_category").isNotNull()),
        ["cat_sbu", "size_category"], "n_sows"
    )
    .withColumn("level", F.lit("SBU_SIZE"))
    .withColumn("field", F.lit("n_sows_per_project"))
)

sow_count_by_sbu_contract = (
    profile_percentiles(
        sow_counts.filter(F.col("contract_bucket").isNotNull()),
        ["cat_sbu", "contract_bucket"], "n_sows"
    )
    .withColumn("level", F.lit("SBU_CONTRACT"))
    .withColumn("field", F.lit("n_sows_per_project"))
)

# SOW Type Frequency & Sequence Position
print("\nProfiling SOW types and sequence positions...")

sow_timing = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.trim(F.col("cat_sow")) != "")
    .filter(F.col("dt_task_start").isNotNull())
    .filter(F.col("dt_task_end").isNotNull())
    .groupBy("ids_project_number", "cat_sow", "cat_sbu", "cat_main_asset_class",
             "size_category", "contract_bucket")
    .agg(
        F.min("dt_task_start").alias("sow_start"),
        F.max("dt_task_end").alias("sow_finish")
    )
    .withColumn("sow_duration_days", F.datediff("sow_finish", "sow_start"))
    .filter(F.col("sow_duration_days") >= 0)
)

sow_with_sequence = (
    sow_timing.withColumn("sequence_order",
        F.row_number().over(
            W.partitionBy("ids_project_number")
             .orderBy(F.col("sow_start"), F.col("cat_sow"))))
)

# Sequence by SOW × SBU
# Helper for sequence profiling (uses mean/stddev instead of full percentile set)
def profile_sequence(df, group_cols, min_rows=5):
    return (
        df.groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_occurrences"),
            F.countDistinct("ids_project_number").alias("n_projects"),
            F.percentile_approx("sequence_order", 0.10).alias("p10"),
            F.percentile_approx("sequence_order", 0.25).alias("p25"),
            F.percentile_approx("sequence_order", 0.50).alias("median"),
            F.percentile_approx("sequence_order", 0.75).alias("p75"),
            F.percentile_approx("sequence_order", 0.90).alias("p90"),
            F.avg("sequence_order").alias("mean"),
            F.stddev_samp("sequence_order").alias("stddev"),
            F.min("sequence_order").alias("min"),
            F.max("sequence_order").alias("max")
        )
        .filter(F.col("n_projects") >= min_rows)
    )

sequence_by_sow_sbu = (
    profile_sequence(sow_with_sequence, ["cat_sow", "cat_sbu"])
    .withColumn("level", F.lit("SOW_SBU"))
    .withColumn("field", F.lit("sequence_position"))
)

sequence_global = (
    profile_sequence(sow_with_sequence, ["cat_sow"], min_rows=10)
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("field", F.lit("sequence_position"))
)

# SOW frequency with correct denominator
print("\nProfiling SOW type frequency...")

sbu_project_counts = (
    sow_with_sequence.groupBy("cat_sbu")
    .agg(F.countDistinct("ids_project_number").alias("total_projects_in_sbu"))
)

sow_frequency_by_sbu = (
    sow_with_sequence.groupBy("cat_sbu", "cat_sow")
    .agg(
        F.countDistinct("ids_project_number").alias("n_projects"),
        F.count("*").alias("n_occurrences")
    )
    .join(sbu_project_counts, "cat_sbu", "inner")
    .withColumn("pct_projects",
        F.round(F.col("n_projects") * 100.0 / F.col("total_projects_in_sbu"), 2))
    .select("cat_sbu", "cat_sow", "n_projects", "total_projects_in_sbu",
            "n_occurrences", "pct_projects")
)

(sow_frequency_by_sbu
 .select("cat_sbu", "cat_sow", "n_projects", "pct_projects")
 .write.format("delta").mode("overwrite")
 .saveAsTable(OUT_SOW_FREQUENCY))

print(f"Saved SOW frequency to: {OUT_SOW_FREQUENCY}")


# Task Count Profiling
print("\nProfiling task counts per SOW...")

task_counts_per_sow = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.trim(F.col("cat_sow")) != "")
    .groupBy("ids_project_number", "cat_sow", "cat_sbu", "size_category", "contract_bucket")
    .agg(F.countDistinct("cat_task_code").alias("n_tasks"))
)

task_count_by_sow = (
    profile_percentiles_short(task_counts_per_sow, ["cat_sow"], "n_tasks", min_rows=10)
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("n_tasks"))
)

task_count_by_sow_sbu = (
    profile_percentiles_short(task_counts_per_sow, ["cat_sow", "cat_sbu"], "n_tasks")
    .withColumn("level", F.lit("SOW_SBU"))
    .withColumn("field", F.lit("n_tasks"))
)

# SOW Duration Profiling
print("\nProfiling SOW durations...")

duration_by_sow = (
    profile_percentiles_short(sow_timing, ["cat_sow"], "sow_duration_days", min_rows=10)
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("sow_duration_days"))
)

duration_by_sow_sbu = (
    profile_percentiles_short(sow_timing, ["cat_sow", "cat_sbu"], "sow_duration_days")
    .withColumn("level", F.lit("SOW_SBU"))
    .withColumn("field", F.lit("sow_duration_days"))
)

# SOW Overlap Profiling (Consecutive Pairs)
print("\nProfiling SOW temporal overlaps...")

consecutive_sows = (
    sow_with_sequence.alias("s1")
    .join(
        sow_with_sequence.alias("s2"),
        (F.col("s1.ids_project_number") == F.col("s2.ids_project_number"))
        & (F.col("s2.sequence_order") == F.col("s1.sequence_order") + 1),
        "inner"
    )
    .select(
        F.col("s1.ids_project_number"),
        F.col("s1.cat_sow").alias("sow_first"),
        F.col("s2.cat_sow").alias("sow_second"),
        F.col("s1.sow_start").alias("first_start"),
        F.col("s1.sow_finish").alias("first_finish"),
        F.col("s1.sow_duration_days").alias("first_duration"),
        F.col("s2.sow_start").alias("second_start"),
        F.col("s2.sow_finish").alias("second_finish")
    )
    .withColumn("overlap_days", F.datediff("first_finish", "second_start"))
    .withColumn("overlap_pct",
        F.when(F.col("first_duration") > 0,
            F.round((F.col("overlap_days") / F.col("first_duration")) * 100.0, 2)))
)

overlap_general = (
    consecutive_sows.agg(
        F.count("*").alias("n_transitions"),
        F.percentile_approx("overlap_days", 0.25).alias("p25_days"),
        F.percentile_approx("overlap_days", 0.50).alias("median_days"),
        F.percentile_approx("overlap_days", 0.75).alias("p75_days"),
        F.avg("overlap_days").alias("mean_days"),
        F.percentile_approx("overlap_pct", 0.25).alias("p25_pct"),
        F.percentile_approx("overlap_pct", 0.50).alias("median_pct"),
        F.percentile_approx("overlap_pct", 0.75).alias("p75_pct"),
        F.avg("overlap_pct").alias("mean_pct")
    )
    .withColumn("level", F.lit("GENERAL"))
    .withColumn("sow_pair", F.lit("all_pairs"))
)

overlap_by_pair = (
    consecutive_sows.groupBy("sow_first", "sow_second")
    .agg(
        F.count("*").alias("n_transitions"),
        F.percentile_approx("overlap_days", 0.25).alias("p25_days"),
        F.percentile_approx("overlap_days", 0.50).alias("median_days"),
        F.percentile_approx("overlap_days", 0.75).alias("p75_days"),
        F.avg("overlap_days").alias("mean_days"),
        F.percentile_approx("overlap_pct", 0.25).alias("p25_pct"),
        F.percentile_approx("overlap_pct", 0.50).alias("median_pct"),
        F.percentile_approx("overlap_pct", 0.75).alias("p75_pct"),
        F.avg("overlap_pct").alias("mean_pct")
    )
    .filter(F.col("n_transitions") >= 5)
    .withColumn("level", F.lit("PAIR"))
    .withColumn("sow_pair", F.concat(F.col("sow_first"), F.lit(" → "), F.col("sow_second")))
)

(overlap_general
 .unionByName(overlap_by_pair, allowMissingColumns=True)
 .write.format("delta").mode("overwrite")
 .saveAsTable(OUT_SOW_OVERLAPS))

print(f"Saved SOW overlaps to: {OUT_SOW_OVERLAPS}")

# Concurrency Depth Profiling
print("\nProfiling project-level SOW concurrency depth...")

project_dates = (
    sow_with_sequence
    .groupBy("ids_project_number", "cat_sbu", "size_category", "contract_bucket")
    .agg(
        F.min("sow_start").alias("project_start"),
        F.max("sow_finish").alias("project_finish")
    )
)

date_spine = (
    project_dates.select(
        "ids_project_number", "cat_sbu", "size_category", "contract_bucket",
        F.explode(F.expr("sequence(project_start, project_finish, interval 7 days)")).alias("sample_date")
    )
)

concurrent_sows_daily = (
    date_spine.alias("d")
    .join(
        sow_with_sequence.alias("s"),
        (F.col("d.ids_project_number") == F.col("s.ids_project_number"))
        & (F.col("d.sample_date").between(F.col("s.sow_start"), F.col("s.sow_finish"))),
        "left"
    )
    .groupBy("d.ids_project_number", "d.sample_date", "d.cat_sbu",
             "d.size_category", "d.contract_bucket")
    .agg(F.countDistinct("s.cat_sow").alias("n_concurrent_sows"))
)

project_concurrency = (
    concurrent_sows_daily
    .groupBy("ids_project_number", "cat_sbu", "size_category", "contract_bucket")
    .agg(
        F.max("n_concurrent_sows").alias("max_concurrent_sows"),
        F.avg("n_concurrent_sows").alias("avg_concurrent_sows"),
        F.percentile_approx("n_concurrent_sows", 0.75).alias("p75_concurrent_sows"),
        F.percentile_approx("n_concurrent_sows", 0.50).alias("median_concurrent_sows"),
        F.percentile_approx("n_concurrent_sows", 0.25).alias("p25_concurrent_sows")
    )
)

def profile_concurrency(df, group_cols, min_rows=None):
    """Profile both max and avg concurrency in one pass."""
    min_rows = min_rows or MIN_GROUP_ROWS
    return (
        df.groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_projects"),
            F.percentile_approx("max_concurrent_sows", 0.10).alias("p10_max"),
            F.percentile_approx("max_concurrent_sows", 0.25).alias("p25_max"),
            F.percentile_approx("max_concurrent_sows", 0.50).alias("median_max"),
            F.percentile_approx("max_concurrent_sows", 0.75).alias("p75_max"),
            F.percentile_approx("max_concurrent_sows", 0.90).alias("p90_max"),
            F.avg("max_concurrent_sows").alias("mean_max"),
            F.stddev_samp("max_concurrent_sows").alias("stddev_max"),
            F.percentile_approx("avg_concurrent_sows", 0.10).alias("p10_avg"),
            F.percentile_approx("avg_concurrent_sows", 0.25).alias("p25_avg"),
            F.percentile_approx("avg_concurrent_sows", 0.50).alias("median_avg"),
            F.percentile_approx("avg_concurrent_sows", 0.75).alias("p75_avg"),
            F.percentile_approx("avg_concurrent_sows", 0.90).alias("p90_avg"),
            F.avg("avg_concurrent_sows").alias("mean_avg"),
            F.stddev_samp("avg_concurrent_sows").alias("stddev_avg")
        )
        .filter(F.col("n_projects") >= min_rows)
    )

concurrency_by_sbu = (
    profile_concurrency(project_concurrency, ["cat_sbu"])
    .withColumn("level", F.lit("SBU"))
    .withColumn("field", F.lit("sow_concurrency"))
)

concurrency_by_sbu_size = (
    profile_concurrency(
        project_concurrency.filter(F.col("size_category").isNotNull()),
        ["cat_sbu", "size_category"]
    )
    .withColumn("level", F.lit("SBU_SIZE"))
    .withColumn("field", F.lit("sow_concurrency"))
)

print("Concurrency by SBU:")
concurrency_by_sbu.show(10, truncate=False)

# Task Duration & Timing Profiling
print("\nProfiling task durations and timing shapes...")

task_durations = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.col("dt_task_start").isNotNull())
    .filter(F.col("dt_task_end").isNotNull())
    .withColumn("task_duration_days", F.datediff("dt_task_end", "dt_task_start"))
    .filter(F.col("task_duration_days") >= 0)
)

avg_duration_by_sow = (
    task_durations.groupBy("cat_sow")
    .agg(
        F.count("*").alias("n_tasks"),
        F.percentile_approx("task_duration_days", 0.10).alias("p10"),
        F.percentile_approx("task_duration_days", 0.25).alias("p25"),
        F.percentile_approx("task_duration_days", 0.50).alias("median"),
        F.percentile_approx("task_duration_days", 0.75).alias("p75"),
        F.percentile_approx("task_duration_days", 0.90).alias("p90"),
        F.avg("task_duration_days").alias("mean")
    )
    .filter(F.col("n_tasks") >= 100)
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("avg_task_duration"))
)

# Task timing shape within SOW
task_timing_within_sow = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.col("dt_task_start").isNotNull())
    .join(sow_timing, ["ids_project_number", "cat_sow"], "inner")
    .filter(F.col("sow_duration_days") > 0)
    .withColumn("task_start_offset_days", F.datediff("dt_task_start", "sow_start"))
    .withColumn("normalized_task_offset",
        F.col("task_start_offset_days") / F.col("sow_duration_days"))
    .filter(F.col("normalized_task_offset").between(0, 1))
)

task_timing_shape = (
    task_timing_within_sow.groupBy("cat_sow")
    .agg(
        F.count("*").alias("n_observations"),
        F.percentile_approx("normalized_task_offset", 0.10).alias("p10"),
        F.percentile_approx("normalized_task_offset", 0.25).alias("p25"),
        F.percentile_approx("normalized_task_offset", 0.50).alias("median"),
        F.percentile_approx("normalized_task_offset", 0.75).alias("p75"),
        F.percentile_approx("normalized_task_offset", 0.90).alias("p90"),
        F.avg("normalized_task_offset").alias("mean"),
        F.stddev_samp("normalized_task_offset").alias("stddev")
    )
    .filter(F.col("n_observations") >= 100)
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("task_start_timing_shape"))
)

# Task duration by position (early/mid/late)
task_duration_by_position = (
    task_timing_within_sow
    .filter(F.col("dt_task_end").isNotNull())
    .withColumn("task_duration_days", F.datediff("dt_task_end", "dt_task_start"))
    .filter(F.col("task_duration_days") >= 0)
    .withColumn("position_bucket",
        F.when(F.col("normalized_task_offset") < 0.33, "early")
         .when(F.col("normalized_task_offset") < 0.67, "mid")
         .otherwise("late"))
    .groupBy("cat_sow", "position_bucket")
    .agg(
        F.count("*").alias("n_tasks"),
        F.percentile_approx("task_duration_days", 0.10).alias("p10"),
        F.percentile_approx("task_duration_days", 0.25).alias("p25"),
        F.percentile_approx("task_duration_days", 0.50).alias("median"),
        F.percentile_approx("task_duration_days", 0.75).alias("p75"),
        F.percentile_approx("task_duration_days", 0.90).alias("p90"),
        F.avg("task_duration_days").alias("mean"),
        F.min("task_duration_days").alias("min"),
        F.max("task_duration_days").alias("max")
    )
    .filter(F.col("n_tasks") >= 50)
    .withColumn("level", F.lit("SOW_TYPE"))
)

# Completion & Early Start Profiling
print("\nProfiling completion and early start patterns...")

completion_stats = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.col("amt_duration_target").isNotNull())
    .withColumn("completion_status",
        F.when(F.col("amt_duration_remaining") == 0, "completed")
         .when(F.col("amt_duration_remaining") == F.col("amt_duration_target"), "not_started")
         .when((F.col("amt_duration_remaining") > 0)
               & (F.col("amt_duration_remaining") < F.col("amt_duration_target")), "active")
         .otherwise("unknown"))
)

completion_by_sow = (
    completion_stats.groupBy("cat_sow", "completion_status")
    .agg(F.count("*").alias("n_tasks"))
    .withColumn("pct",
        F.round(F.col("n_tasks") * 100.0
                / F.sum("n_tasks").over(W.partitionBy("cat_sow")), 2))
)

completion_profile = (
    completion_by_sow.groupBy("cat_sow")
    .pivot("completion_status", ["completed", "active", "not_started"])
    .agg(F.first("pct"))
    .withColumnRenamed("completed", "pct_completed")
    .withColumnRenamed("active", "pct_active")
    .withColumnRenamed("not_started", "pct_not_started")
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("completion_distribution"))
)

early_start_by_sow = (
    df.filter(F.col("cat_sow").isNotNull())
    .filter(F.col("cat_task_start_source").isNotNull())
    .groupBy("cat_sow")
    .agg(
        F.count("*").alias("n_tasks"),
        F.sum(F.when(F.col("cat_task_start_source") == "early start", 1)
              .otherwise(0)).alias("n_early_start")
    )
    .withColumn("pct_early_start",
        F.round(F.col("n_early_start") * 100.0 / F.col("n_tasks"), 2))
    .filter(F.col("n_tasks") >= 100)
    .withColumn("level", F.lit("SOW_TYPE"))
    .withColumn("field", F.lit("pct_early_start"))
)

# Completion Progression Matrix
# At X% project progress, what % is each SOW position complete?
# Profiled at SBU level with global fallback
print("\nV2: Profiling completion progression matrix...")

# Use ALL task snapshots (not just active) to capture version history
task_all_snapshots = (
    spark.read.table(SOURCE_TASK)
    .select(
        "ids_project_number", "cat_task_code", "cat_sow",
        "dt_task_start", "dt_task_end", "dt_effective_from",
        "amt_duration_remaining", "amt_duration_target"
    )
    .filter(F.col("cat_sow").isNotNull())
    .filter(F.trim(F.col("cat_sow")) != "")
    .join(valid_projects, "ids_project_number", "inner")
)

# Aggregate to SOW × snapshot level
sow_snapshot_agg = (
    task_all_snapshots
    .groupBy("ids_project_number", "cat_sow", "dt_effective_from")
    .agg(
        F.count("*").alias("n_tasks"),
        F.sum(F.when(F.col("amt_duration_remaining") == 0, 1)
              .otherwise(0)).alias("n_completed"),
        F.min("dt_task_start").alias("sow_start"),
        F.max("dt_task_end").alias("sow_finish")
    )
    .withColumn("pct_complete",
        F.round(F.col("n_completed") * 100.0
                / F.greatest(F.col("n_tasks"), F.lit(1)), 2))
)

# Join with project metadata for normalization
sow_snapshot_with_project = (
    sow_snapshot_agg
    .join(project_df.select("ids_project_number", "cat_sbu",
                            "dt_construction_start", "dt_construction_end"),
          "ids_project_number", "inner")
    .withColumn("proj_duration",
        F.datediff("dt_construction_end", "dt_construction_start"))
    .filter(F.col("proj_duration") > 30)
    .withColumn("snapshot_timing",
        F.round(F.datediff(F.to_date("dt_effective_from"), "dt_construction_start")
                / F.col("proj_duration").cast("double"), 3))
    .withColumn("sow_position",
        F.round(F.datediff(F.to_date("sow_start"), "dt_construction_start")
                / F.col("proj_duration").cast("double"), 3))
    .filter(F.col("snapshot_timing").between(0, 1.2))
    .filter(F.col("sow_position").between(0, 1.2))
    .withColumn("project_progress_bucket",
        F.when(F.col("snapshot_timing") < 0.2, "0-20")
         .when(F.col("snapshot_timing") < 0.4, "20-40")
         .when(F.col("snapshot_timing") < 0.6, "40-60")
         .when(F.col("snapshot_timing") < 0.8, "60-80")
         .otherwise("80-100"))
    .withColumn("sow_position_bucket",
        F.when(F.col("sow_position") < 0.2, "early")
         .when(F.col("sow_position") < 0.4, "early_mid")
         .when(F.col("sow_position") < 0.6, "mid")
         .when(F.col("sow_position") < 0.8, "mid_late")
         .otherwise("late"))
)

# Profile by SBU (finer granularity)
completion_by_sbu = (
    sow_snapshot_with_project
    .groupBy("cat_sbu", "project_progress_bucket", "sow_position_bucket")
    .agg(
        F.count("*").alias("n_observations"),
        F.round(F.avg("pct_complete"), 2).alias("avg_pct_complete"),
        F.percentile_approx("pct_complete", 0.25).alias("p25_complete"),
        F.percentile_approx("pct_complete", 0.50).alias("median_complete"),
        F.percentile_approx("pct_complete", 0.75).alias("p75_complete")
    )
    .filter(F.col("n_observations") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU"))
)

# Profile globally (fallback)
completion_global = (
    sow_snapshot_with_project
    .groupBy("project_progress_bucket", "sow_position_bucket")
    .agg(
        F.count("*").alias("n_observations"),
        F.round(F.avg("pct_complete"), 2).alias("avg_pct_complete"),
        F.percentile_approx("pct_complete", 0.25).alias("p25_complete"),
        F.percentile_approx("pct_complete", 0.50).alias("median_complete"),
        F.percentile_approx("pct_complete", 0.75).alias("p75_complete")
    )
    .filter(F.col("n_observations") >= 10)
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("cat_sbu", F.lit(None).cast("string"))
)

completion_progression = completion_by_sbu.unionByName(completion_global, allowMissingColumns=True)

print("Completion progression matrix (SBU + Global):")
completion_progression.orderBy("level", "cat_sbu", "project_progress_bucket", "sow_position_bucket").show(40, truncate=False)

(completion_progression
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_COMPLETION_PROGRESSION))

print(f"Saved completion progression to: {OUT_COMPLETION_PROGRESSION}")

# Task Count Change Ratios
# How much do task counts change between consecutive snapshots?
# Profiled at SBU level with global fallback
print("\nV2: Profiling task count changes between snapshots...")

w_snap = W.partitionBy("ids_project_number", "cat_sow").orderBy("dt_effective_from")

sow_with_next = (
    sow_snapshot_agg
    .join(project_df.select("ids_project_number", "cat_sbu"), "ids_project_number", "inner")
    .withColumn("n_tasks_next", F.lead("n_tasks").over(w_snap))
    .filter(F.col("n_tasks_next").isNotNull())
    .filter(F.col("n_tasks") >= 5)
    .withColumn("task_count_ratio",
        F.round(F.col("n_tasks_next") / F.col("n_tasks").cast("double"), 3))
    # Cap extreme outliers — ratios beyond 0.2x-5x are likely data artifacts
    .filter(F.col("task_count_ratio").between(0.2, 5.0))
)

# By SBU
task_count_change_by_sbu = (
    sow_with_next.groupBy("cat_sbu")
    .agg(
        F.count("*").alias("n_transitions"),
        F.round(F.avg("task_count_ratio"), 3).alias("avg_ratio"),
        F.percentile_approx("task_count_ratio", 0.10).alias("p10_ratio"),
        F.percentile_approx("task_count_ratio", 0.25).alias("p25_ratio"),
        F.percentile_approx("task_count_ratio", 0.50).alias("median_ratio"),
        F.percentile_approx("task_count_ratio", 0.75).alias("p75_ratio"),
        F.percentile_approx("task_count_ratio", 0.90).alias("p90_ratio")
    )
    .filter(F.col("n_transitions") >= MIN_GROUP_ROWS)
    .withColumn("level", F.lit("SBU"))
    .withColumn("field", F.lit("task_count_change_ratio"))
)

# Global fallback
task_count_change_global = (
    sow_with_next.agg(
        F.count("*").alias("n_transitions"),
        F.round(F.avg("task_count_ratio"), 3).alias("avg_ratio"),
        F.percentile_approx("task_count_ratio", 0.10).alias("p10_ratio"),
        F.percentile_approx("task_count_ratio", 0.25).alias("p25_ratio"),
        F.percentile_approx("task_count_ratio", 0.50).alias("median_ratio"),
        F.percentile_approx("task_count_ratio", 0.75).alias("p75_ratio"),
        F.percentile_approx("task_count_ratio", 0.90).alias("p90_ratio")
    )
    .withColumn("level", F.lit("GLOBAL"))
    .withColumn("field", F.lit("task_count_change_ratio"))
    .withColumn("cat_sbu", F.lit(None).cast("string"))
)

task_count_change_stats = task_count_change_by_sbu.unionByName(
    task_count_change_global, allowMissingColumns=True)

print("Task count change ratios (SBU + Global):")
task_count_change_stats.show(truncate=False)

# Date Shift Patterns Between Snapshots
# How do SOW dates shift by completion status × position?
print("\nV2: Profiling date shift patterns between snapshots...")

sow_with_shift = (
    sow_snapshot_agg.alias("s1")
    .join(
        sow_snapshot_agg.alias("s2"),
        (F.col("s1.ids_project_number") == F.col("s2.ids_project_number"))
        & (F.col("s1.cat_sow") == F.col("s2.cat_sow"))
        & (F.col("s2.dt_effective_from") > F.col("s1.dt_effective_from")),
        "inner"
    )
    # Only immediate next snapshot (no intermediate)
    .join(
        sow_snapshot_agg.alias("s3"),
        (F.col("s3.ids_project_number") == F.col("s1.ids_project_number"))
        & (F.col("s3.cat_sow") == F.col("s1.cat_sow"))
        & (F.col("s3.dt_effective_from") > F.col("s1.dt_effective_from"))
        & (F.col("s3.dt_effective_from") < F.col("s2.dt_effective_from")),
        "left_anti"
    )
    .select(
        F.col("s1.ids_project_number"),
        F.col("s1.cat_sow"),
        F.col("s1.pct_complete").alias("pct_complete_v1"),
        F.datediff("s2.sow_finish", "s1.sow_finish").alias("finish_shift_days"),
        F.datediff("s2.sow_start", "s1.sow_start").alias("start_shift_days"),
        (F.datediff("s2.sow_finish", "s2.sow_start")
         - F.datediff("s1.sow_finish", "s1.sow_start")).alias("duration_change_days"),
        F.col("s1.sow_start").alias("sow_start_v1")
    )
    .join(project_df.select("ids_project_number", "dt_construction_start", "dt_construction_end"),
          "ids_project_number", "inner")
    .withColumn("proj_duration",
        F.datediff("dt_construction_end", "dt_construction_start"))
    .filter(F.col("proj_duration") > 30)
    .withColumn("sow_position",
        F.round(F.datediff(F.to_date("sow_start_v1"), "dt_construction_start")
                / F.col("proj_duration").cast("double"), 3))
)

date_shift_profiled = (
    sow_with_shift
    .withColumn("completion_bucket",
        F.when(F.col("pct_complete_v1") >= 95, "completed")
         .when(F.col("pct_complete_v1") >= 50, "mostly_done")
         .when(F.col("pct_complete_v1") >= 10, "in_progress")
         .otherwise("not_started"))
    .withColumn("position_bucket",
        F.when(F.col("sow_position") < 0.25, "early")
         .when(F.col("sow_position") < 0.50, "early_mid")
         .when(F.col("sow_position") < 0.75, "mid_late")
         .otherwise("late"))
    .groupBy("completion_bucket", "position_bucket")
    .agg(
        F.count("*").alias("n_observations"),
        F.round(F.avg("finish_shift_days"), 1).alias("avg_finish_shift"),
        F.percentile_approx("finish_shift_days", 0.25).alias("p25_finish_shift"),
        F.percentile_approx("finish_shift_days", 0.50).alias("median_finish_shift"),
        F.percentile_approx("finish_shift_days", 0.75).alias("p75_finish_shift"),
        F.round(F.avg("start_shift_days"), 1).alias("avg_start_shift"),
        F.percentile_approx("start_shift_days", 0.50).alias("median_start_shift"),
        F.round(F.avg("duration_change_days"), 1).alias("avg_duration_change"),
        F.percentile_approx("duration_change_days", 0.50).alias("median_duration_change")
    )
    .filter(F.col("n_observations") >= 10)
)

print("Date shift patterns:")
date_shift_profiled.orderBy("completion_bucket", "position_bucket").show(20, truncate=False)

(date_shift_profiled
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_DATE_SHIFT_PROFILES))

print(f"Saved date shift profiles to: {OUT_DATE_SHIFT_PROFILES}")

# Consolidate Profile Table
print("\nConsolidating profile table...")

# Standard column set for profile table
PROFILE_COLS = [
    "level", "cat_sbu", "cat_main_asset_class", "size_category",
    "contract_bucket", "cat_sow", "field", "n_projects", "pct_missing",
    "min", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95",
    "p99", "max", "mean", "stddev"
]

def format_to_profile_schema(df, **overrides):
    """Add missing columns as NULL, rename as needed, select standard columns."""
    for col in PROFILE_COLS:
        if col not in df.columns:
            df = df.withColumn(col, F.lit(None).cast("double" if col != "level" and col != "field"
                and col != "cat_sbu" and col != "cat_main_asset_class"
                and col != "size_category" and col != "contract_bucket"
                and col != "cat_sow" else "string"))
    # Apply overrides (e.g., rename p50 → median)
    for old_name, new_name in overrides.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    # Ensure n_projects exists
    if "n_projects" not in df.columns and "n_observations" in df.columns:
        df = df.withColumnRenamed("n_observations", "n_projects")
    if "n_projects" not in df.columns and "n_tasks" in df.columns:
        df = df.withColumnRenamed("n_tasks", "n_projects")
    return df.select(*[c for c in PROFILE_COLS if c in df.columns])

# Format each profile component
sow_count_all = (
    format_to_profile_schema(sow_count_by_sbu, p50="median")
    .unionByName(format_to_profile_schema(sow_count_by_sbu_size, p50="median"), allowMissingColumns=True)
    .unionByName(format_to_profile_schema(sow_count_by_sbu_contract, p50="median"), allowMissingColumns=True)
)

sequence_all = (
    format_to_profile_schema(sequence_by_sow_sbu)
    .unionByName(format_to_profile_schema(sequence_global), allowMissingColumns=True)
)

task_count_all = (
    format_to_profile_schema(task_count_by_sow)
    .unionByName(format_to_profile_schema(task_count_by_sow_sbu), allowMissingColumns=True)
)

duration_all = (
    format_to_profile_schema(duration_by_sow)
    .unionByName(format_to_profile_schema(duration_by_sow_sbu), allowMissingColumns=True)
)

timing_shape_formatted = format_to_profile_schema(task_timing_shape)
avg_duration_formatted = format_to_profile_schema(avg_duration_by_sow)

# Concurrency: split into max and avg field rows
concurrency_max_rows = (
    concurrency_by_sbu
    .withColumn("field_out", F.lit("max_concurrent_sows"))
    .select(
        "level", "cat_sbu", "n_projects",
        F.col("field_out").alias("field"),
        F.col("p10_max").alias("p10"), F.col("p25_max").alias("p25"),
        F.col("median_max").alias("median"), F.col("p75_max").alias("p75"),
        F.col("p90_max").alias("p90"), F.col("mean_max").alias("mean"),
        F.col("stddev_max").alias("stddev")
    )
)
concurrency_avg_rows = (
    concurrency_by_sbu
    .withColumn("field_out", F.lit("avg_concurrent_sows"))
    .select(
        "level", "cat_sbu", "n_projects",
        F.col("field_out").alias("field"),
        F.col("p10_avg").alias("p10"), F.col("p25_avg").alias("p25"),
        F.col("median_avg").alias("median"), F.col("p75_avg").alias("p75"),
        F.col("p90_avg").alias("p90"), F.col("mean_avg").alias("mean"),
        F.col("stddev_avg").alias("stddev")
    )
)
concurrency_max_size_rows = (
    concurrency_by_sbu_size
    .withColumn("field_out", F.lit("max_concurrent_sows"))
    .select(
        "level", "cat_sbu", "size_category", "n_projects",
        F.col("field_out").alias("field"),
        F.col("p10_max").alias("p10"), F.col("p25_max").alias("p25"),
        F.col("median_max").alias("median"), F.col("p75_max").alias("p75"),
        F.col("p90_max").alias("p90"), F.col("mean_max").alias("mean"),
        F.col("stddev_max").alias("stddev")
    )
)
concurrency_avg_size_rows = (
    concurrency_by_sbu_size
    .withColumn("field_out", F.lit("avg_concurrent_sows"))
    .select(
        "level", "cat_sbu", "size_category", "n_projects",
        F.col("field_out").alias("field"),
        F.col("p10_avg").alias("p10"), F.col("p25_avg").alias("p25"),
        F.col("median_avg").alias("median"), F.col("p75_avg").alias("p75"),
        F.col("p90_avg").alias("p90"), F.col("mean_avg").alias("mean"),
        F.col("stddev_avg").alias("stddev")
    )
)

concurrency_all = (
    format_to_profile_schema(concurrency_max_rows)
    .unionByName(format_to_profile_schema(concurrency_avg_rows), allowMissingColumns=True)
    .unionByName(format_to_profile_schema(concurrency_max_size_rows), allowMissingColumns=True)
    .unionByName(format_to_profile_schema(concurrency_avg_size_rows), allowMissingColumns=True)
)

# Combine everything into profile table
profile_all = (
    sow_count_all
    .unionByName(sequence_all, allowMissingColumns=True)
    .unionByName(task_count_all, allowMissingColumns=True)
    .unionByName(duration_all, allowMissingColumns=True)
    .unionByName(timing_shape_formatted, allowMissingColumns=True)
    .unionByName(avg_duration_formatted, allowMissingColumns=True)
    .unionByName(concurrency_all, allowMissingColumns=True)
)

(profile_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_PROFILE))

print(f"Saved profile to: {OUT_PROFILE} ({profile_all.count()} rows)")

# Build Constraints Table
print("\nBuilding constraints table...")

def build_constraint_json(df, key_fields, value_fields):
    """Build bounds_json from specified columns."""
    map_args = []
    for f in value_fields:
        if f in df.columns:
            map_args.extend([F.lit(f), F.col(f)])
    return df.withColumn("bounds_json", F.to_json(F.create_map(*map_args)))

CONSTRAINT_COLS = ["level", "cat_sbu", "cat_main_asset_class", "size_category",
                   "contract_bucket", "cat_sow", "field", "n", "bounds_json"]

def format_to_constraint_schema(df):
    for col in CONSTRAINT_COLS:
        if col not in df.columns:
            if col == "n":
                if "n_projects" in df.columns:
                    df = df.withColumnRenamed("n_projects", "n")
                elif "n_observations" in df.columns:
                    df = df.withColumnRenamed("n_observations", "n")
                elif "n_tasks" in df.columns:
                    df = df.withColumnRenamed("n_tasks", "n")
                elif "n_transitions" in df.columns:
                    df = df.withColumnRenamed("n_transitions", "n")
                else:
                    df = df.withColumn("n", F.lit(None).cast("long"))
            else:
                df = df.withColumn(col, F.lit(None).cast("string"))
    return df.select(*[c for c in CONSTRAINT_COLS if c in df.columns])

# SOW count constraints
sow_count_c = build_constraint_json(
    sow_count_all, [],
    ["p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "min", "max", "mean"]
)
sow_count_c = format_to_constraint_schema(sow_count_c)

# Sequence constraints
sequence_c = build_constraint_json(
    sequence_all, [],
    ["p10", "p25", "median", "p75", "p90", "mean", "stddev", "min", "max"]
)
sequence_c = format_to_constraint_schema(sequence_c)

# Task count constraints
task_count_c = build_constraint_json(
    task_count_all, [],
    ["p5", "p10", "p25", "median", "p75", "p90", "p95", "min", "max", "mean"]
)
task_count_c = format_to_constraint_schema(task_count_c)

# Duration constraints
duration_c = build_constraint_json(
    duration_all, [],
    ["p5", "p10", "p25", "median", "p75", "p90", "p95", "min", "max", "mean"]
)
duration_c = format_to_constraint_schema(duration_c)

# Timing shape constraints
timing_c = build_constraint_json(
    timing_shape_formatted, [],
    ["p10", "p25", "median", "p75", "p90", "mean", "stddev"]
)
timing_c = format_to_constraint_schema(timing_c)

# Duration by position constraints
dur_pos_c = (
    task_duration_by_position
    .withColumn("field", F.concat(F.lit("task_duration_"), F.col("position_bucket")))
)
dur_pos_c = build_constraint_json(
    dur_pos_c, [],
    ["p10", "p25", "median", "p75", "p90", "mean", "min", "max"]
)
dur_pos_c = format_to_constraint_schema(dur_pos_c)

# Concurrency constraints
concurrency_c = build_constraint_json(
    concurrency_all, [],
    ["p10", "p25", "median", "p75", "p90", "mean", "stddev"]
)
concurrency_c = format_to_constraint_schema(concurrency_c)

# V2: Task count change ratio constraints (SBU + Global)
task_change_c_all = None
for row_df in [task_count_change_by_sbu, task_count_change_global]:
    tc = (
        row_df
        .withColumn("bounds_json",
            F.to_json(F.create_map(
                F.lit("p10"), F.col("p10_ratio"),
                F.lit("p25"), F.col("p25_ratio"),
                F.lit("median"), F.col("median_ratio"),
                F.lit("p75"), F.col("p75_ratio"),
                F.lit("p90"), F.col("p90_ratio"),
                F.lit("mean"), F.col("avg_ratio")
            )))
    )
    tc = format_to_constraint_schema(tc)
    if task_change_c_all is None:
        task_change_c_all = tc
    else:
        task_change_c_all = task_change_c_all.unionByName(tc, allowMissingColumns=True)

# Combine all constraints
constraints_all = (
    sow_count_c
    .unionByName(sequence_c, allowMissingColumns=True)
    .unionByName(task_count_c, allowMissingColumns=True)
    .unionByName(duration_c, allowMissingColumns=True)
    .unionByName(timing_c, allowMissingColumns=True)
    .unionByName(dur_pos_c, allowMissingColumns=True)
    .unionByName(concurrency_c, allowMissingColumns=True)
    .unionByName(task_change_c_all, allowMissingColumns=True)
)

(constraints_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_CONSTRAINTS))

print(f"Saved constraints to: {OUT_CONSTRAINTS} ({constraints_all.count()} rows)")

# Summary
print("\n=== PROFILING SUMMARY ===")
print(f"Projects analyzed: {valid_projects_count}")
print(f"Active tasks analyzed: {df.count()}")
print(f"Tasks with SOW: {df.filter(F.col('cat_sow').isNotNull()).count()}")
print(f"Distinct SOW types: {df.select('cat_sow').filter(F.col('cat_sow').isNotNull()).distinct().count()}")
print(f"\nTables created/updated:")
print(f"  1. {OUT_PROFILE} ({profile_all.count()} rows)")
print(f"  2. {OUT_CONSTRAINTS} ({constraints_all.count()} rows)")
print(f"  3. {OUT_SOW_FREQUENCY} ({sow_frequency_by_sbu.count()} rows)")
print(f"  4. {OUT_SOW_OVERLAPS} ({overlap_by_pair.count()} pair + 1 general)")
print(f"  5. {OUT_COMPLETION_PROGRESSION} (V2: completion CDF matrix)")
print(f"  6. {OUT_DATE_SHIFT_PROFILES} (V2: date shift patterns)")
print("\n=== PROFILING COMPLETE ===")
