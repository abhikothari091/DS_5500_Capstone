# One-notebook profiling with a single consolidated output table
from pyspark.sql import functions as F

SOURCE = "analysts.self_managed.project_meta_data_master"
OUT_PROFILE = "analysts.self_managed.project_meta_profile"  # single, consolidated table
OUT_CONSTRAINTS = "analysts.self_managed.project_meta_profile_constraints"  # optional JSON scaffold

# Minimum rows required for SBU×Asset combos (avoid noisy tiny groups)
MIN_GROUP_ROWS = 5

df = spark.read.table(SOURCE).select(
    "ids_project_number",
    "cat_sbu",
    "cat_region",
    "cat_main_asset_class",
    "dt_construction_start",
    "dt_construction_end",
    "n_floors_above_grade",
    "gross_area",
    "amt_contract",
    "amt_cost",
    "is_n_floors_missing",
    "is_gross_area_missing",
    "is_dt_construction_end_missing",
    "is_cat_main_asset_class_missing"
)

df = (
    df
    .withColumn("dt_construction_start", F.to_date("dt_construction_start"))
    .withColumn("dt_construction_end",   F.to_date("dt_construction_end"))
)

# ensure end >= start when both present
df = df.withColumn(
    "dt_construction_end",
    F.when(
        (F.col("dt_construction_end").isNotNull()) &
        (F.col("dt_construction_start").isNotNull()) &
        (F.col("dt_construction_end") < F.col("dt_construction_start")),
        F.col("dt_construction_start")
    ).otherwise(F.col("dt_construction_end"))
)

df = (
    df
    .withColumn("duration_days", F.datediff("dt_construction_end", "dt_construction_start"))
    .withColumn("n_floors_above_grade", F.coalesce(F.col("n_floors_above_grade").cast("int"), F.lit(1)))
    .withColumn("n_floors_above_grade", F.when(F.col("n_floors_above_grade") < 1, F.lit(1)).otherwise(F.col("n_floors_above_grade")))
    .withColumn("gross_area",   F.col("gross_area").cast("double"))
    .withColumn("gross_area",   F.when(F.col("gross_area") < 1, F.lit(1.0)).otherwise(F.col("gross_area")))
    .withColumn("amt_contract", F.when(F.col("amt_contract").cast("double") < 0, F.lit(0.0)).otherwise(F.col("amt_contract").cast("double")))
    .withColumn("amt_cost",     F.when(F.col("amt_cost").cast("double") < 0, F.lit(0.0)).otherwise(F.col("amt_cost").cast("double")))
    .withColumn("cost_contract_ratio", F.when(F.col("amt_contract") > 0, F.col("amt_cost")/F.col("amt_contract")))
    .withColumn("contract_per_sqft",   F.when(F.col("gross_area")   > 0, F.col("amt_contract")/F.col("gross_area")))
    .withColumn("cost_per_sqft",       F.when(F.col("gross_area")   > 0, F.col("amt_cost")/F.col("gross_area")))
)

NUM_FIELDS = [
    "duration_days",
    "n_floors_above_grade",
    "gross_area",
    "amt_contract",
    "amt_cost",
    "cost_contract_ratio",
    "contract_per_sqft",
    "cost_per_sqft"
]

def build_profile(df_in, level_label, group_cols):
    # totals for missingness
    totals = df_in.groupBy(*group_cols).agg(F.count(F.lit(1)).alias("n_total"))
    pieces = []
    for field in NUM_FIELDS:
        x = df_in.select(*group_cols, F.col(field).alias("x"))
        # stats on non-null
        stats = (
            x.where(F.col("x").isNotNull())
             .groupBy(*group_cols)
             .agg(
                 F.count("x").alias("n_nonnull"),
                 F.min("x").alias("min"),
                 F.percentile_approx("x", 0.05, 10000).alias("p5"),
                 F.percentile_approx("x", 0.25, 10000).alias("p25"),
                 F.percentile_approx("x", 0.50, 10000).alias("median"),
                 F.percentile_approx("x", 0.75, 10000).alias("p75"),
                 F.percentile_approx("x", 0.95, 10000).alias("p95"),
                 F.max("x").alias("max"),
                 F.avg("x").alias("mean"),
                 F.stddev_samp("x").alias("stddev")
             )
        )
        one = (
            totals.join(stats, on=group_cols, how="left")
                  .withColumn("n_nonnull", F.coalesce(F.col("n_nonnull"), F.lit(0)))
                  .withColumn("pct_missing", F.round((F.col("n_total") - F.col("n_nonnull"))/F.col("n_total")*100.0, 2))
                  .withColumn("field", F.lit(field))
        )
        pieces.append(one)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.unionByName(p, allowMissingColumns=True)
    out = (out
        .withColumn("level", F.lit(level_label))
        .withColumn("min",     F.round(F.col("min"), 2))
        .withColumn("p5",      F.round(F.col("p5"), 2))
        .withColumn("p25",     F.round(F.col("p25"), 2))
        .withColumn("median",  F.round(F.col("median"), 2))
        .withColumn("p75",     F.round(F.col("p75"), 2))
        .withColumn("p95",     F.round(F.col("p95"), 2))
        .withColumn("max",     F.round(F.col("max"), 2))
        .withColumn("mean",    F.round(F.col("mean"), 2))
        .withColumn("stddev",  F.round(F.col("stddev"), 2))
    )
    return out

# SBU level
prof_sbu = build_profile(df, "SBU", ["cat_sbu"])

# SBU × Asset (filtered to groups with enough rows)
counts_sa = df.groupBy("cat_sbu","cat_main_asset_class").agg(F.count(F.lit(1)).alias("n_group"))
valid_sa = counts_sa.where(F.col("n_group") >= F.lit(MIN_GROUP_ROWS)) \
                    .select("cat_sbu","cat_main_asset_class")
df_sa = df.join(valid_sa, on=["cat_sbu","cat_main_asset_class"], how="inner")
prof_sa = build_profile(df_sa, "SBU_ASSET", ["cat_sbu","cat_main_asset_class"])

# Consolidate to ONE table
profile_all = (
    prof_sbu.select(
        F.col("level"),
        F.col("cat_sbu"),
        F.lit(None).cast("string").alias("cat_main_asset_class"),
        F.col("field"), F.col("n_total"), F.col("n_nonnull"), F.col("pct_missing"),
        F.col("min"), F.col("p5"), F.col("p25"), F.col("median"), F.col("p75"), F.col("p95"),
        F.col("max"), F.col("mean"), F.col("stddev")
    )
    .unionByName(
        prof_sa.select(
            F.col("level"),
            F.col("cat_sbu"),
            F.col("cat_main_asset_class"),
            F.col("field"), F.col("n_total"), F.col("n_nonnull"), F.col("pct_missing"),
            F.col("min"), F.col("p5"), F.col("p25"), F.col("median"), F.col("p75"), F.col("p95"),
            F.col("max"), F.col("mean"), F.col("stddev")
        ),
        allowMissingColumns=True
    )
)

(profile_all
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema","true")
 .saveAsTable(OUT_PROFILE))

print("Saved consolidated profile to:", OUT_PROFILE)

# This produces one row per group+field with a JSON blob of bounds we can ingest later.
bounds = []
quantiles = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
for level, group_cols in [("SBU", ["cat_sbu"]), ("SBU_ASSET", ["cat_sbu","cat_main_asset_class"])]:
    base = df if level=="SBU" else df_sa
    for fld in NUM_FIELDS:
        qexprs = [F.percentile_approx(F.col(fld), q, 10000).alias(f"p{int(q*100)}") for q in quantiles]
        qtab = (base.groupBy(*group_cols)
                     .agg(F.count(F.lit(1)).alias("n"),
                          *qexprs,
                          F.min(fld).alias("min"),
                          F.max(fld).alias("max")))
        qtab = qtab.withColumn("field", F.lit(fld)) \
                   .withColumn("level", F.lit(level))
        bounds.append(qtab)

if bounds:
    qall = bounds[0]
    for b in bounds[1:]:
        qall = qall.unionByName(b, allowMissingColumns=True)

    # build a compact JSON dict per row
    cols_keep = ["min","p1","p5","p10","p25","p50","p75","p90","p95","p99","max"]
    def json_map(cols):
        return F.to_json(F.create_map([F.lit(k) if i%2==0 else F.col(cols[i//2])
                                       for i,k in enumerate(sum([[c,c] for c in cols], []))]))

    qall = (qall
        .withColumn("p1",  F.col("p1").cast("double"))
        .withColumn("p5",  F.col("p5").cast("double"))
        .withColumn("p10", F.col("p10").cast("double"))
        .withColumn("p25", F.col("p25").cast("double"))
        .withColumn("p50", F.col("p50").cast("double"))
        .withColumn("p75", F.col("p75").cast("double"))
        .withColumn("p90", F.col("p90").cast("double"))
        .withColumn("p95", F.col("p95").cast("double"))
        .withColumn("p99", F.col("p99").cast("double"))
        .withColumn("bounds_json", json_map(cols_keep))
        .select("level","cat_sbu","cat_main_asset_class","field","n","bounds_json")
    )

    (qall.write.format("delta").mode("overwrite")
         .option("overwriteSchema","true")
         .saveAsTable(OUT_CONSTRAINTS))

    print("Saved constraints scaffold to:", OUT_CONSTRAINTS)
