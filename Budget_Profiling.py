import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import Window

SRC_BUDGET  = "platinum.models.budget"
SRC_PROJECT = "platinum.models.project"
OUT_PROFILE = "analysts.self_managed.budget_profile_v1"
OUT_CONSTR  = "analysts.self_managed.budget_constraints_v1"

CANONICAL_DIVISIONS = [
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","21","22","23","26","27","28",
    "31","32","33","95","98","99",
]
MANDATORY_DIVISIONS = {"98", "99", "01"}

CONTRACT_MIN  = 25_000_000
START_YEAR    = 2019
MIN_WEIGHT    = 0.05
MIN_PROJECTS  = 5

PCTS       = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
PCT_LABELS = ["p10", "p25", "p50", "p75", "p90", "p99"]

def collect_pcts(pdf, col):
    vals = pdf[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 3:
        return {k: None for k in PCT_LABELS}
    return dict(zip(PCT_LABELS, np.quantile(vals, PCTS).tolist()))

def base_row(level, sbu=None, contract_bucket=None, size_cat=None, asset=None):
    return {
        "level": level,
        "cat_sbu": sbu,
        "contract_bucket": contract_bucket,
        "size_category": size_cat,
        "cat_main_asset_class": asset,
    }

profile_rows = []
constr_rows  = []

# --- 0. Universe ---
proj = (
    spark.table(SRC_PROJECT)
    .filter(F.col("is_latest") == True)
    .filter(F.col("amt_contract") >= CONTRACT_MIN)
    .filter(F.year("dt_construction_start") >= START_YEAR)
    .filter(F.col("cat_sbu").isNotNull())
    .select("ids_project_number", "cat_sbu", "cat_main_asset_class",
            "amt_contract", "amt_area_gross",
            "dt_construction_start", "dt_construction_end")
)

n_proj = proj.count()
print(f"Profiling universe: {n_proj} projects")

bud_raw = (
    spark.table(SRC_BUDGET)
    .filter(F.col("is_best_months_budget") == True)
    .withColumn("div_prefix", F.trim(F.split(F.col("cat_budget_division"), " - ")[0]))
    .filter(F.col("div_prefix").isin(CANONICAL_DIVISIONS))
    .join(proj, on="ids_project_number", how="inner")
)

bud = (
    bud_raw
    .withColumn("contract_bucket",
        F.when(F.col("amt_contract") <  50e6,  "Medium")
         .when(F.col("amt_contract") < 100e6,  "Large")
         .when(F.col("amt_contract") < 200e6,  "XLarge")
         .otherwise("XXLarge"))
    .withColumn("size_category",
        F.when(F.col("amt_area_gross") <  50_000, "Small")
         .when(F.col("amt_area_gross") < 100_000, "Medium")
         .when(F.col("amt_area_gross") < 200_000, "Large")
         .otherwise("XLarge"))
    .withColumn("dt_budget_date", F.col("dt_budget").cast("date"))
    .withColumn("budget_month_num",
        F.round(
            F.months_between(F.col("dt_budget_date"), F.col("dt_construction_start")), 0
        ).cast("int"))
    .withColumn("project_duration_months",
        F.greatest(
            F.months_between(F.col("dt_construction_end"),
                             F.col("dt_construction_start")).cast("int"),
            F.lit(1)))
    .withColumn("lifecycle_pct_raw",
        F.col("budget_month_num") / F.col("project_duration_months"))
    .withColumn("lifecycle_pct",
        F.when(F.col("lifecycle_pct_raw") < 0, F.lit(0.0))
         .when(F.col("lifecycle_pct_raw") > 1, F.lit(1.0))
         .otherwise(F.col("lifecycle_pct_raw")))
    .withColumn("cat_budget_category",
        F.when(F.lower(F.col("cat_budget_category")).startswith("subcontract"), "subcontractor")
         .when(F.lower(F.col("cat_budget_category")).startswith("value engine"), "value engineering")
         .when(F.lower(F.col("cat_budget_category")).startswith("liberty equip"), "liberty equipment")
         .when(F.lower(F.col("cat_budget_category")).startswith("change order f"), "change order fee")
         .when(F.col("cat_budget_category").isNull(), "other")
         .otherwise(F.lower(F.trim(F.col("cat_budget_category")))))
    .cache()
)

# --- L1: Division presence probabilities ---
n_proj_by_sbu = proj.groupBy("cat_sbu").agg(
    F.countDistinct("ids_project_number").alias("n_total")
)

div_presence_sbu = (
    bud.select("ids_project_number", "cat_sbu", "div_prefix").distinct()
    .groupBy("cat_sbu", "div_prefix")
    .agg(F.countDistinct("ids_project_number").alias("n_with"))
    .join(n_proj_by_sbu, on="cat_sbu")
    .withColumn("presence_prob", F.col("n_with") / F.col("n_total"))
    .toPandas()
)

div_presence_global = (
    bud.select("ids_project_number", "div_prefix").distinct()
    .groupBy("div_prefix")
    .agg(F.countDistinct("ids_project_number").alias("n_with"))
    .withColumn("n_total", F.lit(n_proj))
    .withColumn("presence_prob", F.col("n_with") / F.col("n_total"))
    .toPandas()
)

for _, r in div_presence_sbu.iterrows():
    row = base_row("SBU", sbu=r["cat_sbu"])
    row.update({
        "field": "division_presence_prob",
        "cat_budget_division": r["div_prefix"],
        "str_value": str(round(r["presence_prob"], 4)),
        "is_mandatory": r["div_prefix"] in MANDATORY_DIVISIONS,
        "n_projects": int(r["n_with"]),
    })
    profile_rows.append(row)

for _, r in div_presence_global.iterrows():
    row = base_row("GLOBAL")
    row.update({
        "field": "division_presence_prob",
        "cat_budget_division": r["div_prefix"],
        "str_value": str(round(r["presence_prob"], 4)),
        "is_mandatory": r["div_prefix"] in MANDATORY_DIVISIONS,
        "n_projects": int(r["n_with"]),
    })
    profile_rows.append(row)

# --- L2: Cost code counts + full_cost_code lookup ---
code_counts = (
    bud.select("ids_project_number", "cat_sbu", "div_prefix", "cat_cost_code").distinct()
    .groupBy("ids_project_number", "cat_sbu", "div_prefix")
    .agg(F.count("cat_cost_code").alias("n_cost_codes"))
    .toPandas()
)

code_freq = (
    bud.select("ids_project_number", "div_prefix", "cat_cost_code", "cat_full_cost_code").distinct()
    .groupBy("div_prefix", "cat_cost_code", "cat_full_cost_code")
    .agg(F.countDistinct("ids_project_number").alias("n_projects"))
    .orderBy("div_prefix", F.desc("n_projects"))
    .toPandas()
)

for div, grp in code_counts.groupby("div_prefix"):
    for sbu, sg in grp.groupby("cat_sbu"):
        if len(sg) < 3:
            continue
        pcts = collect_pcts(sg, "n_cost_codes")
        row = base_row("SBU", sbu=sbu)
        row.update({"field": "n_cost_codes_per_division",
                    "cat_budget_division": div, "n_projects": len(sg), **pcts})
        constr_rows.append(row)
    pcts = collect_pcts(grp, "n_cost_codes")
    row = base_row("GLOBAL")
    row.update({"field": "n_cost_codes_per_division",
                "cat_budget_division": div, "n_projects": len(grp), **pcts})
    constr_rows.append(row)

for div, grp in code_freq.groupby("div_prefix"):
    lookup = grp[["cat_cost_code", "cat_full_cost_code", "n_projects"]].to_dict("records")
    row = base_row("GLOBAL")
    row.update({
        "field": "cost_code_frequency_table",
        "cat_budget_division": div,
        "json_blob": json.dumps(lookup),
        "n_projects": int(grp["n_projects"].max()),
    })
    constr_rows.append(row)

# --- L3: Budget category mix per division (cost-weighted, dual threshold collapse) ---
cat_mix = (
    bud.filter(F.col("amt_budget_original") > 0)
    .groupBy("div_prefix", "cat_budget_category")
    .agg(
        F.sum("amt_budget_original").alias("total_amt"),
        F.countDistinct("ids_project_number").alias("n_projects"),
    )
    .toPandas()
)

div_proj_counts = (
    bud.select("div_prefix", "ids_project_number").distinct()
    .groupBy("div_prefix")
    .agg(F.countDistinct("ids_project_number").alias("n_div_projects"))
    .toPandas()
    .set_index("div_prefix")["n_div_projects"]
    .to_dict()
)

for div, grp in cat_mix.groupby("div_prefix"):
    total_amt = grp["total_amt"].sum()
    weights   = {}
    other_amt = 0.0
    dropped   = []
    for _, r in grp.iterrows():
        w = r["total_amt"] / total_amt
        if w >= MIN_WEIGHT and r["n_projects"] >= MIN_PROJECTS:
            weights[r["cat_budget_category"]] = w
        else:
            other_amt += r["total_amt"]
            dropped.append(r["cat_budget_category"])
    if dropped:
        print(f"  [{div}] collapsed: {dropped}")
    # Redistribute collapsed weight proportionally across surviving named categories
    # rather than emitting an explicit 'other' bucket — keeps category values clean
    if other_amt > 0 and len(weights) > 0:
        surviving_total = sum(weights.values())
        for k in weights:
            weights[k] += other_amt / total_amt * (weights[k] / surviving_total)
    elif other_amt > 0:
        # No surviving named categories at all — fall back to single 'other' bucket
        weights["other"] = 1.0
    total_w = sum(weights.values())
    weights = {k: round(v / total_w, 4) for k, v in weights.items()}
    row = base_row("GLOBAL")
    row.update({
        "field": "category_weight_vector",
        "cat_budget_division": div,
        "json_blob": json.dumps(weights),
        "n_projects": div_proj_counts.get(div, 0),
    })
    profile_rows.append(row)

# --- L4: amt_budget_original / amt_contract ratios ---
# Aggregate raw amounts first, then divide once — avoids ratio summing across cost codes
amt_ratios = (
    bud.filter(F.col("is_current_snap") == True)
    .filter(F.col("amt_budget_original") > 0)
    .groupBy("ids_project_number", "cat_sbu", "cat_main_asset_class",
             "contract_bucket", "size_category", "div_prefix", "cat_budget_category")
    .agg(F.sum("amt_budget_original").alias("div_cat_amt"))
    .join(proj.select("ids_project_number", "amt_contract"),
          on="ids_project_number", how="inner")
    .withColumn("div_cat_ratio", F.col("div_cat_amt") / F.col("amt_contract"))
    .toPandas()
)

def emit_amount_row(pdf, level, div, cat, **kwargs):
    pcts = collect_pcts(pdf, "div_cat_ratio")
    if pcts["p50"] is None:
        return
    row = base_row(level, **kwargs)
    row.update({
        "field": "amt_budget_original_ratio",
        "cat_budget_division": div,
        "cat_budget_category": cat,
        "n_projects": len(pdf),
        **pcts,
    })
    profile_rows.append(row)

for (div, cat), grp in amt_ratios.groupby(["div_prefix", "cat_budget_category"]):
    if len(grp) < 5:
        continue
    for (sbu, asset), sg in grp.groupby(["cat_sbu", "cat_main_asset_class"]):
        if len(sg) >= 5:
            emit_amount_row(sg, "SBU_ASSET", div, cat, sbu=sbu, asset=asset)
    for (sbu, cb), sg in grp.groupby(["cat_sbu", "contract_bucket"]):
        if len(sg) >= 5:
            emit_amount_row(sg, "SBU_CONTRACT", div, cat, sbu=sbu, contract_bucket=cb)
    for sbu, sg in grp.groupby("cat_sbu"):
        if len(sg) >= 5:
            emit_amount_row(sg, "SBU", div, cat, sbu=sbu)
    emit_amount_row(grp, "GLOBAL", div, cat)

# Total budget-to-contract ratio at project level for envelope enforcement
proj_totals_pd = (
    bud.filter(F.col("is_current_snap") == True)
    .groupBy("ids_project_number", "cat_sbu", "contract_bucket",
             "size_category", "cat_main_asset_class", "amt_contract")
    .agg(F.sum("amt_budget_original").alias("total_budget_original"))
    .withColumn("budget_to_contract_ratio",
        F.col("total_budget_original") / F.col("amt_contract"))
    .toPandas()
)

for sbu, sg in proj_totals_pd.groupby("cat_sbu"):
    pcts = collect_pcts(sg, "budget_to_contract_ratio")
    row = base_row("SBU", sbu=sbu)
    row.update({"field": "total_budget_to_contract_ratio",
                "n_projects": len(sg), **pcts})
    profile_rows.append(row)

pcts = collect_pcts(proj_totals_pd, "budget_to_contract_ratio")
row = base_row("GLOBAL")
row.update({"field": "total_budget_to_contract_ratio",
            "n_projects": len(proj_totals_pd), **pcts})
profile_rows.append(row)

# --- L5: S-curve shapes per division ---
# total_div_budget computed before lifecycle filter so pre/post-TCO rows
# still contribute to the denominator
div_total_budget = (
    bud.filter(F.col("amt_budget_original") > 0)
    .groupBy("ids_project_number", "div_prefix")
    .agg(F.sum("amt_budget_original").alias("total_div_budget"))
)

scurve_binned = (
    bud.filter(F.col("amt_budget_original") > 0)
    .filter(F.col("lifecycle_pct").between(0, 1))
    .withColumn("decile_bin",
        (F.floor(F.col("lifecycle_pct") * 10) / 10).cast("double"))
    .join(div_total_budget, on=["ids_project_number", "div_prefix"], how="inner")
)

# Within each decile bin take the latest snapshot date, then sum amt_cost_jtd
# across all cost codes — gives true division-level cumulative spend at that point
latest_in_bin = (
    scurve_binned
    .groupBy("ids_project_number", "div_prefix", "decile_bin")
    .agg(F.max("dt_budget_date").alias("latest_date"))
)

scurve_base = (
    scurve_binned
    .join(latest_in_bin, on=["ids_project_number", "div_prefix", "decile_bin"], how="inner")
    .filter(F.col("dt_budget_date") == F.col("latest_date"))
    .groupBy("ids_project_number", "div_prefix", "decile_bin", "total_div_budget")
    .agg(F.sum("amt_cost_jtd").alias("cum_cost_at_decile"))
    .withColumn("raw_cume_share",
        F.col("cum_cost_at_decile") / F.col("total_div_budget"))
    .toPandas()
)

def monotonize(series):
    out, running_max = [], 0.0
    for v in series:
        v = max(0.0, min(1.0, v))
        running_max = max(running_max, v)
        out.append(running_max)
    return out

scurve_mono = []
for (proj_id, div), grp in scurve_base.groupby(["ids_project_number", "div_prefix"]):
    grp_sorted = grp.sort_values("decile_bin")
    mono_vals  = monotonize(grp_sorted["raw_cume_share"].tolist())
    for i, (_, r) in enumerate(grp_sorted.iterrows()):
        scurve_mono.append({
            "ids_project_number": proj_id,
            "div_prefix": div,
            "decile_bin": r["decile_bin"],
            "cume_share": mono_vals[i],
        })

scurve_mono_df = pd.DataFrame(scurve_mono)

for div, grp in scurve_mono_df.groupby("div_prefix"):
    shape = {}
    for decile, dg in grp.groupby("decile_bin"):
        vals = dg["cume_share"].dropna()
        if len(vals) < 3:
            continue
        shape[str(round(decile, 1))] = {
            "p10": round(float(np.quantile(vals, 0.10)), 4),
            "p50": round(float(np.quantile(vals, 0.50)), 4),
            "p90": round(float(np.quantile(vals, 0.90)), 4),
        }
    if not shape:
        continue
    row = base_row("GLOBAL")
    row.update({
        "field": "scurve_shape",
        "cat_budget_division": div,
        "json_blob": json.dumps(shape),
        "n_projects": grp["ids_project_number"].nunique(),
    })
    constr_rows.append(row)

# --- L6: Change order rate profiles (project-month grain) ---
# Aggregate to project × month × division first to get correct CO probability —
# avoids dilution from the many zero-CO rows at row grain
co_pm = (
    bud.groupBy("ids_project_number", "cat_sbu", "div_prefix", "dt_budget_date")
    .agg(
        F.max(F.when(F.col("amt_budget__in_pending")  > 0, 1).otherwise(0)).alias("has_co_in"),
        F.max(F.when(F.col("amt_budget__out_pending") < 0, 1).otherwise(0)).alias("has_co_out"),
        F.sum(F.col("amt_budget_approved_in")).alias("total_co_in"),
        F.sum(F.abs(F.col("amt_budget_approved_out"))).alias("total_co_out"),
    )
    .join(proj.select("ids_project_number", "amt_contract"),
          on="ids_project_number", how="inner")
    .withColumn("co_in_ratio",  F.col("total_co_in")  / F.col("amt_contract"))
    .withColumn("co_out_ratio", F.col("total_co_out") / F.col("amt_contract"))
    .toPandas()
)

def emit_co_row(pdf, level, div, **kwargs):
    if len(pdf) < 5:
        return
    prob_in  = round(pdf["has_co_in"].mean(),  4)
    prob_out = round(pdf["has_co_out"].mean(), 4)
    in_vals  = pdf.loc[pdf["has_co_in"]  == 1, "co_in_ratio"].replace(
                   [np.inf, -np.inf], np.nan).dropna()
    out_vals = pdf.loc[pdf["has_co_out"] == 1, "co_out_ratio"].replace(
                   [np.inf, -np.inf], np.nan).dropna()
    blob = {
        "prob_co_in":       prob_in,
        "prob_co_out":      prob_out,
        "p50_co_in_ratio":  round(float(np.quantile(in_vals,  0.50)), 6) if len(in_vals)  >= 3 else None,
        "p90_co_in_ratio":  round(float(np.quantile(in_vals,  0.90)), 6) if len(in_vals)  >= 3 else None,
        "p50_co_out_ratio": round(float(np.quantile(out_vals, 0.50)), 6) if len(out_vals) >= 3 else None,
        "p90_co_out_ratio": round(float(np.quantile(out_vals, 0.90)), 6) if len(out_vals) >= 3 else None,
    }
    row = base_row(level, **kwargs)
    row.update({
        "field": "change_order_rate",
        "cat_budget_division": div,
        "json_blob": json.dumps(blob),
        "n_projects": pdf["ids_project_number"].nunique(),
    })
    constr_rows.append(row)

for (div, sbu), grp in co_pm.groupby(["div_prefix", "cat_sbu"]):
    emit_co_row(grp, "SBU", div, sbu=sbu)

for div, grp in co_pm.groupby("div_prefix"):
    emit_co_row(grp, "GLOBAL", div)

# --- L7: Cost decomposition ratios ---
cost_decomp = (
    bud.filter(F.col("amt_cost_jtd") > 0)
    .withColumn("direct_ratio",    F.col("amt_cost_direct")    / F.col("amt_cost_jtd"))
    .withColumn("committed_ratio", F.col("amt_cost_committed") / F.col("amt_cost_jtd"))
    .groupBy("div_prefix", "cat_budget_category")
    .agg(
        F.expr("percentile_approx(direct_ratio,    array(0.25, 0.50, 0.75))").alias("direct_pcts"),
        F.expr("percentile_approx(committed_ratio, array(0.25, 0.50, 0.75))").alias("committed_pcts"),
        F.count("*").alias("n_rows"),
    )
    .toPandas()
)

for _, r in cost_decomp.iterrows():
    row = base_row("GLOBAL")
    row.update({
        "field": "cost_decomposition_ratios",
        "cat_budget_division": r["div_prefix"],
        "cat_budget_category": r["cat_budget_category"],
        "json_blob": json.dumps({
            "direct":    {"p25": r["direct_pcts"][0],    "p50": r["direct_pcts"][1],    "p75": r["direct_pcts"][2]},
            "committed": {"p25": r["committed_pcts"][0], "p50": r["committed_pcts"][1], "p75": r["committed_pcts"][2]},
        }),
        "n_projects": int(r["n_rows"]),
    })
    profile_rows.append(row)

# --- Write ---
profile_df = spark.createDataFrame(pd.DataFrame(profile_rows))
constr_df  = spark.createDataFrame(pd.DataFrame(constr_rows))

(profile_df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(OUT_PROFILE))

(constr_df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(OUT_CONSTR))

print(f"budget_profile_v1:     {profile_df.count():,} rows")
print(f"budget_constraints_v1: {constr_df.count():,} rows")