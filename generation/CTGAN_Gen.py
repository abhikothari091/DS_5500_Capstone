#%pip install sdv
import numpy as np, pandas as pd, random
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.cag import Inequality
from sdv.constraints.base import Constraint

seed = 42
random.seed(seed); np.random.seed(seed)
try:
    import torch; torch.manual_seed(seed)
except Exception:
    pass

MIN_CTGAN = 15 # lower threshold => more SBUs use CTGAN
FORCE_COPULA_FOR = {"commercial"}  # keep commercial on Copula for better performance

# Simple ratio target 
R_TARGET = 0.985   # 1.5% below contract on average
R_ALPHA = 0.25    # how strongly to pull toward target
R_SIGMA = 0.010   # mild spread 
R_LOW, R_HIGH = 0.93, 1.02  # clamp to allow small overruns, avoid extremes

# inflation index
TURNER_INDEX = {
    2012: 830, 2013: 864, 2014: 902, 2015: 943, 2016: 989, 2017: 1038, 2018: 1096,
    2019: 1156, 2020: 1177, 2021: 1199, 2022: 1295, 2023: 1373, 2024: 1426,
    2025: 1476, 2026: 1476 * 1.04, 2027: 1476 * 1.04**2, 2028: 1476 * 1.04**3,
    2029: 1476 * 1.04**4, 2030: 1476 * 1.04**5
}
def get_inflation_factor(year: float) -> float:
    base = TURNER_INDEX[2024]
    try:
        y = int(year) if pd.notna(year) else 2024
    except Exception:
        y = 2024
    return TURNER_INDEX.get(y, base) / base

# NEW: load profiling bounds (single consolidated table)
# We will use p1–p99 if present; else fall back to p5–p95; else min–max
PROFILE_TBL = "analysts.self_managed.project_meta_profile"
prof_df = spark.read.table(PROFILE_TBL).toPandas()

prof_bounds_sa = {}  # (sbu, asset) -> {field: (low, high)}
prof_bounds_sbu = {} # sbu -> {field: (low, high)}
FIELDS_WITH_BOUNDS = {"n_floors_above_grade","gross_area","amt_contract","amt_cost"}  # bounds applied on these

if not prof_df.empty:
    for c in ["level","cat_sbu","cat_main_asset_class","field"]:
        if c in prof_df.columns:
            prof_df[c] = prof_df[c].astype(str)

    cols = set(prof_df.columns)
    def pick_low(row):
        for k in ["p1","p5","min"]:
            if k in cols and pd.notna(row.get(k)): return float(row[k])
        return None
    def pick_high(row):
        for k in ["p99","p95","max"]:
            if k in cols and pd.notna(row.get(k)): return float(row[k])
        return None

    sub_sa = prof_df[(prof_df["level"]=="SBU_ASSET") & (prof_df["field"].isin(FIELDS_WITH_BOUNDS))].copy()
    for _, r in sub_sa.iterrows():
        key = (str(r["cat_sbu"]), str(r["cat_main_asset_class"]))
        d = prof_bounds_sa.setdefault(key, {})
        lo, hi = pick_low(r), pick_high(r)
        if (lo is not None) and (hi is not None) and lo <= hi:
            d[str(r["field"])] = (lo, hi)

    sub_sbu = prof_df[(prof_df["level"]=="SBU") & (prof_df["field"].isin(FIELDS_WITH_BOUNDS))].copy()
    for _, r in sub_sbu.iterrows():
        key = str(r["cat_sbu"])
        d = prof_bounds_sbu.setdefault(key, {})
        lo, hi = pick_low(r), pick_high(r)
        if (lo is not None) and (hi is not None) and lo <= hi:
            d[str(r["field"])] = (lo, hi)

# load the data
REAL = "analysts.self_managed.project_meta_data_master"
df0 = spark.read.table(REAL).toPandas()

# NEW: derive missingness flags from source (since table dropped them)
df0_flags = pd.DataFrame({
    "is_n_floors_missing": df0["n_floors_above_grade"].isna(),
    "is_gross_area_missing": df0["gross_area"].isna(),
    "is_dt_construction_end_missing": df0["dt_construction_end"].isna(),
    "is_cat_main_asset_class_missing": df0["cat_main_asset_class"].isna()
})

# NEW: build asset-class imputation map (per-SBU mode, fallback global mode)
ac_series = df0["cat_main_asset_class"].astype(str).str.strip()
global_ac_mode = ac_series[ac_series != ""].mode().iloc[0] if len(ac_series[ac_series != ""].mode()) else "Unknown"
ac_mode_by_sbu = (
    df0.assign(cat_sbu=df0["cat_sbu"].astype(str).str.strip())
       .groupby("cat_sbu")["cat_main_asset_class"]
       .agg(lambda s: s.astype(str).str.strip().replace("", np.nan).dropna().mode().iloc[0] if not s.astype(str).str.strip().replace("", np.nan).dropna().empty else global_ac_mode)
       .to_dict()
)

# NEW: empirical distributions from REAL for correlation-preserving sampling
# floors PMF (SBU×Asset → SBU)
floors_pmf_sa, floors_pmf_sbu = {}, {}
_src = df0[["cat_sbu","cat_main_asset_class","n_floors_above_grade"]].copy()
_src["cat_sbu"] = _src["cat_sbu"].astype(str)
_src["n_floors_above_grade"] = pd.to_numeric(_src["n_floors_above_grade"], errors="coerce")
_src = _src.dropna(subset=["n_floors_above_grade"]).copy()
_src["n_floors_above_grade"] = _src["n_floors_above_grade"].astype(int)
tmp = _src.dropna(subset=["cat_main_asset_class"])
for (sbu, asset), g in tmp.groupby(["cat_sbu","cat_main_asset_class"]):
    vc = g["n_floors_above_grade"].value_counts(normalize=True)
    floors_pmf_sa[(str(sbu), str(asset))] = {int(k): float(v) for k, v in vc.items()}
for sbu, g in _src.groupby(["cat_sbu"]):
    vc = g["n_floors_above_grade"].value_counts(normalize=True)
    floors_pmf_sbu[str(sbu)] = {int(k): float(v) for k, v in vc.items()}

# NEW: area-per-floor stats and $/sf + cost/contract ratio stats from REAL (SBU×Asset → SBU)
def _q(g, p): 
    try: return float(np.nanpercentile(g, p))
    except Exception: return None

# area per floor
apf_sa, apf_sbu = {}, {}
tmp_apf = df0.copy()
tmp_apf["n_floors_above_grade"] = pd.to_numeric(tmp_apf["n_floors_above_grade"], errors="coerce")
tmp_apf["gross_area"] = pd.to_numeric(tmp_apf["gross_area"], errors="coerce")
tmp_apf = tmp_apf[(tmp_apf["gross_area"]>0) & (tmp_apf["n_floors_above_grade"]>0)]
tmp_apf["apf"] = tmp_apf["gross_area"] / tmp_apf["n_floors_above_grade"]
for (sbu, asset), g in tmp_apf.dropna(subset=["cat_main_asset_class"]).groupby(["cat_sbu","cat_main_asset_class"]):
    arr = g["apf"].values.astype(float)
    apf_sa[(str(sbu), str(asset))] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))
for sbu, g in tmp_apf.groupby(["cat_sbu"]):
    arr = g["apf"].values.astype(float)
    apf_sbu[str(sbu)] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))

# contract per sqft
cps_sa, cps_sbu = {}, {}
tmp_cps = df0.copy()
tmp_cps["gross_area"] = pd.to_numeric(tmp_cps["gross_area"], errors="coerce")
tmp_cps["amt_contract"] = pd.to_numeric(tmp_cps["amt_contract"], errors="coerce")
tmp_cps = tmp_cps[(tmp_cps["gross_area"]>0) & (tmp_cps["amt_contract"]>0)]
tmp_cps["cps"] = tmp_cps["amt_contract"] / tmp_cps["gross_area"]
for (sbu, asset), g in tmp_cps.dropna(subset=["cat_main_asset_class"]).groupby(["cat_sbu","cat_main_asset_class"]):
    arr = g["cps"].values.astype(float)
    cps_sa[(str(sbu), str(asset))] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))
for sbu, g in tmp_cps.groupby(["cat_sbu"]):
    arr = g["cps"].values.astype(float)
    cps_sbu[str(sbu)] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))

# cost/contract ratio
ratio_sa, ratio_sbu = {}, {}
tmp_r = df0.copy()
tmp_r["amt_contract"] = pd.to_numeric(tmp_r["amt_contract"], errors="coerce")
tmp_r["amt_cost"] = pd.to_numeric(tmp_r["amt_cost"], errors="coerce")
tmp_r = tmp_r[(tmp_r["amt_contract"]>0) & (tmp_r["amt_cost"]>0)]
tmp_r["r"] = tmp_r["amt_cost"] / tmp_r["amt_contract"]
for (sbu, asset), g in tmp_r.dropna(subset=["cat_main_asset_class"]).groupby(["cat_sbu","cat_main_asset_class"]):
    arr = g["r"].values.astype(float)
    ratio_sa[(str(sbu), str(asset))] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))
for sbu, g in tmp_r.groupby(["cat_sbu"]):
    arr = g["r"].values.astype(float)
    ratio_sbu[str(sbu)] = (_q(arr, 50), _q(arr, 25), _q(arr, 75))

# final output columns 
BASE_COLS = [
    "cat_region","cat_sbu",
    "n_floors_above_grade","gross_area",
    "amt_contract","amt_cost",
    "dt_construction_start","dt_construction_end",
    "cat_main_asset_class"
]

# add back the two missingness flags (derived) + keep others
MISSING_FLAGS = ["is_n_floors_missing","is_gross_area_missing","is_cat_main_asset_class_missing","is_dt_construction_end_missing"]

# build df from BASE_COLS first, then inject flags we derived
df = df0[BASE_COLS].copy()
for col in MISSING_FLAGS:
    df[col] = df0_flags[col].values

# filter out rows with missing SBU (as before)
df = df[df["cat_sbu"].notna()].copy()

# data cleanup
df["dt_construction_start"] = pd.to_datetime(df["dt_construction_start"], errors="coerce")
df["dt_construction_end"] = pd.to_datetime(df["dt_construction_end"],   errors="coerce")
# ensure end >= start
bad_dates = (
    df["dt_construction_start"].notna() &
    df["dt_construction_end"].notna() &
    (df["dt_construction_end"] < df["dt_construction_start"])
)
df.loc[bad_dates, "dt_construction_end"] = df.loc[bad_dates, "dt_construction_start"]

# floors for training: simple rule 
df["n_floors_above_grade"] = (
    pd.to_numeric(df["n_floors_above_grade"], errors="coerce")
      .fillna(1).round().clip(lower=1).astype(int)
)

# gross_area for training only (avoid NaNs during fit)
df["gross_area"] = pd.to_numeric(df["gross_area"], errors="coerce")
ga_median = df["gross_area"].dropna().median()
df["gross_area"] = df["gross_area"].fillna(ga_median)

# derived features (training only)
df["duration_days"] = (df["dt_construction_end"] - df["dt_construction_start"]).dt.days
df["start_year"] = df["dt_construction_start"].dt.year
df["inflation_factor"] = df["start_year"].apply(get_inflation_factor)

# adjusted amounts + logs (training core)
df["adj_amt_contract"] = pd.to_numeric(df["amt_contract"], errors="coerce") * df["inflation_factor"]
df["adj_amt_cost"] = pd.to_numeric(df["amt_cost"], errors="coerce") * df["inflation_factor"]
df["log_adj_amt_contract"] = np.log(df["adj_amt_contract"].clip(lower=1))
df["log_adj_amt_cost"] = np.log(df["adj_amt_cost"].clip(lower=1))

# global primary key
df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype=np.int64))

# keep only training columns
TRAIN_KEEP = [
    "row_id",
    "cat_region","cat_sbu",
    "n_floors_above_grade","gross_area",
    "dt_construction_start","dt_construction_end",
    "is_n_floors_missing","is_gross_area_missing","is_cat_main_asset_class_missing","is_dt_construction_end_missing",
    "duration_days","start_year","inflation_factor",
    "adj_amt_contract","adj_amt_cost","log_adj_amt_contract","log_adj_amt_cost",
    "cat_main_asset_class"
]
df = df[TRAIN_KEEP].copy()

# feature engineering on metadata
def make_metadata(frame: pd.DataFrame) -> SingleTableMetadata:
    md = SingleTableMetadata(); md.detect_from_dataframe(frame)
    for col in ["cat_region", "cat_sbu", "cat_main_asset_class"]:
        md.update_column(col, sdtype="categorical")
    for c in ["n_floors_above_grade","duration_days","start_year","inflation_factor",
              "gross_area","adj_amt_contract","adj_amt_cost","log_adj_amt_contract","log_adj_amt_cost"]:
        if c in frame: md.update_column(c, sdtype="numerical")
    for c in ["is_n_floors_missing","is_gross_area_missing","is_cat_main_asset_class_missing","is_dt_construction_end_missing"]:
        if c in frame: md.update_column(c, sdtype="boolean")
    if "dt_construction_start" in frame:
        md.update_column("dt_construction_start", sdtype="datetime", datetime_format="%Y-%m-%d")
    if "dt_construction_end" in frame:
        md.update_column("dt_construction_end", sdtype="datetime", datetime_format="%Y-%m-%d")
    md.update_column("row_id", sdtype="id"); md.set_primary_key("row_id")
    return md

# adding the required constraints
class GrossAreaPositiveConstraint(Constraint):
    """Ensure gross_area > 0 by sampling from fitted lognormal; format to 2 decimals."""
    _is_single_table = True
    def get_columns(self): return ["gross_area"]
    def fit(self, data, metadata=None):
        self.metadata = metadata
        series = pd.to_numeric(data["gross_area"], errors="coerce")
        series = series[series > 0].dropna()
        if len(series) == 0:
            self.mu, self.sigma = np.log(1000.0), 0.75
        else:
            logs = np.log(series)
            self.mu  = float(logs.mean())
            self.sigma = float(logs.std(ddof=1) or 0.5)
    def get_updated_metadata(self, metadata): return metadata
    def transform(self, data): return data
    def reverse_transform(self, data):
        ga  = pd.to_numeric(data["gross_area"], errors="coerce")
        bad = ga.isna() | (ga <= 0)
        if bad.any():
            n_bad = int(bad.sum())
            repl = np.random.lognormal(mean=self.mu, sigma=max(self.sigma, 1e-6), size=n_bad)
            repl = np.clip(repl, 1.0, None)
            data.loc[bad, "gross_area"] = repl
        data["gross_area"] = pd.to_numeric(data["gross_area"], errors="coerce").round(2)
        return data

class SimpleFloorsConstraint(Constraint):
    """Previous behavior: floors int ≥1, cap to 3 for mission critical."""
    _is_single_table = True
    def get_columns(self): return ["cat_sbu","n_floors_above_grade"]
    def fit(self, data, metadata=None): self.metadata = metadata
    def get_updated_metadata(self, metadata): return metadata
    def transform(self, data): return data
    def reverse_transform(self, data):
        floors = pd.to_numeric(data["n_floors_above_grade"], errors="coerce").fillna(1)
        floors = floors.round().clip(lower=1)
        sbu = data["cat_sbu"].astype(str).str.lower()
        mc_mask = sbu.eq("mission critical")
        floors.loc[mc_mask] = floors.loc[mc_mask].clip(upper=3)
        data["n_floors_above_grade"] = floors.astype(int)
        return data

class InflationFactorConstraint(Constraint):
    """Recompute inflation_factor from start_year using Turner Index mapping."""
    _is_single_table = True
    def get_columns(self): return ["start_year","inflation_factor"]
    def fit(self, data, metadata=None): self.metadata = metadata
    def get_updated_metadata(self, metadata): return metadata
    def transform(self, data): return data
    def reverse_transform(self, data):
        years = pd.to_numeric(data["start_year"], errors="coerce")
        data["inflation_factor"] = years.apply(get_inflation_factor)
        return data

class SimpleCostRatioNudgeConstraint(Constraint):
    _is_single_table = True
    def get_columns(self): return ["adj_amt_cost","adj_amt_contract"]
    def fit(self, data, metadata=None): self.metadata = metadata
    def get_updated_metadata(self, metadata): return metadata
    def transform(self, data): return data
    def reverse_transform(self, data):
        c = pd.to_numeric(data["adj_amt_contract"], errors="coerce").clip(lower=1)
        k = pd.to_numeric(data["adj_amt_cost"], errors="coerce").clip(lower=1)
        r = (k / c).replace([np.inf, -np.inf], np.nan).fillna(R_TARGET)
        eps = np.random.normal(0.0, R_SIGMA, size=len(r))
        r_new = (1.0 - R_ALPHA) * r + R_ALPHA * R_TARGET + eps
        r_new = np.clip(r_new, R_LOW, R_HIGH)
        data["adj_amt_contract"] = c
        data["adj_amt_cost"] = (c * r_new).clip(lower=1)
        # keep logs coherent if present
        if "log_adj_amt_contract" in data:
            data["log_adj_amt_contract"] = np.log(data["adj_amt_contract"].clip(lower=1))
        if "log_adj_amt_cost" in data:
            data["log_adj_amt_cost"] = np.log(data["adj_amt_cost"].clip(lower=1))
        return data

# FIX: slightly stronger soft clamp & variety to reduce "sticky" duplicates
def _soften_continuous(val, low, high, jitter_pct=0.06, margin_pct=0.15):
    try:
        v = float(val)
    except Exception:
        return val
    if not np.isfinite(v):
        return val
    lo = None if low  is None else float(low)  * (1 - margin_pct)
    hi = None if high is None else float(high) * (1 + margin_pct)
    if (lo is not None) and (v < lo):
        v = float(low) * (1 + np.random.uniform(0.0, margin_pct))
    elif (hi is not None) and (v > hi):
        v = float(high) * (1 - np.random.uniform(0.0, margin_pct))
    else:
        v = v * (1 + np.random.uniform(-jitter_pct, jitter_pct))
    if low  is not None: v = max(v, float(low))
    if high is not None: v = min(v, float(high))
    return v

# discrete sampler for floors with smoothed PMF
def _sample_floor_from_pmf(sbu, asset, low, high):
    pmf = floors_pmf_sa.get((sbu, asset))
    if pmf is None:
        pmf = floors_pmf_sbu.get(sbu)
    if pmf is None:
        lo = int(np.floor(low)) if low is not None else 1
        hi = int(np.ceil(high)) if high is not None else 5
        if lo > hi: lo, hi = 1, 5
        return np.random.randint(lo, hi + 1)
    lo = int(np.floor(low)) if low is not None else min(pmf.keys())
    hi = int(np.ceil(high)) if high is not None else max(pmf.keys())
    if lo > hi:
        lo, hi = min(pmf.keys()), max(pmf.keys())
    keys = list(range(lo, hi + 1))
    eps = 1e-3
    weights = np.array([pmf.get(k, 0.0) + eps for k in keys], dtype=float)
    weights = weights / weights.sum()
    return int(np.random.choice(keys, p=weights))

# sample a value near a central tendency within interquartile-ish range
def _sample_around(mid, lo, hi, widen=0.10):
    if mid is None or lo is None or hi is None or not np.isfinite(mid):
        return None
    lo2 = lo * (1 - widen)
    hi2 = hi * (1 + widen)
    if lo2 >= hi2:
        lo2, hi2 = lo, hi
    return float(np.random.triangular(left=lo2, mode=mid, right=hi2))

# safer stats getter
def _get_stats(sa_dict, sbu_dict, sbu, asset):
    t = sa_dict.get((sbu, asset))
    if isinstance(t, (tuple, list)) and len(t) == 3:
        return t
    t = sbu_dict.get(sbu)
    if isinstance(t, (tuple, list)) and len(t) == 3:
        return t
    return (None, None, None)

# soft, SBU×Asset-aware row post-processor with correlation preservation
def clamp_row_by_profile(row):
    sbu = str(row.get("cat_sbu"))
    asset = str(row.get("cat_main_asset_class"))
    bounds = prof_bounds_sa.get((sbu, asset))
    if bounds is None:
        bounds = prof_bounds_sbu.get(sbu)

    # floors (discrete with PMF + soft bounds)
    low_f = high_f = None
    if bounds and "n_floors_above_grade" in bounds:
        low_f, high_f = bounds["n_floors_above_grade"]
    fval = row.get("n_floors_above_grade", 1)
    try:
        fval = int(fval)
    except Exception:
        fval = 1
    # FIX: more variety in floors
    need_resample = (low_f is not None and fval < low_f) or (high_f is not None and fval > high_f) or (np.random.rand() < 0.50)
    if need_resample:
        nf = _sample_floor_from_pmf(sbu, asset, low_f, high_f)
    else:
        nf = fval
        if np.random.rand() < 0.30:
            step = np.random.choice([-1, 1])
            cand = nf + step
            if (low_f is None or cand >= low_f) and (high_f is None or cand <= high_f):
                nf = cand
    if sbu.lower() == "mission critical":
        nf = min(nf, 3)
    nf = max(int(nf), 1)
    row["n_floors_above_grade"] = nf

    # gross area: derive from area-per-floor stats * floors (correlated), then soft bound
    apf_mid, apf_lo, apf_hi = _get_stats(apf_sa, apf_sbu, sbu, asset)
    ga = row.get("gross_area")
    try:
        ga = float(ga)
    except Exception:
        ga = np.nan
    # FIX: add variability in APF resampling
    if np.isnan(ga) or np.random.rand() < 0.55:
        apf = _sample_around(apf_mid, apf_lo, apf_hi, widen=0.12) if apf_mid is not None else None
        if apf is None:
            apf = 10000.0  # safe fallback sqft/floor
        ga = float(apf) * float(nf)
    if bounds and "gross_area" in bounds:
        lo, hi = bounds["gross_area"]
        ga = _soften_continuous(ga, lo, hi, jitter_pct=0.06, margin_pct=0.15)
    # FIX: tiny anti-stick nudge
    ga *= (1.0 + np.random.uniform(-0.01, 0.01))
    row["gross_area"] = round(max(ga, 1.0), 2)

    # contract per sqft: sample near median (group-aware), amounts derived from $/sf × area
    cps_mid, cps_lo, cps_hi = _get_stats(cps_sa, cps_sbu, sbu, asset)
    cps = _sample_around(cps_mid, cps_lo, cps_hi, widen=0.15) if cps_mid is not None else None
    if cps is None:
        cps = 350.0  # conservative fallback $/sf
    if sbu.lower() == "mission critical":
        cps *= (1.0 + np.random.uniform(-0.08, 0.08))  # extra spread for MC only
    contract = cps * row["gross_area"]
    if bounds and "amt_contract" in bounds:
        lo, hi = bounds["amt_contract"]
        contract = _soften_continuous(contract, lo, hi, jitter_pct=0.05, margin_pct=0.15)
    # FIX: tiny anti-stick nudge
    contract *= (1.0 + np.random.uniform(-0.01, 0.01))
    row["amt_contract"] = contract

    # cost/contract ratio: sample near median (group-aware), then business band
    r_mid, r_lo, r_hi = _get_stats(ratio_sa, ratio_sbu, sbu, asset)
    r = _sample_around(r_mid, r_lo, r_hi, widen=0.06) if r_mid is not None else None
    if r is None:
        r = R_TARGET
    r = float(np.clip(r, R_LOW, R_HIGH))
    cost = contract * r
    if bounds and "amt_cost" in bounds:
        lo, hi = bounds["amt_cost"]
        cost = _soften_continuous(cost, lo, hi, jitter_pct=0.05, margin_pct=0.15)
    row["amt_cost"] = cost

    return row

# count for number of records per sbu
TARGET_TOTAL = 1000  
proportions = df["cat_sbu"].astype(str).value_counts(normalize=True)
counts = (proportions * TARGET_TOTAL).round().astype(int)
delta = TARGET_TOTAL - int(counts.sum())
if delta != 0:
    counts.iloc[counts.argmax()] += delta  # adjust rounding delta to largest bucket

groups = {k: v.copy() for k, v in df.groupby("cat_sbu", dropna=True)}
sizes  = {k: len(v) for k, v in groups.items()}

# avoid modeling on ultra-tiny groups 
too_small_threshold = 5
tiny_sb_us = [k for k, n in sizes.items() if n < too_small_threshold]
if tiny_sb_us:
    carry = int(sum(counts.get(k, 0) for k in tiny_sb_us))
    for k in tiny_sb_us:
        counts[k] = 0
    viable = [k for k, n in sizes.items() if n >= too_small_threshold]
    if viable and carry > 0:
        counts[max(viable, key=lambda k: sizes[k])] += carry

# sample for per sbu
def postprocess_block(s: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(s["dt_construction_start"], errors="coerce")
    end = pd.to_datetime(s["dt_construction_end"],   errors="coerce")
    bad = start.notna() & end.notna() & (end < start)
    end.loc[bad] = start.loc[bad]
    s["dt_construction_start"] = start
    s["dt_construction_end"]   = end

    # floors 
    floors = pd.to_numeric(s["n_floors_above_grade"], errors="coerce").fillna(1).round().clip(lower=1)
    sbu = s["cat_sbu"].astype(str).str.lower().str.strip()   # FIX: normalize SBU
    mc_mask = sbu.eq("mission critical")
    floors.loc[mc_mask] = floors.loc[mc_mask].clip(upper=3)
    s["n_floors_above_grade"] = floors.astype(int)

    # inflation factor & nominal amounts
    s["start_year"] = pd.to_numeric(s["start_year"], errors="coerce")
    s["inflation_factor"] = s["start_year"].apply(get_inflation_factor)
    factor = pd.to_numeric(s["inflation_factor"], errors="coerce").replace(0, np.nan)
    s["amt_contract"] = (pd.to_numeric(s["adj_amt_contract"], errors="coerce").clip(lower=1) / factor)
    s["amt_cost"] = (pd.to_numeric(s["adj_amt_cost"], errors="coerce").clip(lower=1) / factor)

    # gross area: >0 and formatted
    s["gross_area"] = pd.to_numeric(s["gross_area"], errors="coerce").clip(lower=1).round(2)

    # apply profiling-based, correlation-preserving softening (SBU×Asset → SBU)
    if prof_bounds_sa or prof_bounds_sbu or floors_pmf_sa or floors_pmf_sbu:
        s = s.apply(clamp_row_by_profile, axis=1)

    # FIX: final ratio clamp (strict) to ensure business band PLUS tiny noise
    # (works on whatever clamp_row_by_profile produced)
    k = pd.to_numeric(s["amt_cost"], errors="coerce")
    c = pd.to_numeric(s["amt_contract"], errors="coerce").replace(0, np.nan)
    r = (k / c).replace([np.inf, -np.inf], np.nan)
    r_clamped = r.clip(lower=R_LOW, upper=R_HIGH) * (1.0 + np.random.uniform(-0.005, 0.005, size=len(r)))
    s["amt_cost"] = (c * r_clamped).fillna(s["amt_cost"])  # keep previous if NaN

    # NEW: standardize asset class + impute from SBU mode if missing/blank
    ac = s["cat_main_asset_class"].astype(str).str.strip()
    ac = ac.mask(ac.eq("") | ac.str.lower().eq("nan"))
    # map by SBU (case-insensitive)
    sbu_clean = s["cat_sbu"].astype(str).str.strip()
    fill_by_sbu = sbu_clean.map(lambda x: ac_mode_by_sbu.get(str(x), global_ac_mode))
    ac = ac.fillna(fill_by_sbu).fillna(global_ac_mode)
    s["cat_main_asset_class"] = ac

    # drop rows with any nulls/infs
    s = s.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return s

parts = []

for sbu_key, part in groups.items():
    n_real = len(part)
    n_synth_target = int(counts.get(sbu_key, 0))
    if n_synth_target <= 0:
        continue

    # Re-key within group
    if "row_id" in part.columns:
        part["row_id"] = part["row_id"].astype(np.int64)
    else:
        part.insert(0, "row_id", np.arange(1, len(part) + 1, dtype=np.int64))

    md = make_metadata(part)

    constraints = [
        Inequality(low_column_name="dt_construction_start", high_column_name="dt_construction_end"),
        GrossAreaPositiveConstraint(),
        SimpleFloorsConstraint(),
        InflationFactorConstraint(),
        SimpleCostRatioNudgeConstraint(),  # nudge toward 1.5% gap with mild spread
    ]

    sbu_l = str(sbu_key).lower()
    if (sbu_l in FORCE_COPULA_FOR) or (n_real < MIN_CTGAN):
        synth = GaussianCopulaSynthesizer(metadata=md)
    else:
    
        n_fit = len(part)
        PAC = next((p for p in (8,7,6,5,4,3,2) if n_fit % p == 0), 1)
        BATCH = None
        for bs in range(min(128, n_fit), PAC, -1):
            if bs % PAC == 0 and (n_fit % bs) % PAC == 0:
                BATCH = bs; break
        if BATCH is None: BATCH = min(128, max(PAC, n_fit))
        # make batch at least 2 and even
        BATCH = max(2, BATCH)
        if BATCH % 2 != 0:
            BATCH += 1
        rem = BATCH % PAC
        if rem != 0:
            BATCH += (PAC - rem)

        synth = CTGANSynthesizer(
            metadata=md,
            epochs=700,
            batch_size=BATCH,
            generator_lr=2e-2, discriminator_lr=2e-2,
            discriminator_steps=7,
            embedding_dim=64,
            generator_dim=(128,128),
            discriminator_dim=(128,128),
            pac=PAC,
            cuda=False,
            verbose=False
        )

    synth.add_constraints(constraints)
    synth.fit(part)

    # fill this SBU quota with no nulls
    block = []
    remaining = n_synth_target
    max_passes = 30
    passes = 0
    while remaining > 0 and passes < max_passes:
        passes += 1
        batch = max(remaining, int(remaining * 2))  
        samp = synth.sample(batch).drop(columns=["row_id"], errors="ignore")
        samp = postprocess_block(samp)
        if len(samp) > 0:
            take = min(len(samp), remaining)
            block.append(samp.iloc[:take].copy())
            remaining -= take
    if remaining > 0:
        print(f"WARNING: SBU '{sbu_key}' only filled {n_synth_target - remaining}/{n_synth_target} rows.")

    sbu_out = pd.concat(block, ignore_index=True)
    parts.append(sbu_out)

# post process
syn_full = pd.concat(parts, ignore_index=True)

# keep base columns 
syn = syn_full[BASE_COLS + MISSING_FLAGS].copy()

# unique synthetic ID to map projects later
def _abbr(x):
    try:
        return "".join(ch for ch in str(x) if ch.isalnum())[:3].upper() or "GEN"
    except Exception:
        return "GEN"

syn = syn.reset_index(drop=True)
syn.insert(0, "ids_project_number_synth",
           ["SYN-" + _abbr(s) + "-" + str(i+1).zfill(6) for i, s in enumerate(syn["cat_sbu"])])

# formatting
syn["gross_area"] = pd.to_numeric(syn["gross_area"], errors="coerce").clip(lower=1).round(2)
syn["n_floors_above_grade"] = (
    pd.to_numeric(syn["n_floors_above_grade"], errors="coerce").fillna(1).round().clip(lower=1).astype(int)
)
# FIX: round cost/contract now that ratio is clamped
syn["amt_contract"] = pd.to_numeric(syn["amt_contract"], errors="coerce").clip(lower=1).round(2)
syn["amt_cost"] = pd.to_numeric(syn["amt_cost"], errors="coerce").clip(lower=1).round(2)

# cast to date for Spark write
syn["dt_construction_start"] = pd.to_datetime(syn["dt_construction_start"], errors="coerce").dt.date
syn["dt_construction_end"]   = pd.to_datetime(syn["dt_construction_end"],   errors="coerce").dt.date

# strict no-nulls check and row count assertion
syn = syn.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
expected_total = 1000
assert len(syn) == expected_total, f"Row count mismatch: {len(syn)} != {expected_total}"

# save the table
from pyspark.sql import functions as F
spark_df = spark.createDataFrame(syn) \
    .withColumn("dt_construction_start", F.to_date("dt_construction_start")) \
    .withColumn("dt_construction_end",   F.to_date("dt_construction_end"))

(
    spark_df.write
    .format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("analysts.self_managed.synthetic_project_meta_data_profiled")
)

print("Saved:", spark_df.count(), "rows")
