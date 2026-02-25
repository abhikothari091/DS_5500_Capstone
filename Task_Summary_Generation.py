import numpy as np
import pandas as pd
import random
import json
from datetime import timedelta
from pyspark.sql import functions as F

# Configuration
SEED_TABLE             = "analysts.self_managed.synthetic_project_meta_data_profiled"
SCHEDULE_TABLE         = "analysts.self_managed.synthetic_schedule_baseline_v1"
PROFILE_TABLE          = "analysts.self_managed.task_summary_profile"
CONSTRAINTS_TABLE      = "analysts.self_managed.task_summary_constraints"
FREQUENCY_TABLE        = "analysts.self_managed.task_summary_sow_frequency"
OVERLAP_TABLE          = "analysts.self_managed.task_summary_sow_overlaps"
REBASELINE_METADATA    = "analysts.self_managed.rebaseline_event_metadata_v1"
COMPLETION_PROG_TABLE  = "analysts.self_managed.task_summary_completion_progression"
DATE_SHIFT_TABLE       = "analysts.self_managed.task_summary_date_shift_profiles"
OUT_TABLE              = "analysts.self_managed.synthetic_task_summary_by_sow_v1"

SEED = 42
MEDIAN_DELAY_DAYS = 41.0

random.seed(SEED)
np.random.seed(SEED)

# Defined early — used in pre-indexing section before V2 helpers block
def resolve_project_boundary(dt_tco, dt_next_tco, dt_end):
    candidates = []
    if pd.notna(dt_tco):      candidates.append(pd.Timestamp(dt_tco))
    if pd.notna(dt_next_tco): candidates.append(pd.Timestamp(dt_next_tco))
    return max(candidates) if candidates else pd.Timestamp(dt_end)

# Minimum SOW durations by category (days)
# Based on profiled construction data/heuristic — no SOW should be
# shorter than its realistic minimum execution window
SOW_MIN_DURATIONS = {
    # Structural / civil — physical work takes weeks minimum
    '(13) Below Grade Structure + Foundations': 30,
    '(14) Superstructure (Concrete Structure)': 21,
    '(14A) Steel Erection':                     21,
    '(14B) Deck & Detail Steel':                14,
    '(14C) Concrete on Metal Decks':            14,
    '(14D) Spray Fireproofing':                 14,
    '(14E) Structural Wood Framing':            14,
    # Envelope
    '(17) Facade (Weathertight)':               21,
    '(17B) Facade - Complete':                  21,
    '(18) Roofing (Weathertight)':              14,
    '(18B) Roofing (After Weathertight)':       14,
    # MEP / systems
    '(25) Interior Rough':                      30,
    '(26) Interior Finishes':                   30,
    '(29) Fire Life Safety':                    14,
    '(30) Systems Start-up & Testing':          14,
    '(08) Permanent Power Distributed':         14,
    '(07) Permanent Power to Site':             14,
    # Site work
    '(12) Initial Site Work':                   14,
    '(12A) Enabling':                           14,
    '(12B) Initial Site Work Remaining':        14,
    '(12C) Demo & Abatement':                   14,
    '(28) Final Site Work':                     14,
    # Hoists / cranes — installation takes time
    '(19) Hoist 1-  Install & Usage Duration':  14,
    '(21) Hoist 1 - Infill Duration':           14,
    '(22) Hoist 2 - Infill Duration':           14,
    '(15) Tower Crane 1 - Usage Duration':      14,
    '(16) Tower Crane 2 - Usage Duration':      14,
    # Close-out — inspections take at least a week
    '(31) Final Inspections & Close-out':       7,
    '(27) Worklist & Punch list':               7,
    '(01) Notice to Proceed (Start Construction)': 7,
}
SOW_MIN_DURATION_DEFAULT = 7  # absolute floor for any SOW

# SOW Overrun Profiles (from real Platinum data)
# (probability, median_days, p75_days, p90_days)

SOW_OVERRUN_PROFILES = {
    '(31) Final Inspections & Close-out':      (0.5714, 62,  169, 443),
    '(33) NO CODE (All remaining activities)': (0.5619, 77,  184, 471),
    '(27) Worklist & Punch list':              (0.5083, 66,  168, 316),
    '(26) Interior Finishes':                  (0.3333, 90,  244, 567),
    '(30) Systems Start-up & Testing':         (0.3301, 79,  234, 462),
    '(28) Final Site Work':                    (0.2695, 118, 269, 751),
    '(22) Hoist 2 - Infill Duration':          (0.2667, 78,  273, 442),
    '(32) Weather Contingency':                (0.2602, 43,  257, 556),
    '(29) Fire Life Safety':                   (0.2414, 151, 304, 795),
    '(11) All Mat/Equip Deliveries':           (0.2149, 136, 305, 718),
    '(21) Hoist 1 - Infill Duration':          (0.1954, 271, 437, 700),
    '(25) Interior Rough':                     (0.1814, 174, 408, 831),
    '(17B) Facade - Complete':                 (0.1400, 62,  421, 819),
    '(24) Remaining Elevator Install Duration':(0.1319, 82,  270, 868),
}

# Load All Required Tables
print("Loading tables...")

profiles_df       = spark.read.table(PROFILE_TABLE).toPandas()
constraints_df    = spark.read.table(CONSTRAINTS_TABLE).toPandas()
frequency_df      = spark.read.table(FREQUENCY_TABLE).toPandas()
overlap_df        = spark.read.table(OVERLAP_TABLE).toPandas()
completion_prog_df= spark.read.table(COMPLETION_PROG_TABLE).toPandas()
date_shift_df     = spark.read.table(DATE_SHIFT_TABLE).toPandas()

def parse_json_bounds(json_str):
    if pd.isna(json_str):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}

constraints_df['bounds'] = constraints_df['bounds_json'].apply(parse_json_bounds)

# Load seed projects
seed_df = spark.read.table(SEED_TABLE).select(
    "ids_project_number_synth", "cat_sbu", "cat_region",
    "cat_main_asset_class", "gross_area", "amt_contract",
    "dt_construction_start", "dt_construction_end"
).toPandas()

seed_df['size_category'] = pd.cut(
    seed_df['gross_area'],
    bins=[0, 50000, 100000, 200000, np.inf],
    labels=['Small', 'Medium', 'Large', 'XLarge']
).astype(str)

seed_df['contract_bucket'] = pd.cut(
    seed_df['amt_contract'],
    bins=[0, 50000000, 100000000, 200000000, np.inf],
    labels=['Medium', 'Large', 'XLarge', 'XXLarge']
).astype(str)

seed_df['dt_construction_start'] = pd.to_datetime(seed_df['dt_construction_start'])
seed_df['dt_construction_end']   = pd.to_datetime(seed_df['dt_construction_end'])
seed_df['project_duration_days'] = (
    seed_df['dt_construction_end'] - seed_df['dt_construction_start']
).dt.days

# Load ALL schedule baseline versions for version linkage
schedule_all_df = spark.read.table(SCHEDULE_TABLE).select(
    "ids_project_number_synth", "cat_schedule_code",
    "dt_tco", "dt_next_tco", "dt_effective_from", "is_active_record"
).toPandas()

schedule_all_df['dt_tco']           = pd.to_datetime(schedule_all_df['dt_tco'])
schedule_all_df['dt_next_tco']      = pd.to_datetime(schedule_all_df['dt_next_tco'])
schedule_all_df['dt_effective_from']= pd.to_datetime(schedule_all_df['dt_effective_from'])

# Merge active TCO into seed
schedule_active = schedule_all_df[schedule_all_df['is_active_record'] == True].copy()
seed_df = seed_df.merge(
    schedule_active[['ids_project_number_synth', 'dt_tco', 'dt_next_tco']],
    on='ids_project_number_synth', how='left'
)

# Load rebaseline metadata
rebaseline_df = spark.read.table(REBASELINE_METADATA).toPandas()
rebaseline_df['dt_rebaseline_occurred'] = pd.to_datetime(rebaseline_df['dt_rebaseline_occurred'])

print(f"Loaded {len(seed_df)} projects")
print(f"Loaded {len(schedule_all_df)} schedule versions")
print(f"Loaded {len(rebaseline_df)} rebaseline events")

print("Pre-indexing lookup tables...")

# profiles: (level, field, cat_sbu, size_category, contract_bucket, asset_class) -> row
profiles_index = {}
for _, row in profiles_df.iterrows():
    key = (
        row.get('level'), row.get('field'),
        row.get('cat_sbu'), row.get('size_category'),
        row.get('contract_bucket'), row.get('cat_main_asset_class')
    )
    profiles_index[key] = row

# constraints: (level, field, cat_sbu, cat_sow, size_category, contract_bucket) -> bounds
constraints_index = {}
for _, row in constraints_df.iterrows():
    key = (
        row.get('level'), row.get('field'),
        row.get('cat_sbu'), row.get('cat_sow'),
        row.get('size_category'), row.get('contract_bucket')
    )
    constraints_index[key] = row['bounds']

# overlaps: (level, sow_pair) -> plain dict of scalars — deduplicate by highest n_transitions
overlap_index = {}
for _, row in overlap_df.sort_values('n_transitions', ascending=False).iterrows():
    key = (row['level'], row['sow_pair'])
    if key not in overlap_index:
        overlap_index[key] = {
            'p25_pct':    float(row['p25_pct']),
            'median_pct': float(row['median_pct']),
            'p75_pct':    float(row['p75_pct']),
        }

# completion progression: (level, cat_sbu, progress_bucket, pos_bucket) -> row
completion_index = {}
for _, row in completion_prog_df.iterrows():
    key = (
        row['level'], row.get('cat_sbu'),
        row['project_progress_bucket'], row['sow_position_bucket']
    )
    completion_index[key] = row

# date shift: (completion_bucket, position_bucket) -> row
date_shift_index = {}
for _, row in date_shift_df.iterrows():
    date_shift_index[(row['completion_bucket'], row['position_bucket'])] = row

# frequency: cat_sbu -> {sows, probs}
frequency_index = {}
for sbu_val, grp in frequency_df.groupby('cat_sbu'):
    grp = grp.copy()
    grp['prob'] = grp['pct_projects'] / grp['pct_projects'].sum()
    frequency_index[sbu_val] = {
        'sows':  grp['cat_sow'].values,
        'probs': grp['prob'].values
    }

# Pre-group schedule versions and rebaselines by project
schedules_by_project = {
    pid: grp.sort_values('dt_effective_from').reset_index(drop=True)
    for pid, grp in schedule_all_df.groupby('ids_project_number_synth')
}
rebaselines_by_project = {
    pid: grp.sort_values('rebaseline_number').reset_index(drop=True)
    for pid, grp in rebaseline_df.groupby('ids_project_number_synth')
}

# Pre-build boundary dict for fast lookup in final cap pass
boundary_by_project = {}
for _, proj in seed_df.iterrows():
    pid         = proj['ids_project_number_synth']
    dt_tco_p    = proj.get('dt_tco')
    dt_next_p   = proj.get('dt_next_tco')
    dt_end_p    = proj['dt_construction_end']
    dt_tco_p    = pd.Timestamp(dt_tco_p) if pd.notna(dt_tco_p) else pd.Timestamp(dt_end_p)
    boundary_by_project[pid] = resolve_project_boundary(dt_tco_p, dt_next_p, dt_end_p)

# avg_task_duration: cat_sow -> {median, p25, p75}
avg_duration_index = {}
for _, row in profiles_df[profiles_df['field'] == 'avg_task_duration'].iterrows():
    if pd.notna(row.get('cat_sow')):
        avg_duration_index[row['cat_sow']] = {
            'p25':    float(row['p25'])    if pd.notna(row.get('p25'))    else float(row['median']),
            'median': float(row['median']),
            'p75':    float(row['p75'])    if pd.notna(row.get('p75'))    else float(row['median'])
        }

print("✅ All lookup tables pre-indexed")

# Helpers

def hierarchical_lookup_profile(profiles_df, sbu, size_cat, contract_bucket, asset_class, field):
    lookups = [
        ('SBU_CONTRACT', sbu,  None,     contract_bucket, None),
        ('SBU_SIZE',     sbu,  size_cat, None,            None),
        ('SBU_ASSET',    sbu,  None,     None,            asset_class),
        ('SBU',          sbu,  None,     None,            None),
        ('GLOBAL',       None, None,     None,            None),
    ]
    for level, s, sz, cb, ac in lookups:
        key = (level, field, s, sz, cb, ac)
        if key in profiles_index:
            return profiles_index[key]
    return None

def hierarchical_lookup_constraint(constraints_df, sbu, size_cat, contract_bucket, cat_sow, field):
    lookups = [
        ('SOW_SBU',      sbu,  cat_sow, None,     None),
        ('SOW_TYPE',     None, cat_sow, None,      None),
        ('SBU_CONTRACT', sbu,  None,    None,      contract_bucket),
        ('SBU_SIZE',     sbu,  None,    size_cat,  None),
        ('SBU',          sbu,  None,    None,      None),
        ('GLOBAL',       None, None,    None,      None),
    ]
    for level, s, sow, sz, cb in lookups:
        key = (level, field, s, sow, sz, cb)
        if key in constraints_index:
            return constraints_index[key]
    return {}

def safe_triangular(p25, median, p75):
    """np.random.triangular safe wrapper — handles zero-variance buckets."""
    p25 = min(float(p25), float(median))
    p75 = max(float(p75), float(median))
    if p25 == median == p75:
        return float(median)
    return np.random.triangular(p25, median, p75)

def sample_from_bounds(bounds, use_triangular=True):
    if not bounds:
        return None
    p25    = bounds.get('p25',    bounds.get('median', 0))
    median = bounds.get('median', 0)
    p75    = bounds.get('p75',    bounds.get('median', 0))
    if p25 == median == p75:
        return median
    if p25 == median:
        p25 = max(0, median - 0.5)
    if p75 == median:
        p75 = median + 0.5
    val = safe_triangular(p25, median, p75) if use_triangular else np.random.uniform(p25, p75)
    val *= (1 + np.random.uniform(-0.05, 0.05))
    p95 = bounds.get('p95')
    if p95 is not None and pd.notna(p95):
        val = min(val, p95)
    return val

def sample_n_sows(sbu, size_cat, contract_bucket, asset_class):
    row = hierarchical_lookup_profile(profiles_df, sbu, size_cat, contract_bucket, asset_class, 'n_sows_per_project')
    if row is None:
        return 20
    n = sample_from_bounds({'p25': row.get('p25'), 'median': row.get('median'),
                             'p75': row.get('p75'), 'mean': row.get('mean')})
    return int(np.clip(np.round(n), 5, 40))

def sample_sow_types(sbu, n_sows_needed):
    entry = frequency_index.get(sbu) or frequency_index.get('commercial')
    if entry is None:
        return []
    n = min(n_sows_needed, len(entry['sows']))
    return list(np.random.choice(entry['sows'], size=n, replace=False, p=entry['probs']))

def sample_sequence_position(sow, sbu):
    bounds = hierarchical_lookup_constraint(constraints_df, sbu, None, None, sow, 'sequence_position')
    if not bounds:
        if 'Foundation' in sow or 'Below Grade' in sow: return 6
        elif 'Finishes' in sow or 'Punch' in sow or 'Close' in sow: return 18
        elif 'Precon' in sow or 'Buys' in sow or 'Submittals' in sow: return 3
        else: return 10
    mean   = bounds.get('mean',   10)
    stddev = bounds.get('stddev', 3)
    return int(np.clip(np.round(np.random.normal(mean, stddev)),
                       bounds.get('min', 1), bounds.get('max', 30)))

def get_overlap_pattern(sow_first, sow_second):
    row = (overlap_index.get(('PAIR', f"{sow_first} → {sow_second}"))
           or overlap_index.get(('GENERAL', 'all_pairs')))
    if row is None:
        return 0.84
    return np.clip(
        safe_triangular(
            float(row['p25_pct']),
            float(row['median_pct']),
            float(row['p75_pct'])
        ), -50, 100
    ) / 100.0

def sample_task_count(sow, sbu):
    # Milestones are markers, not real scopes — cap at realistic small counts
    if sow in ('(01) Notice to Proceed (Start Construction)',
                '(02) Substantial Completion (TCO or 1st Turnover Milestone)',
                '(00) GMP Execution Date'):
        return np.random.randint(1, 10)
    bounds = hierarchical_lookup_constraint(constraints_df, sbu, None, None, sow, 'n_tasks')
    if not bounds:
        return 100
    return int(np.clip(np.round(sample_from_bounds(bounds)), 10, 10000))

def sample_sow_duration(sow, sbu):
    bounds = hierarchical_lookup_constraint(constraints_df, sbu, None, None, sow, 'sow_duration_days')
    if not bounds:
        return 180
    return int(np.clip(np.round(sample_from_bounds(bounds)), 30, 1000))

def calculate_avg_task_duration(sow, sbu=None):
    """Look up avg task duration from pre-indexed profile table."""
    row = avg_duration_index.get(sow)
    if row is not None:
        return round(max(1, safe_triangular(row['p25'], row['median'], row['p75'])), 1)
    return 30

def sample_concurrency_depth(sbu, size_cat):
    row = hierarchical_lookup_profile(profiles_df, sbu, size_cat, None, None, 'max_concurrent_sows')
    if row is None:
        return {'max': 10}
    max_c = int(np.round(sample_from_bounds({
        'p25': row.get('p25'), 'median': row.get('median'),
        'p75': row.get('p75'), 'mean':   row.get('mean')
    })))
    return {'max': max(3, max_c)}

def resolve_project_boundary(dt_tco, dt_next_tco, dt_end):
    candidates = []
    if pd.notna(dt_tco):      candidates.append(pd.Timestamp(dt_tco))
    if pd.notna(dt_next_tco): candidates.append(pd.Timestamp(dt_next_tco))
    return max(candidates) if candidates else pd.Timestamp(dt_end)

def apply_proportional_duration_scaling(sow_data, dt_start, dt_boundary):
    if not sow_data:
        return sow_data
    latest_finish = max(s['dt_sow_finish'] for s in sow_data)
    if latest_finish <= dt_boundary:
        return sow_data
    project_span = (dt_boundary - dt_start).days
    actual_span  = (latest_finish - dt_start).days
    if actual_span <= 0 or project_span <= 0:
        return sow_data
    scale_factor = max(project_span / actual_span, 0.50)
    for sow in sow_data:
        scaled_offset   = max(0, int((sow['dt_sow_start'] - dt_start).days * scale_factor))
        scaled_duration = max(30, int(sow['sow_duration_days'] * scale_factor))
        sow['dt_sow_start']      = dt_start + timedelta(days=scaled_offset)
        sow['sow_duration_days'] = scaled_duration
        sow['dt_sow_finish']     = sow['dt_sow_start'] + timedelta(days=scaled_duration)
        sow['_was_scaled']       = scale_factor < 0.99
    return sow_data

def apply_sow_overruns(sow_data, dt_boundary):
    n_sows         = len(sow_data)
    late_threshold = int(n_sows * 0.6)
    for i, sow in enumerate(sow_data):
        sow_name   = sow['cat_sow']
        is_late    = i >= late_threshold
        was_scaled = sow.get('_was_scaled', False)
        if sow_name in SOW_OVERRUN_PROFILES:
            prob, median_days, p75_days, p90_days = SOW_OVERRUN_PROFILES[sow_name]
            days_from_boundary = (sow['dt_sow_finish'] - dt_boundary).days
            eligible = days_from_boundary > -30 or (was_scaled and is_late)
            if eligible and np.random.rand() < prob:
                overrun_days = max(7, int(np.random.triangular(7, median_days, p75_days)))
                sow['dt_sow_finish']     = dt_boundary + timedelta(days=overrun_days)
                sow['sow_duration_days'] = (sow['dt_sow_finish'] - sow['dt_sow_start']).days
            elif sow['dt_sow_finish'] > dt_boundary:
                if sow['dt_sow_start'] >= dt_boundary:
                    sow['dt_sow_start'] = dt_boundary - timedelta(days=30)
                sow['dt_sow_finish']     = dt_boundary
                sow['sow_duration_days'] = (dt_boundary - sow['dt_sow_start']).days
        else:
            if sow['dt_sow_finish'] > dt_boundary:
                if sow['dt_sow_start'] >= dt_boundary:
                    sow['dt_sow_start'] = dt_boundary - timedelta(days=30)
                sow['dt_sow_finish']     = dt_boundary
                sow['sow_duration_days'] = (dt_boundary - sow['dt_sow_start']).days
    for sow in sow_data:
        sow.pop('_was_scaled', None)
    return sow_data

def estimate_dependency_metrics(sequence_position, n_sows_total):
    normalized_pos = sequence_position / max(n_sows_total, 1)
    if normalized_pos < 0.25:
        avg_pred = 0.5 + np.random.uniform(0, 1.5)
        avg_succ = 2   + np.random.uniform(0, 3)
        pct_crit = 2   + np.random.uniform(0, 4)
    elif normalized_pos < 0.50:
        avg_pred = 1.5 + np.random.uniform(0, 2)
        avg_succ = 1.5 + np.random.uniform(0, 2)
        pct_crit = 3   + np.random.uniform(0, 6)
    elif normalized_pos < 0.75:
        avg_pred = 2   + np.random.uniform(0, 2)
        avg_succ = 0.5 + np.random.uniform(0, 1.5)
        pct_crit = 1.5 + np.random.uniform(0, 3.5)
    else:
        avg_pred = 2.5 + np.random.uniform(0, 2.5)
        avg_succ = 0.2 + np.random.uniform(0, 1)
        pct_crit = 1   + np.random.uniform(0, 3)
    return {
        'avg_predecessors': round(avg_pred, 2),
        'avg_successors':   round(avg_succ, 2),
        'pct_critical_path':round(pct_crit, 2)
    }

def lookup_completion(sbu, progress_bucket, pos_bucket):
    for level_val, sbu_val in [('SBU', sbu), ('GLOBAL', None)]:
        row = completion_index.get((level_val, sbu_val, progress_bucket, pos_bucket))
        if row is not None:
            p25    = float(row['p25_complete'])
            median = float(row['median_complete'])
            p75    = float(row['p75_complete'])
            return safe_triangular(
                max(0.0,   min(p25, median)),
                median,
                min(100.0, max(p75, median))
            )
    return 0.0

def lookup_date_shift(sbu, completion_bucket, position_bucket):
    """sbu kept for API consistency; table is global-only (no cat_sbu column)."""
    row = date_shift_index.get((completion_bucket, position_bucket))
    if row is not None:
        finish_shift = safe_triangular(
            row['p25_finish_shift'],
            row['median_finish_shift'],
            row['p75_finish_shift']
        )
        return {
            'finish_shift':    int(finish_shift),
            'start_shift':     int(row['median_start_shift']),
            'duration_change': int(row['median_duration_change'])
        }
    return {'finish_shift': 0, 'start_shift': 0, 'duration_change': 0}

def sample_task_count_ratio(sbu):
    bounds = hierarchical_lookup_constraint(constraints_df, sbu, None, None, None, 'task_count_change_ratio')
    if not bounds:
        return 1.0
    ratio = sample_from_bounds(bounds)
    return np.clip(ratio, 0.5, 2.0) if ratio else 1.0

# Bucketing helpers
def get_progress_bucket(timing):
    if timing < 0.2:   return '0-20'
    elif timing < 0.4: return '20-40'
    elif timing < 0.6: return '40-60'
    elif timing < 0.8: return '60-80'
    else:              return '80-100'

def get_position_bucket(normalized_pos):
    if normalized_pos < 0.2:   return 'early'
    elif normalized_pos < 0.4: return 'early_mid'
    elif normalized_pos < 0.6: return 'mid'
    elif normalized_pos < 0.8: return 'mid_late'
    else:                      return 'late'

def get_completion_bucket(pct):
    if pct >= 95:   return 'completed'
    elif pct >= 50: return 'mostly_done'
    elif pct >= 10: return 'in_progress'
    else:           return 'not_started'

def get_date_shift_position_bucket(normalized_pos):
    if normalized_pos < 0.25:  return 'early'
    elif normalized_pos < 0.50:return 'early_mid'
    elif normalized_pos < 0.75:return 'mid_late'
    else:                      return 'late'

def build_task_record(project_id, schedule_code, dt_effective_from, dt_effective_to,
                      is_active, sow_dict, sbu, n_sows_total):
    sow          = sow_dict['cat_sow']
    n_tasks      = sow_dict['n_tasks']
    pct_complete = sow_dict.get('pct_complete', 0)

    if pct_complete < 10:    max_active_pct = 0.025
    elif pct_complete < 50:  max_active_pct = 0.21
    elif pct_complete < 90:  max_active_pct = 0.15
    else:                    max_active_pct = 0.02

    n_completed    = int(n_tasks * pct_complete / 100.0)
    n_remaining    = n_tasks - n_completed
    pct_active     = np.random.uniform(0, max_active_pct)
    n_active       = min(int(n_tasks * pct_active), n_remaining)
    n_not_started  = max(0, n_tasks - n_completed - n_active)
    n_early_start  = int(n_active * 0.55)

    deps         = estimate_dependency_metrics(sow_dict['sequence_order'], n_sows_total)
    avg_duration = calculate_avg_task_duration(sow, sbu)

    return {
        'ids_project_number_synth': project_id,
        'cat_schedule_code':        schedule_code,
        'cat_sow':                  sow,
        'dt_effective_from':        pd.Timestamp(dt_effective_from).date(),
        'dt_effective_to':          pd.Timestamp(dt_effective_to).date(),
        'is_active_record':         is_active,
        'sow_sequence_order':       sow_dict['sequence_order'],
        'n_tasks_total':            n_tasks,
        'n_tasks_completed':        n_completed,
        'n_tasks_active':           n_active,
        'n_tasks_not_started':      n_not_started,
        'n_tasks_early_start':      n_early_start,
        'avg_duration_days':        avg_duration,
        'dt_sow_start':             sow_dict['dt_sow_start'].date(),
        'dt_sow_finish':            sow_dict['dt_sow_finish'].date(),
        'sow_duration_days':        sow_dict['sow_duration_days'],
        'avg_n_predecessors':       deps['avg_predecessors'],
        'avg_n_successors':         deps['avg_successors'],
        'pct_critical_path':        deps['pct_critical_path']
    }

def safety_check_sow(sd):
    """Ensure SOW dates and durations are always valid."""
    min_dur = SOW_MIN_DURATIONS.get(sd['cat_sow'], SOW_MIN_DURATION_DEFAULT)
    if sd['dt_sow_finish'] < sd['dt_sow_start']:
        sd['dt_sow_finish']     = sd['dt_sow_start'] + timedelta(days=min_dur)
        sd['sow_duration_days'] = min_dur
    if sd['sow_duration_days'] < min_dur:
        sd['sow_duration_days'] = min_dur
        sd['dt_sow_finish']     = sd['dt_sow_start'] + timedelta(days=min_dur)
    return sd

# Main Generation Loop
print("\nGenerating task summaries with schedule version linkage...")

task_records = []

for idx, project in seed_df.iterrows():
    if idx % 100 == 0:
        print(f"Processing project {idx+1}/{len(seed_df)}...")

    project_id      = project['ids_project_number_synth']
    sbu             = str(project['cat_sbu'])
    size_cat        = str(project['size_category'])
    contract_bucket = str(project['contract_bucket'])
    asset_class     = str(project.get('cat_main_asset_class', 'None'))
    dt_start        = project['dt_construction_start']
    dt_end          = project['dt_construction_end']
    dt_tco          = project.get('dt_tco')
    dt_next_tco     = project.get('dt_next_tco')
    project_duration= project['project_duration_days']

    dt_tco     = pd.Timestamp(dt_tco)     if pd.notna(dt_tco)      else dt_end
    dt_boundary= resolve_project_boundary(dt_tco, dt_next_tco, dt_end)

    # Use pre-grouped dicts — O(1) instead of DataFrame filter
    project_schedules  = schedules_by_project.get(project_id,  pd.DataFrame())
    project_rebaselines= rebaselines_by_project.get(project_id, pd.DataFrame())

    if len(project_schedules) == 0:
        continue

    # VERSION 0: Original Baseline
    n_sows    = sample_n_sows(sbu, size_cat, contract_bucket, asset_class)
    sow_types = sample_sow_types(sbu, n_sows)
    if len(sow_types) == 0:
        continue

    sow_data = []
    for sow in sow_types:
        # Anchor gravity — force milestone SOWs to correct sequence positions
        # without touching distribution sampling for real SOWs
        if sow == '(01) Notice to Proceed (Start Construction)':
            position = -100
        elif sow == '(02) Substantial Completion (TCO or 1st Turnover Milestone)':
            position = 999
        elif sow == '(00) GMP Execution Date':
            position = -50
        else:
            position = sample_sequence_position(sow, sbu)
        sow_data.append({
            'cat_sow': sow,
            'sequence_position_sampled': position
        })
    sow_data = sorted(sow_data, key=lambda x: x['sequence_position_sampled'])
    for i, sd in enumerate(sow_data, start=1):
        sd['sequence_order'] = i

    # Place SOWs using overlap patterns (no per-SOW TCO clipping here)
    first_dur = sample_sow_duration(sow_data[0]['cat_sow'], sbu)
    sow_data[0]['dt_sow_start']      = dt_start
    sow_data[0]['sow_duration_days'] = first_dur
    sow_data[0]['dt_sow_finish']     = dt_start + timedelta(days=first_dur)

    for i in range(1, len(sow_data)):
        prev = sow_data[i-1]
        curr = sow_data[i]
        overlap_pct = get_overlap_pattern(prev['cat_sow'], curr['cat_sow'])
        offset      = int(prev['sow_duration_days'] * (1 - overlap_pct))
        curr_start  = prev['dt_sow_start'] + timedelta(days=max(0, offset))
        # Guard: don't let SOW start more than 60d past boundary
        if curr_start > dt_boundary + timedelta(days=60):
            curr_start = dt_boundary - timedelta(days=60)
        duration = sample_sow_duration(curr['cat_sow'], sbu)
        curr['dt_sow_start']      = curr_start
        curr['sow_duration_days'] = duration
        curr['dt_sow_finish']     = curr_start + timedelta(days=duration)

    # Proportional scaling then SOW-specific overruns
    sow_data = apply_proportional_duration_scaling(sow_data, dt_start, dt_boundary)
    sow_data = apply_sow_overruns(sow_data, dt_boundary)

    # Safety check + initialise task counts and completion
    for sd in sow_data:
        sd = safety_check_sow(sd)
        sd['n_tasks']      = sample_task_count(sd['cat_sow'], sbu)
        sd['n_tasks_base'] = sd['n_tasks']
        sd['pct_complete'] = 0.0

    # Version 0 schedule metadata
    v0_schedule      = project_schedules.iloc[0]
    v0_code          = v0_schedule['cat_schedule_code']
    v0_effective_from= v0_schedule['dt_effective_from']
    is_only_version  = len(project_schedules) == 1

    v0_effective_to  = (
        pd.Timestamp('2099-12-31') if is_only_version
        else project_schedules.iloc[1]['dt_effective_from'] - timedelta(days=1)
    )

    n_sows_total = len(sow_data)
    for sd in sow_data:
        task_records.append(build_task_record(
            project_id, v0_code, v0_effective_from, v0_effective_to,
            is_only_version, sd, sbu, n_sows_total
        ))

    if len(project_rebaselines) == 0:
        continue

    # SUBSEQUENT VERSIONS: Rebaseline Snapshots
    prev_sow_data = [sd.copy() for sd in sow_data]

    for _, rb_row in project_rebaselines.iterrows():
        rb_number  = int(rb_row['rebaseline_number'])
        rb_date    = rb_row['dt_rebaseline_occurred']
        delay_days = int(rb_row.get('delay_days', 0))

        # Match schedule version for this rebaseline number
        rb_version_pattern = f"-{str(rb_number).zfill(3)}."
        matching = project_schedules[
            project_schedules['cat_schedule_code'].str.contains(rb_version_pattern, regex=False)
        ]
        matching_schedule = (
            matching.iloc[0] if len(matching) > 0
            else project_schedules.iloc[min(rb_number, len(project_schedules)-1)]
        )

        rb_schedule_code   = matching_schedule['cat_schedule_code']
        rb_effective_from  = matching_schedule['dt_effective_from']
        is_last_version    = (rb_number == len(project_rebaselines))

        # Determine effective_to
        if is_last_version:
            rb_effective_to = pd.Timestamp('2099-12-31')
        else:
            next_pattern = f"-{str(rb_number+1).zfill(3)}."
            next_schedules = project_schedules[
                project_schedules['cat_schedule_code'].str.contains(next_pattern, regex=False)
            ]
            rb_effective_to = (
                next_schedules.iloc[0]['dt_effective_from'] - timedelta(days=1)
                if len(next_schedules) > 0
                else project_schedules.iloc[min(rb_number+1, len(project_schedules)-1)]['dt_effective_from'] - timedelta(days=1)
            )

        # Safety: effective_to must be >= effective_from
        if rb_effective_to < rb_effective_from:
            rb_effective_to = rb_effective_from

        # Project progress at this rebaseline
        rb_timing = (
            min(1.0, max(0.0, (rb_date - dt_start).days / project_duration))
            if project_duration > 0 else 0.5
        )
        progress_bucket = get_progress_bucket(rb_timing)
        delay_scale     = max(0.2, min(3.0, delay_days / MEDIAN_DELAY_DAYS))

        # Build new version by applying deltas to previous version
        new_sow_data = []
        for sd in prev_sow_data:
            new_sd         = sd.copy()
            n_sows_total   = len(prev_sow_data)
            normalized_pos = sd['sequence_order'] / max(n_sows_total, 1)

            # 1. Completion progression (monotonically increasing)
            pos_bucket   = get_position_bucket(normalized_pos)
            pct_complete = lookup_completion(sbu, progress_bucket, pos_bucket)
            pct_complete = max(pct_complete, sd.get('pct_complete', 0))
            new_sd['pct_complete'] = min(100.0, pct_complete)

            # 2. Date shifts (only for incomplete SOWs)
            if pct_complete < 95:
                comp_bucket      = get_completion_bucket(pct_complete)
                date_pos_bucket  = get_date_shift_position_bucket(normalized_pos)
                shifts           = lookup_date_shift(sbu, comp_bucket, date_pos_bucket)
                actual_finish_shift = int(shifts['finish_shift'] * delay_scale)
                actual_start_shift  = int(shifts['start_shift']  * delay_scale)
                new_sd['dt_sow_finish'] = sd['dt_sow_finish'] + timedelta(days=actual_finish_shift)
                new_sd['dt_sow_start']  = sd['dt_sow_start']  + timedelta(days=actual_start_shift)
                # Recalculate duration and enforce minimum
                raw_duration = (new_sd['dt_sow_finish'] - new_sd['dt_sow_start']).days
                min_dur      = SOW_MIN_DURATIONS.get(new_sd['cat_sow'], SOW_MIN_DURATION_DEFAULT)
                if raw_duration < min_dur:
                    new_sd['dt_sow_finish']     = new_sd['dt_sow_start'] + timedelta(days=min_dur)
                new_sd['sow_duration_days'] = max(min_dur, raw_duration)

            # 3. Task count perturbation (always from V0 base, never compounding)
            ratio          = sample_task_count_ratio(sbu)
            new_sd['n_tasks']      = max(5, int(sd['n_tasks_base'] * ratio))
            new_sd['n_tasks_base'] = sd['n_tasks_base']

            new_sd = safety_check_sow(new_sd)
            new_sow_data.append(new_sd)

        # Write records for this version
        for sd in new_sow_data:
            task_records.append(build_task_record(
                project_id, rb_schedule_code, rb_effective_from, rb_effective_to,
                is_last_version, sd, sbu, len(new_sow_data)
            ))

        prev_sow_data = [sd.copy() for sd in new_sow_data]

# ============================================================
# Final pass: cap extreme drift on ALL versions
# Run once after full generation — not inside project loop
# Ceiling = p90 * jitter(1.0-1.15) for profiled SOWs
# Ceiling = 180d for unprofiled SOWs
# ============================================================
print("Applying final boundary caps...")

for record in task_records:
    dt_boundary_r = boundary_by_project.get(record['ids_project_number_synth'])
    if dt_boundary_r is None:
        continue
    sow_finish = pd.Timestamp(record['dt_sow_finish'])
    if sow_finish > dt_boundary_r:
        sow_name = record['cat_sow']
        p90_days = SOW_OVERRUN_PROFILES.get(sow_name, (None, None, None, 180))[3]
        jitter     = np.random.uniform(1.0, 1.15)
        max_finish = dt_boundary_r + timedelta(days=int(p90_days * jitter))
        if sow_finish > max_finish:
            sow_start = pd.Timestamp(record['dt_sow_start'])
            if sow_start >= max_finish:
                record['dt_sow_start'] = (max_finish - timedelta(days=30)).date()
                sow_start              = max_finish - timedelta(days=30)
            record['dt_sow_finish']     = max_finish.date()
            record['sow_duration_days'] = (max_finish - sow_start).days

print(f"\nGenerated {len(task_records)} task summary records")

# Post-Processing & Validation
print("Post-processing...")

task_df = pd.DataFrame(task_records)
task_df['dt_sow_start']       = pd.to_datetime(task_df['dt_sow_start'])
task_df['dt_sow_finish']      = pd.to_datetime(task_df['dt_sow_finish'])
task_df['dt_effective_from']  = pd.to_datetime(task_df['dt_effective_from'])
task_df['dt_effective_to']    = pd.to_datetime(task_df['dt_effective_to'])

# ── Final DataFrame cleanup: enforce minimum durations on all records ──
# Catches any edge cases that slipped through generation (e.g. extreme
# rebaseline compounding). Guaranteed last line of defence before save.
def enforce_min_duration(row):
    min_dur = SOW_MIN_DURATIONS.get(row['cat_sow'], SOW_MIN_DURATION_DEFAULT)
    if row['sow_duration_days'] < min_dur:
        row['dt_sow_finish']     = row['dt_sow_start'] + timedelta(days=min_dur)
        row['sow_duration_days'] = min_dur
    return row

task_df = task_df.apply(enforce_min_duration, axis=1)

print("\n=== GENERATION VALIDATION ===")
print(f"Total projects:    {task_df['ids_project_number_synth'].nunique()}")
print(f"Total SOW records: {len(task_df)}")

versions_per_project = task_df.groupby('ids_project_number_synth')['cat_schedule_code'].nunique()
print(f"\nSchedule versions per project — min: {versions_per_project.min()}, "
      f"median: {versions_per_project.median():.0f}, max: {versions_per_project.max()}")

sows_per_version = task_df.groupby(['ids_project_number_synth', 'cat_schedule_code']).size()
print(f"SOWs per version   — median: {sows_per_version.median():.0f}, "
      f"p75: {sows_per_version.quantile(0.75):.0f}")

active_df = task_df[task_df['is_active_record'] == True]
print(f"\nSOW durations (active records):")
for sow_name in ['(26) Interior Finishes', '(25) Interior Rough',
                 '(13) Below Grade Structure + Foundations']:
    s = active_df[active_df['cat_sow'] == sow_name]['sow_duration_days']
    if len(s) > 0:
        print(f"  {sow_name}: median={s.median():.0f}d  p25={s.quantile(0.25):.0f}d  p75={s.quantile(0.75):.0f}d")

print(f"\nCritical path % (active): "
      f"median={active_df['pct_critical_path'].median():.2f}%  "
      f"p75={active_df['pct_critical_path'].quantile(0.75):.2f}%")

v0_df = task_df[task_df['cat_schedule_code'].str.contains('-000', na=False)]
if len(v0_df) > 0:
    v0_pct = v0_df['n_tasks_completed'].sum() * 100 / max(v0_df['n_tasks_total'].sum(), 1)
    print(f"\nCompletion — V0 (original baseline): {v0_pct:.1f}% complete")
if len(active_df) > 0:
    act_pct = active_df['n_tasks_completed'].sum() * 100 / max(active_df['n_tasks_total'].sum(), 1)
    print(f"Completion — Active version:          {act_pct:.1f}% complete")

invalid        = task_df[task_df['dt_sow_finish'] < task_df['dt_sow_start']]
invalid_windows= task_df[task_df['dt_effective_to'] < task_df['dt_effective_from']]
active_count   = task_df.groupby('ids_project_number_synth')['is_active_record'].sum()

print(f"\n{'✅' if len(invalid) == 0 else '❌'} SOWs with finish < start: {len(invalid)}")
print(f"{'✅' if len(invalid_windows) == 0 else '❌'} Invalid effective windows: {len(invalid_windows)}")
print(f" Active SOWs per project: median={active_count.median():.0f}")
print("\nSaving output...")

task_spark = spark.createDataFrame(task_df)
task_spark = (
    task_spark
    .withColumn("dt_sow_start",      F.to_date("dt_sow_start"))
    .withColumn("dt_sow_finish",     F.to_date("dt_sow_finish"))
    .withColumn("dt_effective_from", F.to_date("dt_effective_from"))
    .withColumn("dt_effective_to",   F.to_date("dt_effective_to"))
)

(task_spark
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_TABLE))

print(f"Saved {task_spark.count()} records to: {OUT_TABLE}")
print("\n=== GENERATION COMPLETE ===")