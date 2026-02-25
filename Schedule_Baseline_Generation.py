import numpy as np
import pandas as pd
import random
import json
from datetime import timedelta
from pyspark.sql import functions as F

# Configuration
SEED_TABLE = "analysts.self_managed.synthetic_project_meta_data_profiled"
PROFILE_TABLE = "analysts.self_managed.schedule_baseline_profile"
CONSTRAINTS_TABLE = "analysts.self_managed.schedule_baseline_constraints"
OUT_SCHEDULE = "analysts.self_managed.synthetic_schedule_baseline_v1"
OUT_METADATA = "analysts.self_managed.rebaseline_event_metadata_v1"

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Structural business rules (constants)
MIN_NTP_TO_TCO_DAYS = 30      # Minimum project duration
MIN_TCO_TO_CO_DAYS = 14       # Minimum close-out period
MIN_REBASELINE_DELAY = 7      # Minimum meaningful delay (< 1 week is noise)
SLACK_MULTIPLIER = 0.9        # Conservative: delays ≤ 90% of remaining slack
SLACK_MARGIN = 0.20           # Allow 20% exceedance of slack-based cap
P95_MARGIN = 0.20             # Soft cap margin (can exceed p95 by 20%)
EFFECTIVE_FROM_OFFSET = 90    # Days before NTP for baseline creation

# Global delay category distribution (from empirical analysis)
# Probabilities normalized to sum to exactly 1.0 based on analysis
DELAY_CATEGORY_PMF = {
    'minimal': 0.093,
    'small': 0.313,      # Modal category
    'moderate': 0.232,
    'major': 0.179,
    'large': 0.081,
    'severe': 0.102      
}

# Load Profiles and Constraints
print("Loading profiles and constraints...")

# Load profile table to Pandas
profiles_df = spark.read.table(PROFILE_TABLE).toPandas()

# Load constraints table to Pandas
constraints_df = spark.read.table(CONSTRAINTS_TABLE).toPandas()

# Parse JSON blobs in constraints
def parse_json_bounds(json_str):
    """Parse JSON string to dict, handle nulls"""
    if pd.isna(json_str):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}

constraints_df['bounds'] = constraints_df['bounds_json'].apply(parse_json_bounds)

print(f"Loaded {len(profiles_df)} profile rows, {len(constraints_df)} constraint rows")

# Load Seed Projects
print("Loading synthetic seed projects...")

seed_df = spark.read.table(SEED_TABLE).select(
    "ids_project_number_synth",
    "cat_sbu",
    "cat_region",
    "cat_main_asset_class",
    "gross_area",
    "amt_contract",
    "dt_construction_start",
    "dt_construction_end"
).toPandas()

# Create size buckets (matching profiling)
seed_df['size_category'] = pd.cut(
    seed_df['gross_area'],
    bins=[0, 50000, 100000, 200000, np.inf],
    labels=['Small', 'Medium', 'Large', 'XLarge']
).astype(str)

# Create contract buckets (matching profiling)
seed_df['contract_bucket'] = pd.cut(
    seed_df['amt_contract'],
    bins=[0, 50000000, 100000000, 200000000, np.inf],
    labels=['Medium', 'Large', 'XLarge', 'XXLarge']
).astype(str)

# Ensure dates are datetime
seed_df['dt_construction_start'] = pd.to_datetime(seed_df['dt_construction_start'])
seed_df['dt_construction_end'] = pd.to_datetime(seed_df['dt_construction_end'])

# Calculate project duration
seed_df['project_duration_days'] = (
    seed_df['dt_construction_end'] - seed_df['dt_construction_start']
).dt.days

print(f"Loaded {len(seed_df)} synthetic projects")

# Helper Functions
def hierarchical_lookup(profiles_df, level_priority, sbu, size_cat, contract_bucket, asset_class, field, min_n=5):
    """
    Lookup profile with hierarchical fallback.
    Returns: row from profiles_df or None
    """
    # Try levels in priority order
    lookups = [
        ('SBU_CONTRACT', {'cat_sbu': sbu, 'contract_bucket': contract_bucket}),
        ('SBU_SIZE', {'cat_sbu': sbu, 'size_category': size_cat}),
        ('SBU_ASSET', {'cat_sbu': sbu, 'cat_main_asset_class': asset_class}),
        ('SBU', {'cat_sbu': sbu}),
        ('GLOBAL', {})
    ]
    
    for level, filters in lookups:
        mask = (profiles_df['level'] == level) & (profiles_df['field'] == field)
        for col, val in filters.items():
            if val is not None and val != 'nan':
                mask &= (profiles_df[col] == val)
            else:
                mask &= (profiles_df[col].isna() | (profiles_df[col] == 'None'))
        
        result = profiles_df[mask]
        if len(result) > 0:
            row = result.iloc[0]
            if pd.notna(row.get('n_projects', 0)) and row.get('n_projects', 0) >= min_n:
                return row
            elif level in ['SBU', 'GLOBAL']:  # Always use SBU/Global if found
                return row
    
    return None

def hierarchical_lookup_constraint(constraints_df, sbu, size_cat, contract_bucket, field):
    """
    Lookup constraint with hierarchical fallback.
    Returns: bounds dict
    """
    lookups = [
        ('SBU_CONTRACT', {'cat_sbu': sbu, 'contract_bucket': contract_bucket}),
        ('SBU_SIZE', {'cat_sbu': sbu, 'size_category': size_cat}),
        ('SBU', {'cat_sbu': sbu}),
        ('GLOBAL', {}),
        ('TIMING', {})  # For timing fields
    ]
    
    for level, filters in lookups:
        mask = (constraints_df['level'] == level) & (constraints_df['field'] == field)
        for col, val in filters.items():
            if val is not None and val != 'nan':
                mask &= (constraints_df[col] == val)
            else:
                mask &= (constraints_df[col].isna() | (constraints_df[col] == 'None'))
        
        result = constraints_df[mask]
        if len(result) > 0:
            return result.iloc[0]['bounds']
    
    return {}

def sample_from_bounds(bounds, use_triangular=True):
    """Sample from bounds dict with triangular distribution"""
    if not bounds:
        return None
    
    p25 = bounds.get('p25', bounds.get('p50', 0))
    p50 = bounds.get('p50', bounds.get('median', 0))
    p75 = bounds.get('p75', bounds.get('p50', 0))
    
    if use_triangular:
        return np.random.triangular(left=p25, mode=p50, right=p75)
    else:
        return np.random.uniform(p25, p75)

def soft_bound(val, p95, p99, margin=0.20, jitter=0.06):
    """
    Apply soft bounding with p95 soft cap and p99 hard cap.
    Similar to Stage 1 soft bounding pattern.
    """
    if val is None or not np.isfinite(val):
        return val
    
    # If exceeds p95, nudge toward p95 with margin allowance
    if val > p95:
        val = p95 * (1 + np.random.uniform(0, margin))
    else:
        # Within normal range, add jitter
        val = val * (1 + np.random.uniform(-jitter, jitter))
    
    # Hard cap at p99
    val = min(val, p99)
    
    return val

def sample_rebaseline_count(sbu, size_cat, contract_bucket, asset_class):
    """
    Sample number of rebaselines for a project.
    Uses Poisson distribution to better preserve frequency of 0 rebaselines.
    """
    row = hierarchical_lookup(
        profiles_df, None, sbu, size_cat, contract_bucket, asset_class, 'n_rebaselines'
    )
    
    if row is None:
        return 0  # Fallback: no rebaselines
    
    # Use mean for Poisson parameter (better preserves 0-count frequency)
    mean = row.get('mean', 0)
    if mean <= 0:
        return 0
    
    # Sample from Poisson distribution
    n = np.random.poisson(mean)
    
    # Cap at p99 if available (exclude extreme outliers)
    p99 = row.get('p99')
    if pd.notna(p99):
        n = min(n, int(p99))
    
    # Also cap at max if p99 not available
    max_val = row.get('max')
    if pd.notna(max_val):
        n = min(n, int(max_val))
    
    return max(0, n)

def sample_delay_category():
    """Sample delay category from global distribution"""
    categories = list(DELAY_CATEGORY_PMF.keys())
    probs = list(DELAY_CATEGORY_PMF.values())
    
    # Normalize to ensure sum = 1.0 (handle floating point precision)
    probs = np.array(probs)
    probs = probs / probs.sum()
    
    return np.random.choice(categories, p=probs)

def sample_delay_magnitude(sbu, size_cat, contract_bucket, category, slack_remaining):
    """
    Sample delay magnitude with slack-based conditional capping.
    Applies soft p95 cap and hard p99 cap.
    """
    field = f'delay_{category}'
    bounds = hierarchical_lookup_constraint(constraints_df, sbu, size_cat, contract_bucket, field)
    
    if not bounds:
        # Fallback: use category midpoints
        category_defaults = {
            'minimal': 4, 'small': 17, 'moderate': 41,
            'major': 81, 'large': 149, 'severe': 275
        }
        return category_defaults.get(category, 30)
    
    # Sample raw delay from distribution
    delay_raw = sample_from_bounds(bounds, use_triangular=True)
    
    # Get p95 and p99 for capping
    p95 = bounds.get('p95', bounds.get('p75', delay_raw))
    p99 = bounds.get('p99', bounds.get('max', delay_raw))
    
    # Apply slack-based cap (conditional on remaining schedule)
    if slack_remaining is not None and slack_remaining > 0:
        slack_based_cap = slack_remaining * SLACK_MULTIPLIER
        
        # If delay exceeds slack cap, apply soft constraint
        if delay_raw > slack_based_cap:
            # Allow exceedance with margin
            delay_raw = slack_based_cap * (1 + np.random.uniform(0, SLACK_MARGIN))
    
    # Apply soft p95 cap with margin, hard p99 cap
    delay_final = soft_bound(delay_raw, p95, p99, margin=P95_MARGIN)
    
    # Ensure minimum delay (filter noise)
    delay_final = max(delay_final, MIN_REBASELINE_DELAY)
    
    return int(delay_final)

def sample_timing(rebaseline_number):
    """Sample when rebaseline occurs (normalized 0-1)"""
    field = f'timing_rebaseline_{min(rebaseline_number, 5)}'  # Cap at 5th
    bounds = hierarchical_lookup_constraint(constraints_df, None, None, None, field)
    
    if not bounds:
        # Fallback: reasonable defaults
        defaults = {1: 0.49, 2: 0.68, 3: 0.74, 4: 0.70, 5: 0.70}
        return defaults.get(rebaseline_number, 0.50)
    
    # Sample from timing distribution
    timing = sample_from_bounds(bounds, use_triangular=True)
    
    # Ensure within [0, 1] range
    return np.clip(timing, 0.0, 1.0)

def sample_milestone_offset(sbu, size_cat, contract_bucket, asset_class, field, min_value):
    """Sample milestone offset (NTP→TCO, TCO→CO) with floor constraint"""
    row = hierarchical_lookup(
        profiles_df, None, sbu, size_cat, contract_bucket, asset_class, field
    )
    
    if row is None:
        return min_value  # Fallback to minimum
    
    # Sample from distribution
    p25 = row.get('p25', min_value)
    median = row.get('median', min_value)
    p75 = row.get('p75', min_value)
    
    offset = np.random.triangular(p25, median, p75)
    
    # Apply floor constraint (structural rule)
    offset = max(offset, min_value)
    
    # Add jitter to prevent clustering
    offset = offset * (1 + np.random.uniform(-0.06, 0.06))
    
    return int(offset)

def sample_next_tco_offset(sbu, size_cat, contract_bucket, asset_class):
    """
    Sample dt_next_tco offset (can be negative or positive).
    NOTE: Call once per project, not per baseline.
    """
    
    row = hierarchical_lookup(
        profiles_df, None, sbu, size_cat, contract_bucket, asset_class, 'tco_to_next_tco_days'
    )
    
    if row is None:
        return 0  # Default: no offset
    
    median = row.get('median', 0)
    p25 = row.get('p25', 0)
    p75 = row.get('p75', 0)
    p95 = row.get('p95', p75)
    p99 = row.get('p99', p95)
    
    # Handle degenerate case
    if p25 == median == p75:
        offset = median
    else:
        if p25 == median:
            p25 = median - 5
        if p75 == median:
            p75 = median + 5
        offset = np.random.triangular(p25, median, p75)
    
    # Soft bound positive offsets only
    if offset > 0:
        if pd.notna(p95) and offset > p95:
            offset = p95 * (1 + np.random.uniform(0, 0.20))
        if pd.notna(p99):
            offset = min(offset, p99)
    
    offset = offset * (1 + np.random.uniform(-0.06, 0.06))
    return int(offset)

# Main Generation Loop
print("Generating synthetic schedule baselines...")

schedule_records = []
metadata_records = []

for idx, project in seed_df.iterrows():
    if idx % 100 == 0:
        print(f"Processing project {idx+1}/{len(seed_df)}...")
    
    # Extract project attributes
    project_id = project['ids_project_number_synth']
    sbu = str(project['cat_sbu'])
    region = str(project['cat_region'])
    asset_class = str(project.get('cat_main_asset_class', 'None'))
    size_cat = str(project['size_category'])
    contract_bucket = str(project['contract_bucket'])
    
    dt_start = project['dt_construction_start']
    dt_end_original = project['dt_construction_end']
    project_duration = project['project_duration_days']
    
    # Sample rebaseline count (hierarchical lookup)
    n_rebaselines = sample_rebaseline_count(sbu, size_cat, contract_bucket, asset_class)

    # Decide at project level whether to track dt_next_tco
    project_has_next_tco = (np.random.rand() < 0.35)  # 35% of projects
    if project_has_next_tco:
        base_next_tco_offset = sample_next_tco_offset(sbu, size_cat, contract_bucket, asset_class)
    else:
        base_next_tco_offset = None
    
    # Generate original baseline (version 0)
    dt_ntp = dt_start
    dt_tco_original = dt_end_original
    
    # Sample TCO→CO offset
    tco_to_co_offset = sample_milestone_offset(
        sbu, size_cat, contract_bucket, asset_class, 
        'tco_to_co_days', MIN_TCO_TO_CO_DAYS
    )
    dt_co = dt_tco_original + timedelta(days=tco_to_co_offset)
    dt_next_tco = (dt_tco_original + timedelta(days=base_next_tco_offset)) if project_has_next_tco else None
    
    # Create original baseline record
    baseline_v0 = {
        'ids_project_number_synth': project_id,
        'cat_schedule_code': f"{project_id}-000.01-BL",
        'dt_ntp': dt_ntp.date(),
        'dt_tco': dt_tco_original.date(),
        'dt_next_tco': dt_next_tco.date() if dt_next_tco else None,
        'dt_co': dt_co.date(),
        'n_recent_schedule': n_rebaselines + 1,  # Countdown
        'dt_effective_from': (dt_ntp - timedelta(days=EFFECTIVE_FROM_OFFSET)).date(),
        'dt_effective_to': None,  # Will set later
        'is_active_record': (n_rebaselines == 0),
        'is_first_record_for_new_snapshot': True
    }

    baselines = [baseline_v0]
    cumulative_tco = dt_tco_original  # Track current TCO through rebaselines
    # This prevents the "ratchet effect" where max() pushes all to 1.0
    if n_rebaselines > 0:
        # Sample raw timings from profiled distributions
        raw_timings = [sample_timing(i) for i in range(1, n_rebaselines + 1)]
        
        # Sort to ensure chronological order within this project
        sorted_timings = sorted(raw_timings)
        
        # Ensure minimum separation between consecutive rebaselines
        # 5% gap prevents bunching and ensures effective_to > effective_from
        MIN_TIMING_GAP = 0.05
        MAX_TIMING = 0.95  # Leave room for project completion
        
        adjusted_timings = []
        last_timing_used = 0.0
        
        for timing in sorted_timings:
            # Ensure at least MIN_TIMING_GAP after previous rebaseline
            adjusted = max(timing, last_timing_used + MIN_TIMING_GAP)
            # Cap at MAX_TIMING to leave room at end
            adjusted = min(adjusted, MAX_TIMING)
            adjusted_timings.append(adjusted)
            last_timing_used = adjusted
        
        # Track last rebaseline date for effective_to validation
        last_rebaseline_date = None
        
        # Generate rebaselines using adjusted timings
        for i, normalized_timing in enumerate(adjusted_timings, start=1):
            # Calculate rebaseline date from adjusted timing
            days_from_start = int(project_duration * normalized_timing)
            dt_rebaseline = dt_start + timedelta(days=days_from_start)
            
            # Safety check: ensure this rebaseline is after previous
            # (should always be true with sorted timings, but defensive)
            if last_rebaseline_date is not None and dt_rebaseline <= last_rebaseline_date:
                dt_rebaseline = last_rebaseline_date + timedelta(days=1)
            
            # Calculate remaining slack (from rebaseline date to current TCO)
            slack_remaining = (cumulative_tco - dt_rebaseline).days
            
            # Ensure positive slack (if negative, rebaseline already past TCO)
            if slack_remaining <= 0:
                slack_remaining = 30  # Minimum slack for late rebaselines
            
            # Sample delay category
            category = sample_delay_category()
            
            # Sample delay magnitude (conditional on slack)
            delay_days = sample_delay_magnitude(
                sbu, size_cat, contract_bucket, category, slack_remaining
            )
            
            # Apply delay to current TCO
            cumulative_tco = cumulative_tco + timedelta(days=delay_days)
            
            # Enforce monotonic TCO (structural rule: TCO never decreases)
            prev_tco = pd.Timestamp(baselines[-1]['dt_tco'])
            cumulative_tco = max(cumulative_tco, prev_tco)
            
            # Update CO offset (extends with TCO)
            dt_co = cumulative_tco + timedelta(days=tco_to_co_offset)
            
            dt_next_tco = (cumulative_tco + timedelta(days=base_next_tco_offset)) if project_has_next_tco else None
            
            # Create rebaseline record
            baseline_vi = {
                'ids_project_number_synth': project_id,
                'cat_schedule_code': f"{project_id}-{str(i).zfill(3)}.MO-RB",
                'dt_ntp': dt_ntp.date(),  # NTP doesn't change
                'dt_tco': cumulative_tco.date(),
                'dt_next_tco': dt_next_tco.date() if dt_next_tco else None,
                'dt_co': dt_co.date(),
                'n_recent_schedule': n_rebaselines - i + 1,  # Countdown
                'dt_effective_from': dt_rebaseline.date(),
                'dt_effective_to': None,  # Will set later
                'is_active_record': (i == n_rebaselines),  # Active if last
                'is_first_record_for_new_snapshot': True
            }
            
            # Update previous baseline's effective_to
            prev_effective_from = pd.Timestamp(baselines[-1]['dt_effective_from'])
            new_effective_to = dt_rebaseline - timedelta(days=1)
            
            # Safety: if new_effective_to would be before prev_effective_from, 
            # set to same day (single-day window)
            if new_effective_to < prev_effective_from:
                new_effective_to = prev_effective_from
            
            baselines[-1]['dt_effective_to'] = new_effective_to.date()
            baselines[-1]['is_active_record'] = False
            
            baselines.append(baseline_vi)
            last_rebaseline_date = dt_rebaseline
            
            # Create rebaseline metadata
            metadata_records.append({
                'ids_project_number_synth': project_id,
                'rebaseline_number': i,
                'dt_rebaseline_occurred': dt_rebaseline.date(),
                'dt_tco_original': dt_tco_original.date(),
                'dt_tco_revised': cumulative_tco.date(),
                'delay_days': delay_days,
                'delay_magnitude_category': category,
                'primary_delayed_sow': None,  # Will link for task table
                'n_sows_impacted': None       # Will calculate during task table gen
            })
    
    # Set final baseline's effective_to
    baselines[-1]['dt_effective_to'] = pd.Timestamp('2099-12-31').date()
    
    # Add all baselines for this project
    schedule_records.extend(baselines)

print(f"Generated {len(schedule_records)} schedule baseline records")
print(f"Generated {len(metadata_records)} rebaseline metadata records")

# Post-Processing and Validation
print("Post-processing...")

# Convert to DataFrames
schedule_df = pd.DataFrame(schedule_records)
metadata_df = pd.DataFrame(metadata_records) if metadata_records else pd.DataFrame()

# Ensure date types
date_cols = ['dt_ntp', 'dt_tco', 'dt_next_tco', 'dt_co', 'dt_effective_from', 'dt_effective_to']
for col in date_cols:
    if col in schedule_df.columns:
        schedule_df[col] = pd.to_datetime(schedule_df[col])

if not metadata_df.empty:
    meta_date_cols = ['dt_rebaseline_occurred', 'dt_tco_original', 'dt_tco_revised']
    for col in meta_date_cols:
        metadata_df[col] = pd.to_datetime(metadata_df[col])

# Basic validation
print("\n=== GENERATION VALIDATION ===")
print(f"Total projects: {schedule_df['ids_project_number_synth'].nunique()}")
print(f"Total schedule records: {len(schedule_df)}")

# Count projects by number of versions
versions_per_project = schedule_df.groupby('ids_project_number_synth').size()
print(f"Projects with 0 rebaselines: {(versions_per_project == 1).sum()}")
print(f"Projects with 1+ rebaselines: {(versions_per_project > 1).sum()}")
print(f"Average records per project: {len(schedule_df) / schedule_df['ids_project_number_synth'].nunique():.2f}")

# Check for nulls in required fields
required_fields = ['ids_project_number_synth', 'cat_schedule_code', 'dt_ntp', 'dt_tco', 
                   'dt_effective_from', 'dt_effective_to', 'is_active_record', 'n_recent_schedule']
null_counts = schedule_df[required_fields].isnull().sum()
if null_counts.sum() > 0:
    print("\nWARNING: Nulls found in required fields:")
    print(null_counts[null_counts > 0])

# Check monotonic TCO dates per project
def check_monotonic_tco(group):
    tcos = pd.to_datetime(group.sort_values('dt_effective_from')['dt_tco'])
    return (tcos.diff().dropna() >= timedelta(days=0)).all()

monotonic_check = schedule_df.groupby('ids_project_number_synth').apply(check_monotonic_tco)
violations = (~monotonic_check).sum()
if violations > 0:
    print(f"\nWARNING: {violations} projects have non-monotonic TCO dates")

# Save Outputs
print("\nSaving outputs...")

# Convert to Spark DataFrames
schedule_spark = spark.createDataFrame(schedule_df)
# Cast dates explicitly
for col in date_cols:
    if col in schedule_spark.columns:
        schedule_spark = schedule_spark.withColumn(col, F.to_date(col))

# Save schedule baseline
(schedule_spark
 .write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(OUT_SCHEDULE))

print(f"Saved {schedule_spark.count()} schedule baseline records to: {OUT_SCHEDULE}")

# Save rebaseline metadata
if not metadata_df.empty:
    metadata_spark = spark.createDataFrame(metadata_df)
    for col in meta_date_cols:
        metadata_spark = metadata_spark.withColumn(col, F.to_date(col))
    
    (metadata_spark
     .write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(OUT_METADATA))
    
    print(f"Saved {metadata_spark.count()} rebaseline metadata records to: {OUT_METADATA}")
else:
    print("No rebaseline metadata to save (all projects had 0 rebaselines)")

print("\n=== GENERATION COMPLETE ===")
print(f"Schedule baseline table: {OUT_SCHEDULE}")
print(f"Rebaseline metadata table: {OUT_METADATA}")