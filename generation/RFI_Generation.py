from databricks.sdk.runtime import dbutils
import anthropic
import asyncio
import json
import math
import random
import re
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# Anthropic clients
_api_key     = dbutils.secrets.get(scope="llm-access-tokens", key="anthropic-api-key")
async_client = anthropic.AsyncAnthropic(api_key=_api_key)

# Configuration
SYNTHETIC_PROJECT_TABLE = "analysts.self_managed.synthetic_project_meta_data_profiled"
TASK_SUMMARY_TABLE      = "analysts.self_managed.synthetic_task_summary_by_sow_v1"
REBASELINE_TABLE        = "analysts.self_managed.rebaseline_event_metadata_v1"
RFI_PROFILE_TABLE       = "analysts.self_managed.rfi_profile"
RFI_CONSTRAINTS_TABLE   = "analysts.self_managed.rfi_constraints"
OUT_RFI_TABLE           = "analysts.self_managed.synthetic_rfi_v1"

SEED             = 42
CHECKPOINT_EVERY = 1000

LLM_MODEL       = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS  = 4096
MAX_LLM_RETRIES = 3

BATCH_SIZE           = 20
MIN_BATCH_GROUP      = 5
MAX_CONCURRENT_CALLS = 10
MAX_WORKERS          = 3
INTER_PROJECT_SLEEP  = 1.0
INTER_BATCH_SLEEP    = 0.1

RFI_COUNT_SOFT_MARGIN  = 0.15
RESOLUTION_SOFT_MARGIN = 0.15

# Due date distribution — Suffolk standard is 7 days but real data shows variation
DUE_DATE_OPTIONS = [5, 7, 7, 7, 10, 14, 21]  # weighted toward 7

random.seed(SEED)
np.random.seed(SEED)

# Trade → SOW mapping
TRADE_TO_SOW = {
    'A SUBSTRUCTURE': ['(13) Below Grade Structure', '(13) Foundations'],
    'B SHELL':        ['(14) Superstructure', '(17) Facade', '(18) Roofing'],
    'C INTERIORS':    ['(25) Interior Rough', '(26) Interior Finishes', '(27) Worklist'],
    'D SERVICES':     ['(29) Fire Life Safety', '(30) Systems Start-up', '(08) Permanent Power'],
    'E EQUIPMENT AND FURNISHINGS':           ['(24) Remaining Elevator', '(21) Hoist'],
    'F SPECIAL CONSTRUCTION AND DEMOLITION': ['(12) Initial Site Work', '(12C) Demo & Abatement'],
    'G SITEWORK':     ['(28) Final Site Work', '(12) Initial Site Work'],
    'Z GENERAL':      ['(10) All Submittals', '(33) NO CODE', '(09) Buys'],
}

# Load reference data
print("Loading reference data...")

profile_df     = spark.read.table(RFI_PROFILE_TABLE).toPandas()
constraints_df = spark.read.table(RFI_CONSTRAINTS_TABLE).toPandas()

def parse_bounds(j):
    if pd.isna(j): return {}
    try:    return json.loads(j)
    except: return {}

constraints_df['bounds'] = constraints_df['bounds_json'].apply(parse_bounds)

_lc  = profile_df[profile_df['field'] == 'lifecycle_timing_pct'].set_index('cat_trade_l1')
_di  = profile_df[profile_df['field'] == 'design_issue_pct'].set_index('cat_trade_l1')
_tw  = profile_df[profile_df['field'] == 'trade_weight']
_st  = profile_df[profile_df['field'] == 'status_distribution']
_stz = profile_df[profile_df['field'] == 'sow_timing_zone_pct']
_l3  = profile_df[profile_df['field'] == 'l3_frequency']

_rb  = constraints_df[constraints_df['field'] == 'rebaseline_spike_ratio']
_rb_bounds = _rb['bounds'].iloc[0] if len(_rb) > 0 else {
    'p25': 0.79, 'median': 1.08, 'p75': 1.58
}

_res = constraints_df[
    (constraints_df['field'] == 'resolution_days') &
    (constraints_df['level'] == 'GLOBAL')
]
_res_bounds = _res['bounds'].iloc[0] if len(_res) > 0 else {
    'p25': 4, 'median': 10, 'p75': 21, 'p90': 49
}

_fs_raw = constraints_df[constraints_df['field'] == 'few_shot_example']
few_shot_pool = defaultdict(list)
for _, row in _fs_raw.iterrows():
    key = (row['cat_trade_l1'], row['cat_sbu'])
    b = row['bounds']
    if b.get('lesson'):
        few_shot_pool[key].append(b)

print(f"Reference data loaded. Few-shot pool: {len(few_shot_pool)} trade×SBU cells")

# Load synthetic project + schedule context
print("Loading synthetic projects and schedule context...")

_raw = spark.read.table(SYNTHETIC_PROJECT_TABLE)

try:
    _done = spark.read.table(OUT_RFI_TABLE)
    SKIP_PROJECTS = {
        r["ids_project_number_synth"]
        for r in _done.select("ids_project_number_synth").distinct().collect()
    }
    print(f"Skipping {len(SKIP_PROJECTS)} already-generated projects")
except Exception:
    SKIP_PROJECTS = set()
    print("Output table not found — starting fresh")

projects_spark = (
    _raw
    .filter(~F.col("ids_project_number_synth").isin(SKIP_PROJECTS))
    .withColumnRenamed("gross_area", "amt_area_gross")
    .select(
        "ids_project_number_synth", "cat_sbu", "cat_main_asset_class",
        "cat_region", "amt_area_gross", "amt_contract",
        "dt_construction_start", "dt_construction_end"
    )
    .orderBy("ids_project_number_synth")
)

projects_pd = projects_spark.toPandas()
assert len(projects_pd) > 0, "No projects to generate — all already done or table missing."

projects_pd['dt_construction_start'] = pd.to_datetime(projects_pd['dt_construction_start'])
projects_pd['dt_construction_end']   = pd.to_datetime(projects_pd['dt_construction_end'])
projects_pd['duration_days'] = (
    projects_pd['dt_construction_end'] - projects_pd['dt_construction_start']
).dt.days

sow_pd = (
    spark.read.table(TASK_SUMMARY_TABLE)
    .select("ids_project_number_synth", "cat_sow", "dt_sow_start", "dt_sow_finish")
    .toPandas()
)
sow_pd['dt_sow_start']  = pd.to_datetime(sow_pd['dt_sow_start'])
sow_pd['dt_sow_finish'] = pd.to_datetime(sow_pd['dt_sow_finish'])

rebaseline_pd = (
    spark.read.table(REBASELINE_TABLE)
    .select("ids_project_number_synth", "rebaseline_number",
            "dt_rebaseline_occurred", "delay_magnitude_category",
            "primary_delayed_sow")
    .toPandas()
)
rebaseline_pd['dt_rebaseline_occurred'] = pd.to_datetime(
    rebaseline_pd['dt_rebaseline_occurred']
)

def get_size_category(area):
    if   area < 50000:  return "Small"
    elif area < 100000: return "Medium"
    elif area < 200000: return "Large"
    else:               return "XLarge"

print(f"Loaded {len(projects_pd)} projects to generate")

# Sampling helpers

def get_rfi_count_bounds(sbu, size_cat):
    for level, filters in [
        ('SBU_SIZE', {'cat_sbu': sbu, 'size_category': size_cat}),
        ('SBU',      {'cat_sbu': sbu}),
        ('GLOBAL',   {}),
    ]:
        rows = constraints_df[constraints_df['field'] == 'n_rfis_per_project']
        rows = rows[rows['level'] == level]
        for col, val in filters.items():
            rows = rows[rows[col] == val]
        if len(rows) > 0:
            return rows.iloc[0]['bounds']
    return {'p25': 50, 'median': 200, 'p75': 500, 'p90': 1000}

def sample_n_rfis(sbu, size_cat):
    b        = get_rfi_count_bounds(sbu, size_cat)
    p25      = float(b.get('p25',    50))
    median   = float(b.get('median', 200))
    p75      = float(b.get('p75',    500))
    hard_cap = int(p75 * (1 + RFI_COUNT_SOFT_MARGIN))
    sigma    = (math.log(p75 + 1) - math.log(p25 + 1)) / (2 * 0.6745)
    mu       = math.log(median + 1)
    return max(1, min(int(np.random.lognormal(mu, sigma)), hard_cap))

def get_trade_weights(sbu):
    rows = _tw[(_tw['cat_sbu'] == sbu) & (_tw['level'] == 'SBU')]
    if len(rows) == 0:
        rows = _tw[_tw['level'] == 'GLOBAL']
    w     = dict(zip(rows['cat_trade_l1'], rows['pct_of_sbu'].astype(float)))
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}

def sample_trade(sbu):
    w = get_trade_weights(sbu)
    return np.random.choice(list(w.keys()), p=list(w.values()))

def sample_lifecycle_pct(trade):
    if trade in _lc.index:
        row  = _lc.loc[trade]
        p10, p90  = float(row['p10']), float(row['p90'])
        med, p25, p75 = float(row['median']), float(row['p25']), float(row['p75'])
        hard_cap = p90 * 1.10
        span     = max(hard_cap - p10, 1)
        med_n    = float(np.clip((med - p10) / span, 0.01, 0.99))
        p25_n    = (p25 - p10) / span
        p75_n    = (p75 - p10) / span
        iqr_n    = max(p75_n - p25_n, 0.05)
        concentration = min(0.5 / (iqr_n ** 2), 50.0)
        alpha    = max(med_n * concentration, 0.5)
        beta_    = max((1 - med_n) * concentration, 0.5)
        sample_n = float(np.random.beta(alpha, beta_))
        return float(np.clip(p10 + sample_n * span, p10, hard_cap))
    return float(np.clip(np.random.beta(2, 2) * 165, 0, 165))

def get_sow_timing_zone_probs(trade):
    rows = _stz[_stz['cat_trade_l1'] == trade]
    if len(rows) == 0:
        return {'pre_sow': 0.20, 'during_sow': 0.70, 'post_sow': 0.10}
    p     = dict(zip(rows['status_simplified'], rows['pct_of_trade'].astype(float)))
    total = sum(p.values())
    return {k: v / total for k, v in p.items()}

def sample_sow_timing_zone(trade):
    p = get_sow_timing_zone_probs(trade)
    return np.random.choice(list(p.keys()), p=list(p.values()))

def assign_dt_rfi_created(proj, lc_pct, sow_row, zone):
    start, end, dur = proj['dt_construction_start'], proj['dt_construction_end'], proj['duration_days']
    lo = hi = None
    if sow_row is not None and not pd.isna(sow_row['dt_sow_start']):
        sow_s, sow_f = sow_row['dt_sow_start'], sow_row['dt_sow_finish']
        if zone == 'pre_sow':
            window = max(14, min(60, int((sow_s - start).days * 0.15)))
            lo, hi = sow_s - timedelta(days=window), sow_s - timedelta(days=1)
        elif zone == 'post_sow':
            lo, hi = sow_f + timedelta(days=1), sow_f + timedelta(days=30)
        else:
            lo, hi = sow_s, sow_f
        lo, hi = max(lo, start), min(hi, end)
        if lo > hi: lo = hi = None
    if lo is None:
        offset = timedelta(days=int(lc_pct / 100 * dur))
        dt = start + offset
        if dt < start: dt = start
        if dt > end:   dt = end
        return pd.Timestamp(dt)
    spread = max(0, (hi - lo).days)
    dt = lo + timedelta(days=random.randint(0, spread))
    if dt < start: dt = start
    if dt > end:   dt = end
    return pd.Timestamp(dt)

def sample_status(sbu):
    rows = _st[(_st['cat_sbu'] == sbu) & (_st['level'] == 'SBU')]
    if len(rows) == 0:
        return np.random.choice(['closed', 'open', 'draft'], p=[0.92, 0.06, 0.02])
    statuses = rows['status_simplified'].tolist()
    probs    = (rows['pct'].astype(float) / rows['pct'].astype(float).sum()).tolist()
    return np.random.choice(statuses, p=probs)

def sample_resolution_days():
    b        = _res_bounds
    p25, med = float(b.get('p25', 4)),  float(b.get('median', 10))
    p75, p90 = float(b.get('p75', 21)), float(b.get('p90',    49))
    hard_cap = int(p90 * (1 + RESOLUTION_SOFT_MARGIN))
    sigma    = (math.log(p75 + 1) - math.log(p25 + 1)) / (2 * 0.6745)
    mu       = math.log(med + 1)
    return max(1, min(int(np.random.lognormal(mu, sigma)), hard_cap))

def get_design_issue_pct(trade):
    if trade in _di.index:
        return float(_di.loc[trade]['pct_design_issue']) / 100.0
    return 0.35

def sample_sow_for_trade(trade, project_sows):
    candidates = TRADE_TO_SOW.get(trade, ['(33) NO CODE'])
    available  = [s for s in candidates if s in project_sows]
    if available: return random.choice(available)
    if project_sows: return random.choice(list(project_sows))
    return random.choice(candidates)

def is_near_rebaseline(dt_created, rebaselines, window=30):
    for _, rb in rebaselines.iterrows():
        delta = (dt_created - rb['dt_rebaseline_occurred']).days
        if 0 <= delta <= window:
            return True, rb
    return False, None

def clean_record(m, drop_cols):
    m = dict(m)
    for col in drop_cols:
        m.pop(col, None)
    for dcol in ['dt_rfi_created', 'dt_rfi_due', 'dt_rfi_resolved']:
        v = m.get(dcol)
        if v is not None and hasattr(v, 'date'):
            m[dcol] = v.date()
    return m

# Diversity tracker
class DiversityTracker:
    def __init__(self):
        self.l3_counts   = defaultdict(lambda: defaultdict(int))
        self.recent_subj = defaultdict(lambda: deque(maxlen=8))
        self.l3_by_trade = {}
        for trade, grp in _l3.groupby('cat_trade_l1'):
            self.l3_by_trade[trade] = grp['cat_trade_l3'].tolist()

    def sample_l3(self, trade):
        cats = self.l3_by_trade.get(trade, [])
        if not cats: return None
        counts  = self.l3_counts[trade]
        weights = np.array([1.0 / (counts[c] + 1) for c in cats])
        weights /= weights.sum()
        chosen  = np.random.choice(cats, p=weights)
        counts[chosen] += 1
        return chosen

    def get_recent_subjects(self, trade):
        return list(self.recent_subj[trade])

    def add_subjects(self, trade, subjects):
        for s in subjects:
            self.recent_subj[trade].append(s)

    def stage(self, n_done, n_total):
        pct = n_done / max(n_total, 1)
        if pct < 0.20: return 'early'
        if pct < 0.60: return 'mid'
        return 'late'

# Async LLM generation

def get_few_shot_examples(trade, sbu, n=3):
    pool = list(few_shot_pool.get((trade, sbu), []))
    if len(pool) < n:
        for key, examples in few_shot_pool.items():
            if key[0] == trade and key[1] != sbu:
                pool += examples
    return random.sample(pool, min(n, len(pool)))

def build_batch_prompt(batch_ctx):
    n      = batch_ctx['batch_size']
    trade  = batch_ctx['trade']
    l3_cat = batch_ctx['l3_cat'] or trade
    zone   = batch_ctx['zone']
    stage  = batch_ctx['stage']
    recent = batch_ctx['recent_subjects']

    few_shot_text = ""
    for i, ex in enumerate(batch_ctx['few_shots'], 1):
        few_shot_text += f"\nExample {i}:\n  Lesson: {ex.get('lesson','')}\n"

    zone_context = {
        'pre_sow':    "These RFIs are raised BEFORE the SOW begins — procurement or design clarifications that must be resolved before physical work can start. Frame them as work blockers.",
        'during_sow': "These RFIs are raised DURING active SOW execution — field condition conflicts, coordination issues, or specification ambiguities discovered on site.",
        'post_sow':   "These RFIs are raised AFTER the SOW is complete — punch list clarifications, warranty questions, or close-out disputes.",
    }.get(zone, "")

    diversity_instruction = ""
    if stage == 'mid' and recent:
        diversity_instruction = (
            f"\nEach RFI must cover a DIFFERENT component or issue than these recent subjects: "
            f"{'; '.join(recent)}. Vary the specific system, element, or location within {l3_cat}."
        )
    elif stage == 'late' and recent:
        diversity_instruction = (
            f"\nRequire location specificity (floor, zone, or grid line) in each subject. "
            f"Do NOT reuse these subject patterns: {'; '.join(recent)}."
        )

    rebaseline_ctx = ""
    if batch_ctx.get('has_rebaseline_context'):
        rebaseline_ctx = (
            f"\nSome of these RFIs occur near a rebaseline event "
            f"({batch_ctx.get('delay_magnitude','moderate')} delay, "
            f"primary delayed SOW: {batch_ctx.get('primary_delayed_sow','unknown')}). "
            f"A few RFIs may relate to the schedule disruption."
        )

    followup_instruction = ""
    if '_followup_parent_subject' in batch_ctx:
        followup_instruction = (
            "\nThis RFI is a FOLLOW-UP to an earlier unresolved RFI on the same "
            "trade and SOW. The original response was insufficient. Frame this as "
            "a clarification request digging into a specific unresolved aspect of "
            "the same system or component."
        )

    system_prompt = (
        "You are a construction document specialist generating realistic RFI records "
        "for a large general contractor. Output ONLY valid JSON — an array of objects. "
        "Never include real company names, architect names, project addresses, or PII. "
        "Use generic placeholders: 'the architect', 'the structural engineer', "
        "'Level 3', 'Grid B-4', 'the owner'."
    )

    user_prompt = f"""Generate exactly {n} realistic, DISTINCT RFI records for a construction project.

PROJECT CONTEXT:
- SBU: {batch_ctx['sbu']}
- Sector: {batch_ctx['asset_class']}
- Size: {batch_ctx['size_cat']} ({batch_ctx['area_sqft']:,.0f} sqft)
- Region: {batch_ctx['region']}

RFI BATCH CONTEXT (all {n} RFIs share this context):
- Trade (L1): {trade}
- System (L3): {l3_cat}
- SOW: {batch_ctx['sow']}
- Timing zone: {zone.replace('_', ' ').title()}
- {zone_context}{rebaseline_ctx}{followup_instruction}

REAL EXAMPLES (for tone and domain vocabulary):
{few_shot_text}

Each RFI must describe a DIFFERENT specific issue, component, or location within
{l3_cat}. Every RFI must specifically involve the {l3_cat} system — do not drift
into adjacent trades or systems. Vary both the system element AND the type of issue
(e.g. spec ambiguity, field condition, coordination conflict, submittal gap, code
question) — do not repeat the same issue type consecutively.{diversity_instruction}

Return a JSON ARRAY of exactly {n} objects. Each object must have:
{{
  "str_rfi_subject": "concise subject line (10-15 words)",
  "str_rfi_description": "detailed field description (50-100 words)",
  "str_rfi_response": "design team response resolving the issue (25-60 words)"
}}

Output the JSON array only — no preamble, no markdown fences."""

    return system_prompt, user_prompt


async def call_llm_batch_async(batch_ctx, semaphore):
    sys_p, usr_p = build_batch_prompt(batch_ctx)
    n = batch_ctx['batch_size']

    async with semaphore:
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = await async_client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                    system=sys_p,
                    messages=[{"role": "user", "content": usr_p}],
                    timeout=60.0
                )
                raw = resp.content[0].text.strip()
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$',          '', raw)
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    parsed = [parsed]

                results = []
                for item in parsed[:n]:
                    subj      = str(item.get('str_rfi_subject',     '')).strip()
                    desc      = str(item.get('str_rfi_description', '')).strip()
                    resp_text = str(item.get('str_rfi_response',    '')).strip()
                    if len(subj) >= 10 and len(desc) >= 40:
                        results.append((subj, desc, resp_text))

                while len(results) < n:
                    results.append((
                        f"Clarification required: {batch_ctx['trade']} — {batch_ctx['sow']}",
                        f"Field condition requires clarification for {batch_ctx['trade']} work in {batch_ctx['sow']}.",
                        "Design team to provide clarification per RFI protocol."
                    ))
                return results

            except Exception as e:
                wait = min(2 ** attempt, 4)
                print(f"    LLM attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

    return [(
        f"Clarification required: {batch_ctx['trade']} — {batch_ctx['sow']}",
        f"Field condition requires clarification for {batch_ctx['trade']} work in {batch_ctx['sow']}.",
        "Design team to provide clarification per RFI protocol."
    )] * n


# Per-project async generation

async def generate_project_rfis_async(proj):
    pid      = proj['ids_project_number_synth']
    sbu      = proj['cat_sbu']
    size_cat = get_size_category(proj['amt_area_gross'])
    start    = proj['dt_construction_start']
    end      = proj['dt_construction_end']
    dur      = proj['duration_days']

    if dur <= 0:
        print(f"  Skipping {pid}: zero duration")
        return 0

    proj_sows_df = sow_pd[sow_pd['ids_project_number_synth'] == pid]
    proj_sow_set = set(proj_sows_df['cat_sow'].tolist())
    proj_rb      = rebaseline_pd[rebaseline_pd['ids_project_number_synth'] == pid]
    diversity    = DiversityTracker()

    # Sample RFI count with rebaseline spike
    n_rfis = sample_n_rfis(sbu, size_cat)
    if len(proj_rb) > 0:
        spike_p25 = float(_rb_bounds.get('p25',    0.79))
        spike_med = float(_rb_bounds.get('median', 1.08))
        spike_p75 = float(_rb_bounds.get('p75',    1.58))
        sigma = (math.log(spike_p75 + 0.01) - math.log(spike_p25 + 0.01)) / (2 * 0.6745)
        mu    = math.log(spike_med + 0.01)
        ratio = float(np.clip(np.random.lognormal(mu, sigma), 0.5, 5.0))
        n_rfis = int(n_rfis * min(ratio, 2.0))
        bounds   = get_rfi_count_bounds(sbu, size_cat)
        hard_cap = int(float(bounds.get('p75', n_rfis)) * (1 + RFI_COUNT_SOFT_MARGIN))
        n_rfis   = min(n_rfis, hard_cap)

    # Step A: structured metadata — L3 NOT sampled here, sampled after grouping
    rfi_meta = []
    for seq in range(1, n_rfis + 1):
        trade    = sample_trade(sbu)
        lc_pct   = sample_lifecycle_pct(trade)
        zone     = sample_sow_timing_zone(trade)
        sow_name = sample_sow_for_trade(trade, proj_sow_set)

        sow_rows    = proj_sows_df[proj_sows_df['cat_sow'] == sow_name]
        matched_sow = sow_rows.iloc[0] if len(sow_rows) > 0 else None
        dt_created  = assign_dt_rfi_created(proj, lc_pct, matched_sow, zone)

        # Varied due date — weighted toward 7-day Suffolk standard
        due_days = random.choice(DUE_DATE_OPTIONS)
        dt_due   = min(dt_created + timedelta(days=due_days), end)

        status      = sample_status(sbu)
        dt_resolved = None
        if status == 'closed':
            res_days    = sample_resolution_days()
            dt_resolved = min(dt_created + timedelta(days=res_days), end)
        if status == 'open':
            # Only mark overdue if due date actually passed within project window
            proj_pct = (dt_created - start).days / max(dur, 1)
            if proj_pct >= 0.70 and dt_due < end and random.random() < 0.15:
                status = 'overdue'

        design_issue    = random.random() < get_design_issue_pct(trade)
        near_rb, rb_row = is_near_rebaseline(dt_created, proj_rb)
        sow_seq_timing  = {'pre_sow': 'Pre-SOW', 'post_sow': 'Post-SOW'}.get(zone, 'During')
        rfi_id          = f"SYN-{sbu[:3].upper()}-{pid[-6:]}-RFI-{seq:04d}"

        rfi_meta.append({
            'ids_project_number_synth': pid,
            'id_rfi_synth':             rfi_id,
            'dt_rfi_created':           dt_created,
            'dt_rfi_due':               dt_due,
            'dt_rfi_resolved':          dt_resolved,
            'cat_status':               status,
            'cat_trade_l1':             trade,
            'cat_sow':                  sow_name,
            'sow_sequence_timing':      sow_seq_timing,
            'n_rfi_sequence':           seq,
            'cat_design_issue':         design_issue,
            'is_near_rebaseline':       near_rb,
            'str_rfi_subject':          None,
            'str_rfi_description':      None,
            'str_rfi_response':         None,
            'parent_rfi_id':            None,
            '_trade': trade, '_sow': sow_name, '_zone': zone,
            '_near_rb': near_rb,
            '_delay_mag':   rb_row['delay_magnitude_category'] if near_rb else None,
            '_delayed_sow': rb_row['primary_delayed_sow']      if near_rb else None,
        })

    # Step B: group by (trade, zone, sow) — L3 sampled HERE after grouping
    # Grouping without L3 first, then L3 assigned per group to avoid inflating counts
    raw_group_map = defaultdict(list)
    for i, meta in enumerate(rfi_meta):
        raw_group_map[(meta['_trade'], meta['_zone'], meta['cat_sow'])].append(i)

    merged = defaultdict(list)
    catchall = defaultdict(list)
    for (trade, zone, sow), idxs in raw_group_map.items():
        if len(idxs) >= MIN_BATCH_GROUP:
            # Sample L3 once per group — not per RFI
            l3 = diversity.sample_l3(trade)
            merged[(trade, l3 or trade, zone, sow)].extend(idxs)
        else:
            catchall[trade].extend(idxs)

    for trade, idxs in catchall.items():
        zones    = [rfi_meta[i]['_zone'] for i in idxs]
        sows     = [rfi_meta[i]['cat_sow'] for i in idxs]
        rep_zone = max(set(zones), key=zones.count)
        rep_sow  = max(set(sows),  key=sows.count)
        rep_l3   = diversity.sample_l3(trade) or trade
        merged[(trade, rep_l3, rep_zone, rep_sow, '__catchall__')].extend(idxs)

    # Build all batch contexts
    all_batches = []
    n_done = 0
    for (trade, l3_cat, zone, sow, *_), idxs in merged.items():
        for batch_start in range(0, len(idxs), BATCH_SIZE):
            batch_idxs = idxs[batch_start: batch_start + BATCH_SIZE]
            b_size     = len(batch_idxs)
            batch_ctx  = {
                'sbu':          sbu,
                'asset_class':  proj['cat_main_asset_class'],
                'size_cat':     size_cat,
                'area_sqft':    proj['amt_area_gross'],
                'region':       proj['cat_region'],
                'trade':        trade,
                'l3_cat':       l3_cat,
                'sow':          sow,
                'zone':         zone,
                'batch_size':   b_size,
                'stage':        diversity.stage(n_done, n_rfis),
                'recent_subjects': diversity.get_recent_subjects(trade),
                'few_shots':    get_few_shot_examples(trade, sbu, n=3),
                'has_rebaseline_context': any(
                    rfi_meta[i]['_near_rb'] for i in batch_idxs
                ),
                'delay_magnitude':    rfi_meta[batch_idxs[0]]['_delay_mag'],
                'primary_delayed_sow':rfi_meta[batch_idxs[0]]['_delayed_sow'],
            }
            all_batches.append((batch_ctx, batch_idxs))
            n_done += b_size

    # Fire all batches concurrently in chunks — enables incremental flush
    semaphore   = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    DROP_COLS   = ['_trade', '_sow', '_zone', '_near_rb', '_delay_mag', '_delayed_sow']
    flushed_to  = 0
    ASYNC_CHUNK = MAX_CONCURRENT_CALLS * 2
    n_batches_done = 0

    for chunk_start in range(0, len(all_batches), ASYNC_CHUNK):
        chunk   = all_batches[chunk_start: chunk_start + ASYNC_CHUNK]
        tasks   = [call_llm_batch_async(ctx, semaphore) for ctx, _ in chunk]
        answers = await asyncio.gather(*tasks)

        for (batch_ctx, batch_idxs), llm_results in zip(chunk, answers):
            trade = batch_ctx['trade']
            diversity.add_subjects(trade, [r[0] for r in llm_results])
            n_batches_done += 1
            if n_batches_done % 10 == 0:
                print(f"    [{pid}] {n_batches_done}/{len(all_batches)} batches complete")
            for idx, (subj, desc, resp_text) in zip(batch_idxs, llm_results):
                rfi_meta[idx]['str_rfi_subject']     = subj
                rfi_meta[idx]['str_rfi_description'] = desc
                rfi_meta[idx]['str_rfi_response'] = (
                    None if rfi_meta[idx]['cat_status'] in ('open', 'overdue', 'draft')
                    else resp_text
                )

        # Incremental flush after each async chunk
        ready = [
            m for m in rfi_meta[flushed_to:]
            if m['str_rfi_subject'] is not None
        ]
        if len(ready) >= CHECKPOINT_EVERY:
            clean = [clean_record(m, DROP_COLS) for m in ready]
            _df   = spark.createDataFrame(pd.DataFrame(clean))
            (_df.write.format("delta").mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(OUT_RFI_TABLE))
            flushed_to += len(ready)
            print(f"  → Incremental flush: {len(ready)} records "
                  f"({flushed_to}/{n_rfis} flushed)")

    # Step C: chain generation
    CHAIN_RATE    = 0.10
    MAX_CHAIN_GAP = 21

    parents = [
        m for m in rfi_meta
        if m['cat_status'] in ('closed', 'overdue')
        and m['str_rfi_subject'] is not None
        and not m['str_rfi_subject'].startswith('Clarification required')
    ]
    chain_parents    = random.sample(parents, min(int(len(parents) * CHAIN_RATE), len(parents)))
    chain_candidates = []
    next_seq         = len(rfi_meta) + 1

    for parent in chain_parents:
        gap        = random.randint(7, MAX_CHAIN_GAP)
        dt_created = pd.Timestamp(parent['dt_rfi_created']) + timedelta(days=gap)
        if dt_created > end: continue
        due_days    = random.choice(DUE_DATE_OPTIONS)
        dt_due      = min(dt_created + timedelta(days=due_days), end)
        status      = sample_status(sbu)
        dt_resolved = None
        if status == 'closed':
            res_days    = sample_resolution_days()
            dt_resolved = min(dt_created + timedelta(days=res_days), end)
        if status == 'open':
            proj_pct = (dt_created - start).days / max(dur, 1)
            if proj_pct >= 0.70 and dt_due < end and random.random() < 0.15:
                status = 'overdue'
        chain_candidates.append({
            'parent':      parent,
            'dt_created':  dt_created,
            'dt_due':      dt_due,
            'dt_resolved': dt_resolved,
            'status':      status,
            'l3_cat':      diversity.sample_l3(parent['cat_trade_l1']) or parent['cat_trade_l1'],
        })

    chain_group = defaultdict(list)
    for i, fc in enumerate(chain_candidates):
        chain_group[fc['parent']['cat_trade_l1']].append(i)

    chain_batches = []
    for trade, idxs in chain_group.items():
        for batch_start in range(0, len(idxs), BATCH_SIZE):
            batch_idxs = idxs[batch_start: batch_start + BATCH_SIZE]
            rep        = chain_candidates[batch_idxs[0]]
            parent_rep = rep['parent']
            chain_batches.append(({
                'sbu': sbu, 'asset_class': proj['cat_main_asset_class'],
                'size_cat': size_cat, 'area_sqft': proj['amt_area_gross'],
                'region': proj['cat_region'],
                'trade': trade, 'l3_cat': rep['l3_cat'],
                'sow': parent_rep['cat_sow'],
                'zone': (
                    parent_rep['sow_sequence_timing']
                    .lower()
                    .replace('pre-sow', 'pre_sow')
                    .replace('post-sow', 'post_sow')
                    .replace('during', 'during_sow')
                ),
                'batch_size': len(batch_idxs),
                'stage': 'mid',
                'recent_subjects': diversity.get_recent_subjects(trade),
                'few_shots': get_few_shot_examples(trade, sbu, n=2),
                'has_rebaseline_context': False,
                'delay_magnitude': None, 'primary_delayed_sow': None,
                '_followup_parent_subject': None,
            }, batch_idxs))

    chain_tasks   = [call_llm_batch_async(ctx, semaphore) for ctx, _ in chain_batches]
    chain_answers = await asyncio.gather(*chain_tasks)

    chain_records = []
    for (batch_ctx, batch_idxs), llm_results in zip(chain_batches, chain_answers):
        diversity.add_subjects(batch_ctx['trade'], [r[0] for r in llm_results])
        for batch_pos, fc_idx in enumerate(batch_idxs):
            fc     = chain_candidates[fc_idx]
            parent = fc['parent']
            subj, desc, resp_text = llm_results[batch_pos]
            rfi_id = f"SYN-{sbu[:3].upper()}-{pid[-6:]}-RFI-{next_seq:04d}"
            chain_records.append({
                'ids_project_number_synth': pid,
                'id_rfi_synth':             rfi_id,
                'dt_rfi_created':           fc['dt_created'].date() if hasattr(fc['dt_created'], 'date') else fc['dt_created'],
                'dt_rfi_due':               fc['dt_due'].date()     if hasattr(fc['dt_due'],     'date') else fc['dt_due'],
                'dt_rfi_resolved':          fc['dt_resolved'].date() if fc['dt_resolved'] is not None and hasattr(fc['dt_resolved'], 'date') else fc['dt_resolved'],
                'cat_status':               fc['status'],
                'cat_trade_l1':             parent['cat_trade_l1'],
                'cat_sow':                  parent['cat_sow'],
                'sow_sequence_timing':      parent['sow_sequence_timing'],
                'n_rfi_sequence':           next_seq,
                'cat_design_issue':         parent['cat_design_issue'],
                'is_near_rebaseline':       parent['is_near_rebaseline'],
                'parent_rfi_id':            parent['id_rfi_synth'],
                'str_rfi_subject':          subj,
                'str_rfi_description':      desc,
                'str_rfi_response': (
                    None if fc['status'] in ('open', 'overdue', 'draft')
                    else resp_text
                ),
            })
            next_seq += 1

    # Step D: final flush — remaining main records + all chains
    remaining_main = [
        clean_record(m, DROP_COLS)
        for m in rfi_meta[flushed_to:]
        if m['str_rfi_subject'] is not None
    ]
    final_records = remaining_main + chain_records
    if final_records:
        _df = spark.createDataFrame(pd.DataFrame(final_records))
        (_df.write.format("delta").mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(OUT_RFI_TABLE))
        print(f"  → Final flush: {len(remaining_main)} main + "
              f"{len(chain_records)} chain records")

    total = flushed_to + len(remaining_main) + len(chain_records)
    return total


def run_project(proj):
    """Run async generation for one project in its own event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(generate_project_rfis_async(proj))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# Orchestration
print(f"Starting generation: {len(projects_pd)} projects | "
      f"batch_size={BATCH_SIZE} | workers={MAX_WORKERS} | "
      f"concurrent_calls={MAX_CONCURRENT_CALLS} | model={LLM_MODEL}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(run_project, proj): proj['ids_project_number_synth']
        for _, proj in projects_pd.iterrows()
    }
    for future in as_completed(futures):
        pid = futures[future]
        try:
            total = future.result()
            print(f"  ✓ {pid}: {total} RFIs written")
        except Exception as e:
            print(f"  ✗ {pid} failed: {e}")
        time.sleep(INTER_PROJECT_SLEEP)

# Validation summary
print("\n=== GENERATION COMPLETE ===")
out = spark.read.table(OUT_RFI_TABLE)
print(f"Total records: {out.count()}")
out.groupBy("cat_sbu", "cat_trade_l1").count().orderBy(
    "cat_sbu", "cat_trade_l1"
).show(60, truncate=False)
print("\nStub rate:")
print(out.filter(
    F.col("str_rfi_description").like("Field condition requires clarification%")
).count())
