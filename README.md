# Synthetic Data Generation Framework
## Suffolk Construction — Data Analytics & Data Science

---

## Overview

Suffolk's Platinum analytics layer contains approximately 200 fully modelled construction projects. This is insufficient for training advanced ML models which typically require thousands of examples. This project builds a synthetic data generation framework that produces 1,000+ statistically valid, privacy-preserving construction project records across multiple interconnected tables — preserving real-world distributions, business constraints, and causal relationships between tables.

The framework has already delivered proof-of-concept value: synthetic data was used to train the Tripwires ML model, which achieved good results for the team and directly motivated expanding the framework to additional tables.

---

## How It Works

The framework follows a three-layer architecture for every table it generates.

The **profiling layer** extracts statistical signatures from real Platinum data — distributions, correlations, percentiles, and business rule patterns at three granularities: SBU × Asset Class, SBU only, and global. The hierarchical fallback ensures robust generation even for sparse combinations.

The **generation layer** uses these profiles to produce synthetic records. Hard business rules are enforced as constraints during generation. Soft statistical bounds use jitter to prevent artificial clustering at profile boundaries. A hierarchical lookup always falls back gracefully when a specific combination has insufficient real data.

The **validation layer** runs distribution similarity checks, referential integrity verification, and business rule compliance checks against the generated output.

---

## Output Tables

### synthetic_project_meta_data_profiled

1,000 synthetic construction project records seeded from real Suffolk Platinum data using CTGAN and Gaussian Copula models. This is the root table — every downstream table links to it via `ids_project_number_synth`.

Key fields include SBU, region, asset class, contract value, cost, gross area, floor count, and construction start/end dates. The generation uses custom SDV constraint classes to enforce hard rules during model sampling — positive gross area, integer floor counts, Mission Critical capped at 3 floors, cost/contract ratio alignment. Post-generation soft bounding with jitter refines the statistical distributions and a derived field chain (floors → area → contract → cost) preserves realistic correlations.

---

### synthetic_schedule_baseline_v1

Schedule baseline records for all 1,000 synthetic projects using an SCD Type 2 structure — one record per baseline version per project, capturing the full rebaseline history.

Rebaseline counts are sampled from a zero-inflated Poisson distribution (54% of real projects have no rebaselines). Delay magnitudes use slack-conditional sampling so delays scale with remaining schedule slack, preventing unrealistic late-project mega-delays. TCO monotonicity is enforced across all versions. Each rebaseline version carries its own NTP, TCO, and effective date range.

Paired with this is `rebaseline_event_metadata_v1`, an intermediate enrichment table that captures the delay magnitude, primary delayed SOW, and number of SOWs impacted for each rebaseline event. This feeds downstream into RFI generation to model the volume spike that follows schedule disruptions.

---

### synthetic_task_summary_by_sow_v1

SOW-level task summaries for all 1,000 synthetic projects — 23 SOW records per project covering the full construction work breakdown from initial site work through final inspections.

The generation sequence is critical: concurrency depth is sampled first, then overlap structure is enforced (84% of consecutive SOWs overlap in real data with a median of 339 days), then SOWs are placed in time proportional to the construction window, then tasks are placed within SOWs using profiled timing shapes (early/mid/late position distributions). Key profiling findings include overrun rates by SOW type (Final Inspections at 57%, Interior Finishes at 33%), critical path percentages grounded in real P6 data (1–9% range), and task count distributions by SOW category.

---

### synthetic_rfi_v1

Synthetic Request for Information records for all 1,000 projects — the most complex table in the framework, combining statistical structured fields with LLM-generated text.

**Structured fields** are fully profiled from real Procore and Platinum data: RFI count by SBU × size (commercial XLarge median 1,231, mission critical XLarge median 163), trade weights by SBU, lifecycle timing by trade using beta distributions parameterised from p10–p90 (A SUBSTRUCTURE peaks at 26% lifecycle, C INTERIORS at 90%), SOW timing zone probabilities (Pre-SOW / During / Post-SOW), resolution time, design issue rates by trade, status distribution by SBU, and a rebaseline spike ratio (median 1.08× volume uplift in the 30 days after a rebaseline).

**Text fields** (`str_rfi_subject`, `str_rfi_description`, `str_rfi_response`) are generated using Claude Haiku, selected after a three-model smoke test comparing Haiku, Sonnet, and Opus. Haiku produced the best results — technically specific descriptions with real measurements and grid references, staying anchored to the assigned Uniformat L3 system, at 21 seconds per 10-RFI batch.

**Diversity control** uses a per-project DiversityTracker with inverse-frequency L3 category weighting, a rolling 8-subject do-not-repeat window per trade, perspective rotation across 5 originator types, and explicit floor/zone/grid location forcing. Few-shot examples are retrieved from a pool keyed by trade × SBU × L3, selecting up to 2 examples per L3 category within each cell to maximise system coverage across the Uniformat hierarchy.

**Chain RFIs** (~10% of closed records) generate a follow-up record 7–21 days after the parent, linked via `parent_rfi_id`, capturing the realistic pattern of multi-round clarification sequences.

**Performance**: generation runs asynchronously using `asyncio.gather` with a semaphore limiting concurrent LLM calls to 10 per project, across 3 parallel project threads. The script is fully resumable — it auto-detects already-generated projects in the output table and skips them on rerun.

---

## Key Design Principles

**Statistically grounded, not hardcoded.** Every distribution, rate, and proportion is profiled from real data. Hardcoded values are limited to logical constraints (date ordering, non-negativity, schema compliance).

**Generation ordering matters.** Tables are generated in dependency order — projects first, then schedule, then task summary, then RFIs. Within each table the generation sequence is also carefully ordered (e.g. SOW concurrency before overlap before temporal placement).

**Soft bounding over hard clipping.** Rather than hard-clamping values at profile boundaries, jitter within a controlled margin allows occasional realistic outliers while preventing extreme tail behaviour.

**Referential integrity throughout.** Every synthetic record carries the correct foreign keys linking it to its parent table. `ids_project_number_synth` is the spine connecting all tables.

---

## Tech Stack

PySpark and Databricks for large-scale data processing, Python for generation scripts, SDV (Synthetic Data Vault) with CTGAN and Gaussian Copula for tabular generation, Anthropic Claude API for LLM text generation, Delta Lake for output storage.

---

## Data Sources

All profiling uses Suffolk's Platinum analytics catalogue in Databricks:
`platinum.models.project`, `platinum.models.schedule_baseline`, `platinum.models.task_baseline`, `gold.procore.rfis`, `analysts.self_managed.lessons_learned`, `analysts.self_managed.lessons_learned_meta_data`
