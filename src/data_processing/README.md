# Data processing package

This package prepares documented, reproducible inputs for the ED Surge Simulator. NHAMCS Phase 2A provides codebook-aware preprocessing; NOAA Phase 2B provides weather preprocessing and meteorological heatwave detection. OpenFEMA Phase 2C provides acquisition, an independent source audit, and a declaration-level handoff table. HURDAT2 provides an Atlantic best-track handoff and a reviewed FEMA candidate crosswalk without consolidating physical events.

## Planned workflow

Each source follows separate ingestion, audit, cleaning, feature engineering, validation, and export steps as those phases are implemented. Original source files remain unchanged under `data/raw/`; derived tables and reports are written to `data/processed/` and `reports/`.

Modules are separated by source so the NHAMCS 2022 ED data, NOAA Houston weather, FEMA/HURDAT2 disaster data, and simulation parameters can each use their own codebooks, formats, assumptions, and validation rules.

Observed data, cited literature-based parameters, and synthetic assumptions must remain explicitly labeled. Patient-level NHAMCS records must not be joined to local weather or disaster events without a valid geographic and temporal linkage.

See [`docs/data_sources.md`](../../docs/data_sources.md) for source URLs and preprocessing notes, and [OpenFEMA instructions](../../docs/openfema_acquisition_audit.md) for acquisition/offline auditing.

## Phase 1: ingestion and quality audits

Install the project dependency from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run NHAMCS audit on the default local archive (`data/raw/nhamcs/ed2022-stata.zip`), or pass a CSV/DTA path explicitly:

```bash
python src/data_processing/nhamcs_2022.py
python src/data_processing/nhamcs_2022.py --input data/raw/nhamcs/ED2022.csv --report reports/nhamcs_2022_audit.json
```

Run NOAA GHCN-Daily audit on the default station CSV (`data/raw/noaa/USW00012960.csv`):

```bash
python src/data_processing/noaa_weather.py
python src/data_processing/noaa_weather.py --input data/raw/noaa/USW00012960.csv --report reports/noaa_usw00012960_audit.json
```

Each command writes a JSON audit report under `reports/` by default. The audit remains descriptive and is preserved unchanged. NOAA observations are not converted and heatwave indicators are not calculated. Reports and source files are ignored by Git.

## Phase 2A: NHAMCS preprocessing

Run the CDC codebook-aware preprocessing from the repository root:

```bash
python src/data_processing/nhamcs_preprocessing.py
python src/data_processing/nhamcs_preprocessing.py --input data/raw/nhamcs/ed2022-stata.zip
```

It writes `data/processed/nhamcs_2022_cleaned.csv`, `data/processed/nhamcs_2022_triage_distribution.csv`, and `reports/nhamcs_2022_preprocessing.json`. The cleaned table retains source columns and adds typed fields, category labels, and missing-reason columns. Codebook sentinels are variable-specific; raw values remain intact, Stata extended missings are serialized as marker strings in original columns, and no rows are removed.

The triage table reports counts and percentages for all source rows, percentages among acuity levels 1–5, and separate categories for codes 0 and 7, unknown, blank, and Stata extended missings. Survey-weighted point estimates use `PATWT` and are emitted only when `PATWT`, `CSTRATM`, and `CPSUM` are present and valid. The pipeline does not calculate standard errors or confidence intervals. Output metadata documents denominators, exclusions, codebook rules, and limitations.

Run the small synthetic tests with:

```bash
python -m unittest discover -s tests -v
```

### Validation fixes and numerical precision

The 2022 codebook (item 34, page 36) does not define `IMMEDR=-7`.
Unexpected values are counted in `other_or_undocumented`, with the raw value
retained; they are excluded from the levels 1-5 denominator. Every record
contributes to exactly one distribution category.

Weighted calculations promote a separate PATWT series to float64 and use
`math.fsum` for category sums and denominators. Original weights remain unchanged.
This avoids float32 accumulation errors but cannot recover precision absent from
the source. Raw float32 CSV decimals round-trip at their source dtype; generic
float64 reads may differ slightly from the original binary values. CSTRATM and
CPSUM identify the survey design and are never numerical weights. These are
descriptive point estimates; survey reliability has not been established.

The JSON manifest now records local source path/hash/archive member, runtime and
code version hashes, actual output paths, column dtypes/null counts, denominators,
exclusions, precision limitations and codebook references. Original download URL
and date are marked unknown when unavailable. For CSV reads use each output's
`schema` (`csv_read_dtype`), `keep_default_na=False`, and `na_values=[""]`.
HHMM and mixed raw columns must remain strings to preserve zeros and Stata tokens.
Schema `null_count` and `empty_string_count` distinguish in-memory nulls from raw
empty strings; `csv_null_count` describes the combined count after the documented
CSV read. For lexical raw-value inspection, read all columns with `dtype=str` and
`keep_default_na=False` and omit `na_values` so empty strings remain empty strings.

Run the independent real-data validation after generating the outputs:

```bash
python tests/validate_nhamcs_real_data.py
```

It compares all original cells, independently recalculates counts and weighted
values, checks the synthetic unexpected-code case, and reproduces outputs in a
temporary directory. Results are written to
`reports/nhamcs_2022_validation_after_fixes.md` and its evidence JSON. The original
validation report is retained as history. This check requires the local real data.

## NHAMCS 2018-2022 simulation handoff

The simulation team's primary patient-profile table is produced from the local,
third-party processed CSV. Run from the repository root:

```bash
python src/data_processing/nhamcs_2018_22.py
```

This writes:

- `data/processed/nhamcs_2018_22_simulation_visits.csv`
- `reports/nhamcs_2018_22_preprocessing.json`

The output preserves all input records and their original order. Important source
values use `*_raw` columns; parsed values and explicit status fields are separate.
Valid arrival times provide `arrival_hour` and `arrival_minute_of_day` for relative
arrival-pattern estimation. `triage_acuity_code` supports empirical triage sampling,
and the retained demographic and visit categories support patient-profile sampling.
Use `year_clean` for year-stratified summaries.

Do not assign weekday names from `day_of_week_code`: its 1-7 label mapping has not
been verified. Do not use `wait_time_minutes_clean` as provider service time. The
file has no survey weights, hospital identifiers, complete dates, service durations,
or capacity measures, and it cannot establish absolute Houston ED arrival volumes.
Exact duplicates remain present and are identified by `exact_duplicate_flag`.

Run its focused tests with:

```bash
python -m unittest tests.test_nhamcs_2018_22 -v
```

### Minimal parameter contract

Generate `data/processed/params.json` from the existing processed visits:

```bash
python src/data_processing/simulation_parameters.py
```

Optional `--input` and `--output` arguments select alternative paths. No earlier
pipeline is rerun. The JSON is Git-ignored and deterministic for the same input
bytes and filename. The exporter records the input SHA-256, retains duplicate
records in counts, and pools all source years without survey weighting.

Schema version `1.0` has these exact fields:

| Field | Type / meaning |
|---|---|
| `schema_version` | String `1.0` |
| `source` | Object: `dataset`, `file_name`, `sha256`, `provenance_status`, `weighting`, `pooling` (strings), `records` (integer) |
| `hourly_arrivals` | Object: `hours` (integers 0-23), `counts` (24 integers), `proportions` (24 numbers), `denominator`, `excluded_records` (integers), `excluded_by_status` (`missing` and `invalid` integer counts) |
| `triage` | Same distribution fields, with `codes` (integers 1-5) replacing `hours`; counts/proportions have five entries |
| `unavailable_parameters` | Keys `provider_service_times`, `admission_probabilities`, `boarding_durations`; each object has `value: null`, `status: unavailable_from_primary_dataset`, `reason` and `owner: Simulation Team` |
| `limitations` | Array of scientific and integration limitation strings |

Arrays align by index: `hourly_arrivals.proportions[h]` is the relative probability
for hour `h`; pair `triage.codes` with `triage.proportions` for numeric acuity
sampling. Each distribution sums to one within floating-point tolerance and has
its own denominator. Only rows marked `valid` contribute to that distribution;
a missing arrival does not exclude a valid triage record. Unexpected status
strings, inconsistent valid values, and zero denominators raise an error.

These are sample proportions, not observed Houston rates. The Simulation Team
must supply baseline daily ED volume before using the hourly curve to set arrival
intensity. It must separately define service times, admission, boarding and
staffing/capacity assumptions. A null parameter means unavailable, never zero;
consumers must resolve it explicitly before using that part of the model. Waiting
time is not provider service time. No values are borrowed from NHAMCS 2022, and
unverified weekday codes are not converted into named weekday factors.

This is a documented minimal contract; compatibility with an external simulator's
original key names must be handled by its consumer. No simulator is included here.

Focused tests: `python -m unittest tests.test_simulation_parameters -v`.
