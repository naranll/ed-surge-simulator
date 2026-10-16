# Data processing package

This package prepares documented, reproducible inputs for the ED Surge Simulator. NHAMCS Phase 2A provides codebook-aware preprocessing; NOAA Phase 2B provides weather preprocessing and meteorological heatwave detection. OpenFEMA Phase 2C provides acquisition, an independent source audit, and a declaration-level handoff table. Physical-event consolidation and HURDAT2 ingestion are not implemented.

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
