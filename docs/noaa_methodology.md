# NOAA Phase 2B: preprocessing and meteorological episodes

## Scope and commands

Run from the repository root on `feature/data-preprocessing`:

```powershell
.venv/Scripts/python.exe -B src/data_processing/noaa_preprocessing.py
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B tests/validate_noaa_real_data.py
```

On Ubuntu/LinuxONE, use `python` or `.venv/bin/python` instead. Python 3.10 and
Pandas are sufficient; no dependencies were added. Windows Python 3.10.0/Pandas
2.3.3 was tested. Ubuntu 22.04 and s390x execution remain untested; use a Pandas
build supported on that platform. Installation/build availability is not implied.

Custom destinations:

```text
python src/data_processing/noaa_preprocessing.py --input data/raw/noaa/USW00012960.csv --config configs/noaa_heatwave.json --output-dir data/processed --metadata reports/noaa_preprocessing_metadata.json
```

Original NOAA ingestion/audits and all NHAMCS files are unchanged. No patient-weather
joins, ED demand multipliers, or simulation parameters are produced.

## Source and units

- Native format, elements, missing codes and flags:
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
- Custom CSV format and attributes:
  https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf
- Station archive used for independent unit verification:
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00012960.dly
- Reference-period context:
  https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
- Heatwave methodology examples (not validation of our exact definition):
  https://doi.org/10.1175/JCLI-D-12-00383.1 and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3040608/
- ED epidemiology context: https://www.bmj.com/content/375/bmj-2021-065653

The reviewed local file has SHA256
`e9db3672fe89bb6b7bbd1853af015be4380c74ba14d60035705863da5b818d05`.
All 62,853 TMAX/TMIN/PRCP values matched the official native station archive during
the methodology review. This file uses tenths Celsius and tenths millimeters;
divide those three elements by ten. The original download request, URL and date
are unknown. The CLI rejects a different source hash. A new export requires
independent unit verification before deliberately updating that hash; column
names alone do not establish units. The low-level preparation function requires
an explicit native-unit argument and should only receive already verified data.

## Preservation and eligibility

All 102 source columns are loaded as strings without automatic NA interpretation
and exported unchanged, including whitespace, empty strings and attribute text.
CSV quoting/line endings are serialization, not preserved source bytes; the raw
file itself is never written. `source_row` is the original one-based record number.
Outputs retain original row order. Invalid dates, duplicate keys and other stations
raise errors rather than silently dropping records. Derived-name collisions fail.

Attributes split positionally into measurement, quality, source and optional
observation time. A source flag such as `X` is not a quality flag. Malformed
attributes and nonblank quality flags exclude only the affected element. Original
attributes always remain available. `-9999` is handled only as the native-format
sentinel; blanks, nonnumeric/nonfinite values and noninteger native values receive
separate reasons. Negative PRCP is invalid. Negative temperatures and large values
are retained unless another eligibility rule excludes them. TMAX<TMIN is reported
as a diagnostic; this pipeline does not invent a correction.

Trace precipitation retains reported zero with `prcp_is_trace=True`; zero with a
trace flag is not a confirmed dry day. PRCP observation time `2400` remains text.
Temperature observation times are absent in this source. No timestamps or time
zone conversions are inferred. Observation-time metadata is not NHAMCS HHMM.

## Approved calculation

- Station USW00012960; baseline 1991-01-01 through 2020-12-31.
- TMAX calendar-day 95th percentile; strict `>`; minimum two consecutive days.
- For each of 365 month-days, pool all eligible TMAX in the circular +/-15-day
  window across baseline years; linear interpolation at rank `(n-1)*0.95`.
- Exclude February 29 samples. Its threshold is the mean of the February 28 and
  March 1 thresholds. The actual leap day counts in event duration.
- The current baseline has 10,958 eligible days and 930 samples per regular pool.
- The explicit MVP coverage requirement is 100% eligible baseline days AND 100%
  expected observations in every window. This is a conservative project rule,
  not a NOAA standard. Insufficient coverage fails before exporting.
- All seasons are classified. Winter events can represent relative warm spells.
- No target-year or post-baseline observations enter fitting. Thresholds are not
  rounded before comparisons. Pre-2021 labels are retrospective, not forecasts.

Missing dates are represented on an internal complete calendar, listed in metadata,
and interrupt runs; they do not add synthetic rows to the cleaned source export.
Invalid TMAX also interrupts runs. `exceeds_threshold` is unknown if exposure or
threshold is unavailable. `heatwave_day` is true for confirmed observed runs,
false for assessable non-events, and unknown for unassessable days or short hot
runs adjacent to unknown observations or a series edge. Qualifying segments next
to gaps/edges have censored boundaries: their full duration may be longer or they
may be connected through unknown days. These are observed segments, not claims
that unknown intervening dates were cool.

`event_confirmed_on`/`confirmed_on` is the date of the minimum-duration-th observed
hot day. The event's earlier days are labeled retrospectively. This does not mean
the event was knowable on its first day. Subsequent source extensions may change
boundary status or segmentation; IDs are deterministic for fixed inputs/config.

Configuration supports percentile, window, duration, baseline and coverage changes.
`definition="absolute"` with finite `absolute_threshold_c` selects an explicit
secondary absolute definition; the primary config remains percentile. The optional
absolute list is a scenario menu, not automatically executed or clinically validated.
The baseline table remains the configured percentile reference for absolute runs;
the daily `threshold_c` is the actual absolute threshold used. Unsupported changes
to scientific policies are rejected rather than silently ignored. Each config has
a stable SHA256-derived ID. Only the primary configuration was validated here.

## Outputs and schemas

`data/processed/noaa_houston_cleaned.csv`: all source columns unchanged, plus:

| Fields | Type / meaning |
|---|---|
| source_row, station_id, date | int64, string, ISO date string |
| tmax_c, tmin_c, prcp_mm | Nullable Float64 eligible converted values |
| {tmax,tmin,prcp}_{measurement_flag,quality_flag,source_flag,observation_time} | Strings; empty means absent component |
| {tmax,tmin,prcp}_{validation_status,missing_reason} | Eligibility/exclusion reason; missing reason empty when eligible |
| prcp_is_trace | Nullable boolean; measurement flag T |
| threshold_c | Float64 threshold actually used |
| exceeds_threshold, heatwave_day | Nullable booleans; empty means unknown |
| heatwave_event_id, event_confirmed_on | Nullable strings |
| classification_status, config_id | Status and definition provenance |

`data/processed/heatwave_events.csv`: event_id, station_id, config_id, definition,
start_date, end_date, confirmed_on, duration_days (integer), max_tmax_c,
mean_tmax_c, max_exceedance_c, cumulative_exceedance_c_days (float64), boundary_status.
Dates are ISO strings. Mean temperature is the mean of daily maxima, not hourly
mean temperature. Intensity is TMAX minus that day's threshold; cumulative intensity
has units Celsius-days. Summaries cover observed segments only.

`data/processed/noaa_baseline_thresholds.csv`: 366 rows, with calendar_day,
threshold_c, sample_count, expected_sample_count, contributing_years,
threshold_method and config_id. February 29 has zero direct samples and the
explicit adjacent-threshold method. The other 365 rows are fitted percentiles.

`reports/noaa_preprocessing_metadata.json`: actual paths, source/config/code hashes,
configuration, schema dtypes, null counts, output hashes and row counts, exclusions,
baseline/gap diagnostics, software versions, references and limitations.

Reload source columns with `dtype=str, keep_default_na=False`. Derived numeric/null
columns require the manifest dtypes and empty-cell handling. Boolean outputs are
True/False/empty; preserve unknown as nullable boolean. Empty source fields are
literal source strings, not inferred numeric nulls. JSON never emits NaN tokens.

## Tests and independent validation

Synthetic tests cover scaling, flags, traces, 2400, raw preservation, coverage,
hand-calculated quantiles, wrap/leap handling, strict ties, durations, gaps, invalid
exposure, censoring, confirmation dates, stable IDs, absolute mode and repeat exports.

The independent validator reads raw strings and exported files. It constructs
baseline pools separately with standard-library month-day loops, calculates sorted
linear quantiles, detects event runs independently, and reconciles intensity using
`math.fsum`. Production helpers are used only to regenerate into a temporary
directory. CSVs must match byte-for-byte; metadata is compared after normalizing
intentional output-path changes.

`reports/noaa_protected_files_before.json` records hashes captured before Phase 2B
for raw data and established NHAMCS outputs/reports. It is local validation evidence,
not recreated from current files by the validator. Preserve it when reproducing
the unchanged-file check. Results are written to `reports/noaa_validation.json`
and `reports/noaa_validation.md`. Existing ignore rules exclude data and reports;
no raw datasets should be force-added to Git.

## Limitations

These are project-defined meteorological episodes. Airport exposure does not establish
citywide or individual exposure; station-history homogeneity was not assessed. No
ED surge multiplier or clinical threshold is inferred. Daily hospital arrivals,
appropriate spatial exposure, confounders and lagged epidemiological models would
be needed. National NHAMCS records lack a valid patient/day Houston linkage.
The partial 1969 and 2026 years must not be treated as full annual observations.
Cross-platform numerical/serialization reproducibility is untested. Identical
software/input/configuration is the supported reproducibility scope for this run.
