"""Minimal preprocessing for the third-party NHAMCS 2018-2022 CSV.

The source is treated as a transformed, unweighted visit sample. Important
source values are retained as strings, and no row is dropped or imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/raw/nhamcs/nhamcs_data_2018_22.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/nhamcs_2018_22_simulation_visits.csv"
DEFAULT_REPORT = REPO_ROOT / "reports/nhamcs_2018_22_preprocessing.json"

REQUIRED_COLUMNS = [
    "year", "visit_month", "day_of_week", "arrival_time",
    "target_triage_acuity", "wait_time_minutes", "age", "sex",
    "ems_arrival", "insurance", "is_injury_poison", "residence",
    "race", "region", "episode",
]

RAW_OUTPUT_COLUMNS = [
    "year", "visit_month", "day_of_week", "arrival_time",
    "target_triage_acuity", "wait_time_minutes", "age", "sex",
    "ems_arrival", "insurance", "is_injury_poison", "residence",
    "race", "region", "episode",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: str | Path) -> pd.DataFrame:
    """Load every source cell lexically so blank and formatted values survive."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NHAMCS 2018-2022 source CSV not found: {source}")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")
    return frame


def _blank(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().eq("")


def _integer_field(
    raw: pd.Series, minimum: int, maximum: int
) -> tuple[pd.Series, pd.Series]:
    stripped = raw.astype("string").str.strip()
    missing = stripped.eq("")
    numbers = pd.to_numeric(stripped.mask(missing), errors="coerce")
    whole = numbers.notna() & numbers.eq(numbers.round())
    valid = whole & numbers.between(minimum, maximum)
    clean = numbers.where(valid).astype("Int64")
    status = pd.Series("invalid", index=raw.index, dtype="string")
    status.loc[missing] = "missing"
    status.loc[valid] = "valid"
    return clean, status


def _arrival_fields(raw: pd.Series) -> pd.DataFrame:
    number, status = _integer_field(raw, 0, 2359)
    hour = (number // 100).astype("Int64")
    minute = (number % 100).astype("Int64")
    valid_hhmm = status.eq("valid") & hour.between(0, 23) & minute.between(0, 59)
    status = status.mask(status.eq("valid") & ~valid_hhmm, "invalid")
    number = number.where(valid_hhmm)
    hour = hour.where(valid_hhmm)
    minute = minute.where(valid_hhmm)
    hhmm = number.map(lambda value: pd.NA if pd.isna(value) else f"{int(value):04d}").astype("string")
    minute_of_day = (hour * 60 + minute).astype("Int64")
    return pd.DataFrame({
        "arrival_hhmm": hhmm,
        "arrival_hour": hour,
        "arrival_minute": minute,
        "arrival_minute_of_day": minute_of_day,
        "arrival_time_status": status,
    }, index=raw.index)


def _wait_fields(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    stripped = raw.astype("string").str.strip()
    missing = stripped.eq("")
    number = pd.to_numeric(stripped.mask(missing), errors="coerce")
    finite = number.notna()
    valid = finite & number.ge(0)
    clean = number.where(valid).astype("Float64")
    status = pd.Series("invalid_non_numeric", index=raw.index, dtype="string")
    status.loc[missing] = "missing"
    status.loc[finite & number.lt(0)] = "invalid_negative"
    status.loc[valid] = "valid"
    return clean, status


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "pandas_dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
        }
        for name, series in frame.items()
    ]


def preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a simulation handoff table without dropping or imputing visits."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")

    output = pd.DataFrame(index=frame.index)
    output["source_row_number"] = pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="int64")
    for column in RAW_OUTPUT_COLUMNS:
        output[f"{column}_raw"] = frame[column].astype("string")

    output["year_clean"], output["year_status"] = _integer_field(frame["year"], 2018, 2022)
    output["visit_month_clean"], output["visit_month_status"] = _integer_field(frame["visit_month"], 1, 12)
    output["day_of_week_code"], output["day_of_week_status"] = _integer_field(frame["day_of_week"], 1, 7)
    output = pd.concat([output, _arrival_fields(frame["arrival_time"])], axis=1)
    output["triage_acuity_code"], output["triage_status"] = _integer_field(
        frame["target_triage_acuity"], 1, 5
    )
    output["wait_time_minutes_clean"], output["wait_time_status"] = _wait_fields(
        frame["wait_time_minutes"]
    )
    output["age_clean"], output["age_status"] = _integer_field(frame["age"], 0, 130)

    duplicate_flag = frame.duplicated(keep=False)
    output["exact_duplicate_flag"] = duplicate_flag.astype("bool")

    year_counts = output["year_clean"].value_counts(dropna=False).sort_index()
    triage_counts = output["triage_acuity_code"].value_counts(dropna=False).sort_index()
    duplicate_excess = int(frame.duplicated(keep="first").sum())
    duplicate_groups = int(frame.loc[duplicate_flag].drop_duplicates().shape[0])
    metadata: dict[str, Any] = {
        "dataset": "Third-party processed NHAMCS 2018-2022 visit sample",
        "records": {
            "input": int(len(frame)),
            "output": int(len(output)),
            "dropped": 0,
            "source_order_preserved": True,
        },
        "source_columns": [str(column) for column in frame.columns],
        "output_schema": _schema(output),
        "validation": {
            "year_counts": {str(int(k)): int(v) for k, v in year_counts.items() if pd.notna(k)},
            "triage_counts": {str(int(k)): int(v) for k, v in triage_counts.items() if pd.notna(k)},
            "missing_arrival_time": int(_blank(frame["arrival_time"]).sum()),
            "missing_wait_time_minutes": int(_blank(frame["wait_time_minutes"]).sum()),
            "exact_duplicate_rows_including_all_members": int(duplicate_flag.sum()),
            "exact_duplicate_excess_rows": duplicate_excess,
            "exact_duplicate_groups": duplicate_groups,
            "invalid_year": int(output["year_status"].eq("invalid").sum()),
            "invalid_visit_month": int(output["visit_month_status"].eq("invalid").sum()),
            "invalid_day_of_week": int(output["day_of_week_status"].eq("invalid").sum()),
            "invalid_arrival_time": int(output["arrival_time_status"].eq("invalid").sum()),
            "invalid_triage": int(output["triage_status"].eq("invalid").sum()),
            "invalid_wait_time": int(output["wait_time_status"].str.startswith("invalid").sum()),
        },
        "rules": {
            "row_handling": "All source rows remain in original order. Exact duplicates are flagged and not deleted.",
            "arrival_time": "Whole-number HHMM values are valid only for hours 00-23 and minutes 00-59. Missing and invalid values are not imputed.",
            "day_of_week": "Codes 1-7 are retained. Weekday names are not assigned because the label mapping is unverified.",
            "triage": "Codes 1-5 are retained without adding unverified category labels. Missing or unexpected codes remain explicit.",
            "wait_time": "Nonnegative numeric values, including zero and extremes, are retained. No clipping or imputation is applied. This field is not provider service time.",
            "age": "Nonnegative integer values through 130 are parsed only for usability. Meanings such as infant or top-coded age are not inferred from this third-party source.",
        },
        "limitations": [
            "The dataset is third-party processed and its transformation provenance is incomplete.",
            "Records appear restricted to valid triage acuity levels.",
            "Survey weights and hospital identifiers are unavailable.",
            "Complete patient arrival dates and verified weekday labels are unavailable.",
            "Provider service times, ED length of visit, and hospital capacity are unavailable.",
            "Sample arrival distributions are not Houston-specific observed arrival rates.",
            "The output must not be joined directly to NOAA, FEMA, or HURDAT2 patient-level data.",
        ],
    }
    return output, metadata


def write_outputs(
    output: pd.DataFrame,
    metadata: dict[str, Any],
    source_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
) -> tuple[Path, Path]:
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    report = Path(report_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(destination, index=False, lineterminator="\n", encoding="utf-8", float_format="%.15g")
    final = dict(metadata)
    final["source"] = {
        "path": source.as_posix(),
        "sha256": sha256(source),
        "size_bytes": source.stat().st_size,
        "provenance": "User-provided third-party processed file; transformation provenance incomplete.",
    }
    final["output"] = {
        "path": destination.as_posix(),
        "sha256": sha256(destination),
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
    }
    report.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination, report


def run_pipeline(
    source_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    frame = load_source(source_path)
    output, metadata = preprocess(frame)
    write_outputs(output, metadata, source_path, output_path, report_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.report)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
