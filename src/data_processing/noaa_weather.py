"""Ingest and audit NOAA GHCN-Daily CSV data for Houston station USW00012960.

This phase reports source values only. It does not clean, convert units,
interpolate dates, or calculate heatwave indicators.

CLI::

    python src/data_processing/noaa_weather.py \
        --input data/raw/noaa/USW00012960.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/raw/noaa/USW00012960.csv"
DEFAULT_REPORT = REPO_ROOT / "reports/noaa_usw00012960_audit.json"
EXPECTED_STATION = "USW00012960"
OBSERVATION_FIELDS = ("TMAX", "TMIN", "PRCP")
METADATA_FIELDS = ("STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME")


def load_noaa_csv(path: str | Path) -> pd.DataFrame:
    """Read NOAA's station CSV while only treating empty fields as missing.

    NOAA observation values and any nonempty missing/value codes are left
    untouched for audit and source documentation.
    """
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError("NOAA GHCN-Daily input must be a .csv file.")
    return pd.read_csv(source, low_memory=False, keep_default_na=False, na_values=[""])


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
            return None
        return value
    return str(value)


def _missing_mask(series: pd.Series) -> pd.Series:
    blank = series.astype("string").str.strip().eq("").fillna(False)
    return series.isna() | blank


def _column_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = []
    for name in frame.columns:
        series = frame[name]
        missing_count = int(_missing_mask(series).sum())
        columns.append({
            "name": str(name),
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "missing_count": missing_count,
            "missing_percent": round(missing_count / len(series) * 100, 4) if len(series) else 0.0,
            "distinct_non_missing": int(series.loc[~_missing_mask(series)].nunique(dropna=True)),
        })
    return columns


def _date_continuity(frame: pd.DataFrame) -> dict[str, Any]:
    if "DATE" not in frame.columns:
        return {"status": "not_assessed", "reason": "Required DATE column is absent."}
    parsed = pd.to_datetime(frame["DATE"], errors="coerce", format="%Y-%m-%d")
    valid = parsed.dropna().dt.normalize()
    if valid.empty:
        return {
            "status": "not_assessed",
            "invalid_or_missing_date_count": int(parsed.isna().sum()),
            "reason": "No valid ISO YYYY-MM-DD dates were found.",
        }
    start, end = valid.min(), valid.max()
    expected = pd.date_range(start, end, freq="D")
    observed = pd.DatetimeIndex(valid.unique()).sort_values()
    gaps = expected.difference(observed)
    return {
        "status": "assessed",
        "minimum_date": start.date().isoformat(),
        "maximum_date": end.date().isoformat(),
        "valid_date_row_count": int(len(valid)),
        "unique_observed_dates": int(len(observed)),
        "expected_daily_dates_inclusive": int(len(expected)),
        "missing_calendar_date_count": int(len(gaps)),
        "first_missing_dates": [day.date().isoformat() for day in gaps[:50]],
        "invalid_or_missing_date_count": int(parsed.isna().sum()),
        "duplicate_date_row_count": int(valid.duplicated().sum()),
        "note": "Continuity counts absent calendar dates; it does not impute or infer observations.",
    }


def _numeric_distribution(series: pd.Series) -> dict[str, Any]:
    raw = series
    if pd.api.types.is_string_dtype(raw.dtype) or raw.dtype == object:
        raw = raw.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    missing = _missing_mask(series)
    parse_failures = (~missing) & numeric.isna()
    summary = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "dtype": str(series.dtype),
        "missing_count": int(missing.sum()),
        "nonempty_non_numeric_count": int(parse_failures.sum()),
        "numeric_distribution": {str(key): _json_value(value) for key, value in summary.items()},
        "raw_value_sample": [_json_value(value) for value in series.dropna().drop_duplicates().head(10)],
    }


def audit_noaa(frame: pd.DataFrame, input_path: str | Path | None = None) -> dict[str, Any]:
    """Return row/schema/missingness/date/value audit without cleaning data."""
    missing_fields = [field for field in OBSERVATION_FIELDS if field not in frame.columns]
    date_coverage = _date_continuity(frame)
    if "STATION" in frame.columns:
        station_values = sorted({str(value) for value in frame["STATION"].dropna().unique()})
        station_match = {
            "expected_station": EXPECTED_STATION,
            "observed_station_ids": station_values,
            "all_rows_match_expected_station": bool(station_values) and station_values == [EXPECTED_STATION],
        }
    else:
        station_values = []
        station_match = {"expected_station": EXPECTED_STATION, "observed_station_ids": None, "all_rows_match_expected_station": None}

    metadata: dict[str, Any] = {}
    for name in METADATA_FIELDS:
        if name in frame.columns:
            values = frame[name].dropna()
            if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
                values = values.astype("string").str.strip()
                values = values.loc[values.ne("")]
            distinct = values.drop_duplicates()
            metadata[name] = {
                "available": True,
                "distinct_value_count": int(len(distinct)),
                "values": [_json_value(value) for value in distinct.head(20)],
            }
        else:
            metadata[name] = {"available": False}

    duplicate_keys = None
    if {"STATION", "DATE"}.issubset(frame.columns):
        duplicate_keys = int(frame.duplicated(subset=["STATION", "DATE"]).sum())

    report = {
        "dataset": "NOAA GHCN-Daily station observations",
        "expected_station": EXPECTED_STATION,
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": Path(input_path).name if input_path else None,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "column_names": [str(name) for name in frame.columns],
        "columns": _column_audit(frame),
        "exact_duplicate_row_count": int(frame.duplicated().sum()),
        "duplicate_station_date_key_count": duplicate_keys,
        "station_check": station_match,
        "station_metadata": metadata,
        "date_continuity": date_coverage,
        "observation_fields": {
            name: _numeric_distribution(frame[name]) if name in frame.columns else {"available": False}
            for name in OBSERVATION_FIELDS
        },
        "missing_observation_fields": missing_fields,
        "value_scale_note": (
            "No unit conversion is performed. GHCN-Daily source documentation defines "
            "TMAX/TMIN and PRCP storage scales; verify the exact source product/column conventions."
        ),
        "heatwave_indicators_calculated": False,
    }
    return report


def write_audit_report(report: dict[str, Any], report_path: str | Path = DEFAULT_REPORT) -> Path:
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False, allow_nan=False)
        file_obj.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest and audit NOAA GHCN-Daily station CSV data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Station CSV input (default: {DEFAULT_INPUT})")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help=f"JSON audit destination (default: {DEFAULT_REPORT})")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"NOAA input not found: {args.input}")
    frame = load_noaa_csv(args.input)
    report = audit_noaa(frame, args.input)
    destination = write_audit_report(report, args.report)
    print(f"NOAA audit: {report['row_count']} rows x {report['column_count']} columns")
    if report["station_check"]["all_rows_match_expected_station"] is False:
        print(f"WARNING: observed station identifiers do not match {EXPECTED_STATION}")
    print(f"Report written to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
