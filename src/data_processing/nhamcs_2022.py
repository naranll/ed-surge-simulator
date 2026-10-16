"""Ingest and audit 2022 NHAMCS ED public-use data; do not clean or recode it.

Variable names and values are reported as found. Consult the official codebook
before interpreting any candidate field or numeric code.

CLI::

    python src/data_processing/nhamcs_2022.py \
        --input data/raw/nhamcs/ed2022-stata.zip

CSV and uncompressed Stata files are accepted too. ZIP input is read in place
without extracting or modifying the raw archive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/raw/nhamcs/ed2022-stata.zip"
DEFAULT_REPORT = REPO_ROOT / "reports/nhamcs_2022_audit.json"

# LOV, LOS and ADISP follow the 2022 ED codebook. Other aliases remain
# name-search hints; retain the existing descriptive audit/report structure.
FIELD_NAME_HINTS = {
    "triage_urgency": ("immedr", "triage", "acuity", "urgency"),
    "waiting_time": ("waittime", "wait_time", "waiting"),
    "length_of_visit": ("lov", "length", "visit_length"),
    "hospital_length_of_stay": ("los",),
    "arrival_time": ("arrtime", "arrival", "arr_time"),
    "age": ("age",),
    "disposition": ("adisp", "disp", "disposition"),
}
DATE_NAME_HINTS = ("date", "day", "month", "year")


def _stata_missing_code(value: Any) -> str | None:
    """Return Pandas' preserved Stata missing token ('.', '.a' ... '.z')."""
    if type(value).__name__ == "StataMissingValue" and hasattr(value, "string"):
        return str(value.string)
    return None


def _json_value(value: Any) -> Any:
    """Convert common Pandas/numpy values into strict JSON-compatible values."""
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


def _safe_value_key(value: Any) -> str:
    """Create a hashable, type-aware key without changing the source value."""
    stata_code = _stata_missing_code(value)
    if stata_code is not None:
        payload = {"type": "StataMissingValue", "code": stata_code}
    else:
        payload = {"type": type(value).__name__, "value": _json_value(value)}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _missing_scalar(value: Any) -> bool:
    if _stata_missing_code(value) is not None:
        return True
    try:
        marker = pd.isna(value)
        return bool(marker) if getattr(marker, "ndim", 0) == 0 else False
    except (TypeError, ValueError):
        return False


def _nhamcs_missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: _stata_missing_code(value) is not None)


def _safe_value_counts(series: pd.Series, limit: int = 30) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    display_values: dict[str, Any] = {}
    for value in series:
        key = _safe_value_key(value)
        counts[key] += 1
        display_values.setdefault(key, _json_value(value))
    return [
        {"value": display_values[key], "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _exact_duplicate_count(frame: pd.DataFrame) -> int:
    """Count identical rows without hashing Pandas StataMissingValue objects."""
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in frame.itertuples(index=False, name=None):
        key = tuple(_safe_value_key(value) for value in row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _load_stata(source: Any) -> pd.DataFrame:
    # Preserve Stata value labels as raw codes and retain extended missing
    # markers where Pandas exposes them. Their meanings remain codebook work.
    return pd.read_stata(source, convert_categoricals=False, convert_missing=True)


def load_nhamcs(path: str | Path) -> pd.DataFrame:
    """Load an NHAMCS 2022 CSV, DTA, or ZIP containing exactly one such file."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, low_memory=False)
    if suffix == ".dta":
        return _load_stata(source)
    if suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            candidates = [
                name for name in archive.namelist()
                if Path(name).suffix.lower() in {".dta", ".csv"} and not name.endswith("/")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one .dta or .csv file in {source.name}; "
                    f"found {len(candidates)}."
                )
            member = candidates[0]
            with archive.open(member) as file_obj:
                if Path(member).suffix.lower() == ".dta":
                    return _load_stata(file_obj)
                return pd.read_csv(file_obj, low_memory=False)
    raise ValueError("NHAMCS input must be a .dta, .csv, or ZIP containing one .dta/.csv file.")


def _column_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = []
    for name in frame.columns:
        series = frame[name]
        stata_missing = Counter(
            _stata_missing_code(value) for value in series
            if _stata_missing_code(value) is not None
        )
        missing_count = int(_nhamcs_missing_mask(series).sum())
        distinct_keys = {
            _safe_value_key(value) for value in series
            if not _missing_scalar(value)
        }
        columns.append({
            "name": str(name),
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "missing_count": missing_count,
            "missing_percent": round(missing_count / len(series) * 100, 4) if len(series) else 0.0,
            "distinct_non_missing": len(distinct_keys),
            "stata_extended_missing_values": [
                {
                    "code": code,
                    "count": int(count),
                    "meaning": "Stata extended missing marker; verify any NHAMCS-specific interpretation in the codebook.",
                }
                for code, count in sorted(stata_missing.items())
            ],
        })
    return columns


def _numeric_distributions(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in frame.columns:
        source = frame[name]
        missing = _nhamcs_missing_mask(source)
        values = pd.to_numeric(source, errors="coerce")
        observed_source = int((~missing).sum())
        # Extended Stata missings can force an otherwise numeric variable to
        # object dtype. Include columns that are numeric or predominantly
        # numeric after a non-mutating audit conversion.
        convertible = int(values.notna().sum())
        if not pd.api.types.is_numeric_dtype(source.dtype) and (
            observed_source == 0 or convertible / observed_source < 0.9
        ):
            continue
        describe = values.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        result[str(name)] = {
            "source_dtype": str(source.dtype),
            "numeric_parse_failures": int((~missing & values.isna()).sum()),
            "distribution": {str(key): _json_value(value) for key, value in describe.items()},
        }
    return result


def _date_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    candidates = [
        str(name) for name in frame.columns
        if any(hint in str(name).lower() for hint in DATE_NAME_HINTS)
    ]
    result: dict[str, Any] = {
        "candidate_columns": candidates,
        "parsed_columns": {},
        "note": "Candidate date-like names are not confirmed date variables; verify in the codebook.",
    }
    for name in candidates:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            result["parsed_columns"][name] = {
                "status": "not_parsed_numeric_ambiguous",
                "note": "Numeric date representation requires codebook confirmation.",
            }
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        valid = parsed.dropna()
        result["parsed_columns"][name] = {
            "status": "candidate_parse_only",
            "valid_count": int(len(valid)),
            "invalid_or_missing_count": int(parsed.isna().sum()),
            "minimum": valid.min().isoformat() if len(valid) else None,
            "maximum": valid.max().isoformat() if len(valid) else None,
        }
    if not candidates:
        result["note"] = "No date-like column name found; date coverage cannot be assessed from names alone."
    return result


def _candidate_fields(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = [(str(name), name) for name in frame.columns]
    for concept, hints in FIELD_NAME_HINTS.items():
        matched = []
        for display_name, original_name in names:
            folded = display_name.lower().replace(" ", "_")
            matched_name = any(
                hint in folded if len(hint) > 4 else hint in folded.split("_")
                for hint in hints
            )
            if matched_name:
                series = frame[original_name]
                frequencies = series.value_counts(dropna=False).head(30)
                matched.append({
                    "column": display_name,
                    "dtype": str(series.dtype),
                    "missing_count": int(_nhamcs_missing_mask(series).sum()),
                    "sample_value_counts": _safe_value_counts(series),
                    "codebook_verification_required": True,
                })
        result[concept] = matched
    return result


def audit_nhamcs(frame: pd.DataFrame, input_path: str | Path | None = None) -> dict[str, Any]:
    """Return a descriptive audit without changing or interpreting source values."""
    report = {
        "dataset": "NHAMCS 2022 Emergency Department public-use data",
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": Path(input_path).name if input_path else None,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "column_names": [str(name) for name in frame.columns],
        "columns": _column_audit(frame),
        "exact_duplicate_row_count": _exact_duplicate_count(frame),
        "date_coverage": _date_coverage(frame),
        "numeric_distributions": _numeric_distributions(frame),
        "candidate_fields_requiring_codebook_verification": _candidate_fields(frame),
        "interpretation_policy": (
            "No undocumented missing-value codes or field meanings are assumed. "
            "Candidate fields and their observed values require official codebook review."
        ),
        "stata_missing_value_policy": (
            "Pandas StataMissingValue objects are counted as missing and their Stata tokens "
            "('.', '.a' through '.z') are reported by column; no NHAMCS-specific meaning is inferred."
        ),
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
    parser = argparse.ArgumentParser(description="Ingest and audit NHAMCS 2022 ED data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"CSV, DTA, or ZIP input (default: {DEFAULT_INPUT})")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help=f"JSON audit destination (default: {DEFAULT_REPORT})")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"NHAMCS input not found: {args.input}")
    frame = load_nhamcs(args.input)
    report = audit_nhamcs(frame, args.input)
    destination = write_audit_report(report, args.report)
    print(f"NHAMCS audit: {report['row_count']} rows x {report['column_count']} columns")
    print(f"Report written to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
