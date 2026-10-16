"""Codebook-aware preprocessing for selected 2022 NHAMCS ED fields.

Raw variables are retained. Typed values and reason/category columns are added
for the six requested fields. No NHAMCS sentinel is replaced globally.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Support both ``python -m`` and direct-script execution from the repo root.
    from src.data_processing.nhamcs_2022 import DEFAULT_INPUT, REPO_ROOT, _stata_missing_code, load_nhamcs
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI invocation
    from nhamcs_2022 import DEFAULT_INPUT, REPO_ROOT, _stata_missing_code, load_nhamcs

DEFAULT_CLEANED = REPO_ROOT / "data/processed/nhamcs_2022_cleaned.csv"
DEFAULT_TRIAGE = REPO_ROOT / "data/processed/nhamcs_2022_triage_distribution.csv"
DEFAULT_METADATA = REPO_ROOT / "reports/nhamcs_2022_preprocessing.json"
CODEBOOK_URL = "https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHAMCS/doc22-ed-508.pdf"

TRIAGE_LABELS = {
    0: "no_triage_for_visit",
    1: "immediate",
    2: "emergent",
    3: "urgent",
    4: "semi_urgent",
    5: "nonurgent",
    7: "esa_does_not_conduct_triage",
}
ADISP_LABELS = {1: "home_residence", 2: "nursing_home", 3: "transfer_other_facility", 4: "other"}
FIELD_RULES = {
    "IMMEDR": {"-9": "blank", "-8": "unknown"},
    "WAITTIME": {"-9": "blank", "-7": "not_applicable_not_seen_by_provider"},
    "LOV": {"-9": "blank"},
    "ARRTIME": {"-9": "blank"},
    "AGE": {},  # codebook valid values: 0 (under one year), 1-93, 94 (94+)
    "ADISP": {"-9": "blank", "-8": "unknown", "-7": "not_applicable_not_eligible_live_discharge"},
}


def _code(value: Any) -> str | None:
    special = _stata_missing_code(value)
    if special is not None:
        return special
    if value is None or value is pd.NA or value is pd.NaT:
        return "null"
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {".", *(f".{chr(c)}" for c in range(ord("a"), ord("z") + 1))}:
            return stripped
        if stripped.startswith("-") and stripped[1:].isdigit():
            return stripped
    if isinstance(value, (int, float)) and float(value).is_integer() and value < 0:
        return str(int(value))
    return None


def _number(value: Any) -> float | None:
    if _code(value) is not None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if pd.notna(result) else None


def _missing_reason(variable: str, value: Any) -> str | None:
    code = _code(value)
    if code is None:
        return None
    if code.startswith(".") and len(code) <= 2:
        return f"stata_extended_missing_{code[1:] or 'dot'}"
    if code == "null":
        return "source_null"
    return FIELD_RULES[variable].get(code, f"undocumented_code_{code}")


def _raw_csv_value(value: Any) -> Any:
    """Serialize Stata missing objects as their original marker token."""
    special = _stata_missing_code(value)
    return special if special is not None else value


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """CSV has no stored types; record a per-column read contract."""
    return [{
        "name": str(name), "pandas_dtype": str(series.dtype),
        "csv_read_dtype": "string" if series.dtype == object else str(series.dtype),
        "null_count": int(series.isna().sum()),
        "empty_string_count": int(series.eq("").sum()),
        "csv_null_count": int((series.isna() | series.eq("")).sum()),
    } for name, series in frame.items()]


def _provenance(source_path: str | Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": None, "sha256": None, "archive_member": None,
        "origin": "Caller-provided DataFrame; source file not supplied.",
        "official_dataset_page": "https://www.cdc.gov/nchs/nhamcs/documentation/index.html",
        "download_url": None, "download_date": None,
        "note": "Download URL/date are not inferred from a local filename.",
    }
    if source_path is not None:
        source = Path(source_path).resolve()
        result.update(path=source.as_posix(), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      size_bytes=source.stat().st_size, origin="User-provided local source file.")
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                members = [n for n in archive.namelist() if Path(n).suffix.lower() in {".dta", ".csv"}]
            result["archive_member"] = members[0] if len(members) == 1 else None
    return result


def preprocess_nhamcs(frame: pd.DataFrame, source_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return augmented records, triage distribution, and output metadata."""
    output = frame.copy()
    missing_required = [name for name in FIELD_RULES if name not in frame.columns]
    if missing_required:
        raise ValueError(f"Required NHAMCS fields missing: {', '.join(missing_required)}")

    # pandas CSV export cannot serialize pandas StataMissingValue objects; retain
    # all original markers as literal token strings in the raw source columns.
    for name in output.columns:
        if output[name].dtype == object:
            output[name] = output[name].map(_raw_csv_value)

    reasons: dict[str, pd.Series] = {}
    numeric: dict[str, pd.Series] = {}
    derived: dict[str, pd.Series] = {}
    for variable in FIELD_RULES:
        reasons[variable] = frame[variable].map(lambda v, n=variable: _missing_reason(n, v))
        numeric[variable] = frame[variable].map(_number)
        derived[f"{variable}_missing_reason"] = reasons[variable]

    # Numeric typed fields; values outside documented sentinels are kept for
    # downstream validation rather than silently dropped or clipped.
    derived["IMMEDR_clean"] = numeric["IMMEDR"].astype("Float64")
    derived["IMMEDR_category"] = numeric["IMMEDR"].map(
        lambda v: TRIAGE_LABELS.get(int(v)) if v is not None and float(v).is_integer() else None
    )
    derived["WAITTIME_minutes"] = numeric["WAITTIME"].astype("Float64")
    derived["LOV_minutes"] = numeric["LOV"].astype("Float64")
    derived["AGE_years"] = numeric["AGE"].astype("Float64")
    derived["AGE_category_note"] = numeric["AGE"].map(
        lambda v: "under_1_year" if v == 0 else "94_years_or_older_topcoded" if v == 94 else None
    )
    derived["ADISP_clean"] = numeric["ADISP"].astype("Float64")
    derived["ADISP_category"] = numeric["ADISP"].map(
        lambda v: ADISP_LABELS.get(int(v)) if v is not None and float(v).is_integer() else None
    )

    def _arrival_text(value: Any) -> str:
        if _code(value) is not None:
            return ""
        if isinstance(value, (int, float)) and float(value).is_integer() and 0 <= value <= 2359:
            return f"{int(value):04d}"
        text = str(value).strip()
        return text.zfill(4) if text.isdigit() and len(text) < 4 else text

    arrival_strings = frame["ARRTIME"].map(_arrival_text)
    arrival_valid = arrival_strings.map(lambda s: len(s) == 4 and s.isdigit() and int(s[:2]) <= 23 and int(s[2:]) <= 59)
    derived["ARRTIME_HHMM"] = arrival_strings.where(arrival_valid, pd.NA)
    derived["ARRTIME_missing_reason"] = reasons["ARRTIME"].where(reasons["ARRTIME"].notna(), pd.NA)
    derived["ARRTIME_missing_reason"] = derived["ARRTIME_missing_reason"].mask(
        ~arrival_valid & reasons["ARRTIME"].isna(), "invalid_hhmm"
    )
    derived["arrival_hour"] = arrival_strings.where(arrival_valid).map(
        lambda s: int(s[:2]) if isinstance(s, str) else pd.NA
    ).astype("Int64")
    output = pd.concat([output, pd.DataFrame(derived, index=frame.index)], axis=1)

    triage_numeric = numeric["IMMEDR"]
    urgency_mask = triage_numeric.map(lambda v: v is not None and float(v).is_integer() and int(v) in {1, 2, 3, 4, 5})
    urgency_denominator = int(urgency_mask.sum())
    # Include all defined categories, including zero-count categories, and keep
    # special/invalid/missing categories visible rather than treating them as urgency.
    triage_rows: list[dict[str, Any]] = []
    categories: list[tuple[str, pd.Series]] = [(label, triage_numeric.map(lambda v, c=code: v == c)) for code, label in TRIAGE_LABELS.items()]
    for reason in ["unknown", "blank", "source_null", "stata_extended_missing_dot", *[f"stata_extended_missing_{chr(c)}" for c in range(ord("a"), ord("z") + 1)]]:
        categories.append((reason, reasons["IMMEDR"].eq(reason)))
    # Codebook item 34 does NOT define -7. Count every value outside the
    # explicitly recognized categories, including undocumented negative codes.
    accounted = pd.Series(False, index=frame.index)
    for _, mask in categories:
        accounted |= mask
    categories.append(("other_or_undocumented", ~accounted))

    weighted_ok = all(col in frame.columns for col in ("PATWT", "CSTRATM", "CPSUM"))
    # Promote a separate calculation series, preserving source PATWT and its dtype.
    weights = pd.to_numeric(frame["PATWT"], errors="coerce").astype("float64") if weighted_ok else pd.Series(float("nan"), index=frame.index)
    design_ok = weighted_ok and weights.map(math.isfinite).all() and (weights >= 0).all() and (weights > 0).any() and frame["CSTRATM"].notna().all() and frame["CPSUM"].notna().all()
    weighted_total = math.fsum(weights) if design_ok else None
    weighted_urgency_total = math.fsum(weights[urgency_mask]) if design_ok else None
    for category, mask in categories:
        count = int(mask.sum())
        weight_sum = math.fsum(weights[mask]) if design_ok else None
        is_urgency = category in {"immediate", "emergent", "urgent", "semi_urgent", "nonurgent"}
        triage_rows.append({
            "triage_category": category,
            "unweighted_count": count,
            "unweighted_percent_all_records": count / len(frame) * 100 if len(frame) else None,
            "unweighted_percent_urgency_levels_1_to_5": count / urgency_denominator * 100 if is_urgency and urgency_denominator else None,
            "survey_weighted_estimated_visits": weight_sum,
            "survey_weighted_percent_all_records": weight_sum / weighted_total * 100 if design_ok and weighted_total else None,
            "survey_weighted_percent_urgency_levels_1_to_5": weight_sum / weighted_urgency_total * 100 if design_ok and weighted_urgency_total and is_urgency else None,
        })
    distribution = pd.DataFrame(triage_rows)
    # Keep the schema numeric even when weighted estimates are unavailable.
    for column in distribution.columns.difference(["triage_category", "unweighted_count"]):
        distribution[column] = distribution[column].astype("float64")
    metadata = {
        "dataset": "NHAMCS 2022 ED public-use file",
        "source": _provenance(source_path),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "platform": platform.platform(),
                     "preprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "codebook_url": CODEBOOK_URL,
        "codebook_references": {
            "ARRTIME": "Codebook item 3, page 32: -9 blank; HHMM 0000-2359.",
            "WAITTIME": "Codebook item 4, page 32: -9 blank; -7 not applicable/not seen by provider; 0-1280 minutes reported range.",
            "LOV": "Codebook item 5, page 32: -9 blank; length of ED visit in minutes; 0-5722 reported range.",
            "AGE": "Codebook item 6, page 33: 0 under one year; 1-93; 94 is 94 years or older (top-coded).",
            "IMMEDR": "Codebook item 34, page 36: -9 blank; -8 unknown; 0 no triage this visit; 1-5 urgency; 7 ESA does not conduct triage.",
            "ADISP": "Codebook item 251, page 53: -9 blank; -8 unknown; -7 not applicable; 1-4 disposition of live discharges.",
            "survey_weights": "Section I.H-I, pages 22-23 and Appendix I variance examples, page 102: PATWT for visit estimates; design variables CSTRATM and CPSUM for variance estimation.",
        },
        "records": int(len(frame)),
        "cleaned_output": {
            "path": str(DEFAULT_CLEANED.relative_to(REPO_ROOT)),
            "row_count": int(len(output)), "schema": _schema(output),
            "denominator": "All source visit records; no rows dropped.",
            "exclusions": "No records excluded. Sentinel/special missing codes map to null in typed columns and remain represented in *_missing_reason; original variables are retained.",
            "limitations": "LOS remains unchanged and is hospital length of stay; LOV_minutes is ED length of visit. Values outside codebook ranges are retained for validation, not clipped. Stata extended missings are preserved as marker strings in raw source columns and named in reason fields.",
        },
        "triage_distribution_output": {
            "path": str(DEFAULT_TRIAGE.relative_to(REPO_ROOT)),
            "row_count": int(len(distribution)), "schema": _schema(distribution),
            "record_counts": {"input": len(frame), "accounted": int(distribution.unweighted_count.sum()),
                              "urgency_levels_1_to_5": urgency_denominator,
                              "excluded_from_urgency_denominator": len(frame) - urgency_denominator,
                              "unexpected": int((~accounted).sum()), "dropped": 0},
            "denominator": f"All {len(frame)} source visit records for all-record percentages and weighted estimated visit totals. Acuity-only percentages use {urgency_denominator} records with valid urgency codes 1-5; categories 0 and 7 (non-triage), unknown, blank, source null, extended-missing, and undocumented values are excluded from that denominator but retained as separate rows.",
            "exclusions": "No records dropped from the file; codes 0 and 7 remain separate non-triage categories; unknown/missing values remain separate rows.",
            "weighting": {
                "weight": "PATWT",
                "design_variables": ["CSTRATM", "CPSUM"],
                "computed": bool(design_ok),
                "method": "Convert a separate PATWT series to float64; math.fsum for every category sum and both denominators; percentages are 100 * category sum / denominator. CSTRATM/CPSUM are not numerical weights.",
                "all_records_weight_sum": weighted_total,
                "urgency_weight_sum": weighted_urgency_total,
                "precision": "Float64 accumulation cannot recover precision absent from source weights. Raw PATWT remains unchanged; CSV float32 decimals round-trip at source dtype, but default float64 CSV reads may differ slightly from the original binary values.",
                "reliability_established": False,
                "limitations": "Descriptive annual national visit estimates only. No design-based variance or NCHS reliability assessment. Do not infer population absence from zero-count diagnostic categories. CDC advises against seasonal estimates in 2022 and cautions about MSA/nonresponse. Requiring design identifiers is a conservative availability gate, not proof of survey reliability.",
                "reason_unavailable": None if design_ok else "Required PATWT/CSTRATM/CPSUM not present or weights/design identifiers are invalid; weighted columns are left null, not replaced by unweighted values.",
                "variance": "No standard errors or confidence intervals are produced; use complex-survey methods with the documented design for variance estimation.",
            },
        },
        "variable_rules": FIELD_RULES,
        "unexpected_triage_policy": "IMMEDR=-7 has no meaning assigned in the 2022 codebook. All unexpected codes/values are retained and counted in other_or_undocumented, outside the levels 1-5 denominator.",
        "csv_read_contract": {
            "encoding": "utf-8", "keep_default_na": False, "na_values": [""],
            "note": "Use schema csv_read_dtype per column. Read object/mixed fields and HHMM as strings. CSV blanks combine source empty strings and nulls: schema separates null_count and empty_string_count; csv_null_count is the count under this read contract. For exact lexical preservation use dtype=str, keep_default_na=False without na_values. Literal Stata tokens . and .a-.z remain strings. Empty reason/category fields mean no assigned reason/category.",
        },
        "variable_schema": {
            "IMMEDR": {"meaning": "triage acuity", "typed_columns": ["IMMEDR_clean", "IMMEDR_category", "IMMEDR_missing_reason"]},
            "WAITTIME": {"meaning": "minutes to first provider contact", "typed_columns": ["WAITTIME_minutes", "WAITTIME_missing_reason"]},
            "LOV": {"meaning": "ED length of visit in minutes", "typed_columns": ["LOV_minutes", "LOV_missing_reason"]},
            "ARRTIME": {"meaning": "arrival time in military HHMM", "typed_columns": ["ARRTIME_HHMM", "arrival_hour", "ARRTIME_missing_reason"]},
            "AGE": {"meaning": "age in years; 0 is under one year and 94 is 94 years or older, top-coded", "typed_columns": ["AGE_years", "AGE_category_note", "AGE_missing_reason"]},
            "ADISP": {"meaning": "disposition among eligible live hospital discharges", "typed_columns": ["ADISP_clean", "ADISP_category", "ADISP_missing_reason"]},
            "LOS": {"meaning": "hospital length of stay; retained unchanged and not used as ED length of visit", "typed_columns": []},
        },
        "triage_csv_schema": [
            "triage_category", "unweighted_count", "unweighted_percent_all_records",
            "unweighted_percent_urgency_levels_1_to_5", "survey_weighted_estimated_visits",
            "survey_weighted_percent_all_records", "survey_weighted_percent_urgency_levels_1_to_5",
        ],
        "raw_data_modified": False,
    }
    return output, distribution, metadata


def write_outputs(cleaned: pd.DataFrame, triage: pd.DataFrame, metadata: dict[str, Any], cleaned_path: str | Path = DEFAULT_CLEANED, triage_path: str | Path = DEFAULT_TRIAGE, metadata_path: str | Path = DEFAULT_METADATA) -> tuple[Path, Path, Path]:
    destinations = (Path(cleaned_path), Path(triage_path), Path(metadata_path))
    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(metadata)
    metadata["cleaned_output"]["path"] = destinations[0].resolve().as_posix()
    metadata["triage_distribution_output"]["path"] = destinations[1].resolve().as_posix()
    metadata["metadata_output"] = {"path": destinations[2].resolve().as_posix(), "format": "JSON", "schema_version": 2}
    cleaned.to_csv(destinations[0], index=False)
    triage.to_csv(destinations[1], index=False)
    with destinations[2].open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destinations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess selected NHAMCS 2022 fields using the official codebook.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cleaned-output", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--triage-output", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"NHAMCS input not found: {args.input}")
    cleaned, triage, metadata = preprocess_nhamcs(load_nhamcs(args.input), source_path=args.input)
    paths = write_outputs(cleaned, triage, metadata, args.cleaned_output, args.triage_output, args.metadata_output)
    print(f"Preprocessed {len(cleaned)} NHAMCS records; wrote: {', '.join(str(p) for p in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
