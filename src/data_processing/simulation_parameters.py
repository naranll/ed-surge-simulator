"""Export unweighted sample distributions from processed NHAMCS visits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/nhamcs_2018_22_simulation_visits.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/params.json"


def _distribution(frame: pd.DataFrame, column: str, status: str, codes: list[int]) -> dict:
    statuses = frame[status]
    if not statuses.isin(["valid", "missing", "invalid"]).all():
        raise ValueError(f"Unexpected or missing status in {status}")
    eligible = statuses.eq("valid")
    numbers = pd.to_numeric(frame.loc[eligible, column], errors="coerce")
    if not numbers.isin(codes).all():
        raise ValueError(f"{column} contains out-of-range, missing or noninteger values marked valid")
    denominator = int(eligible.sum())
    if denominator == 0:
        raise ValueError(f"No valid records for {column}; cannot normalize a distribution")
    counts = [int(numbers.eq(code).sum()) for code in codes]
    return {
        "denominator": denominator,
        "excluded_records": int(len(frame) - denominator),
        "excluded_by_status": {s: int(statuses.eq(s).sum()) for s in ["missing", "invalid"]},
        "counts": counts,
        "proportions": [count / denominator for count in counts],
    }


def build_parameters(frame: pd.DataFrame) -> dict:
    required = ["arrival_hour", "arrival_time_status", "triage_acuity_code", "triage_status"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Required processed columns missing: {', '.join(missing)}")
    arrivals = _distribution(frame, "arrival_hour", "arrival_time_status", list(range(24)))
    triage = _distribution(frame, "triage_acuity_code", "triage_status", list(range(1, 6)))
    return {
        "schema_version": "1.0",
        "source": {
            "dataset": "Third-party processed NHAMCS 2018-2022",
            "records": int(len(frame)),
            "provenance_status": "incomplete_third_party_transformation_provenance",
            "weighting": "unweighted_sample",
            "pooling": "all_source_years; duplicate rows retained",
        },
        "hourly_arrivals": {"hours": list(range(24)), **arrivals},
        "triage": {"codes": list(range(1, 6)), **triage},
        "unavailable_parameters": {
            name: {
                "value": None,
                "status": "unavailable_from_primary_dataset",
                "reason": reason,
                "owner": "Simulation Team",
            }
            for name, reason in {
                "provider_service_times": "No verified provider care/service duration field; waiting time is not service time.",
                "admission_probabilities": "No verified admission outcome field.",
                "boarding_durations": "No verified boarding duration field.",
            }.items()
        },
        "limitations": [
            "Relative sample distributions, not observed Houston rates or absolute arrivals per hour.",
            "Survey weights and hospital identifiers are unavailable; records appear restricted to valid triage acuity.",
            "Complete arrival dates and verified weekday labels are unavailable; no named weekday factors are calculated.",
            "Triage codes are preserved without adding clinical labels; their third-party mapping is not independently verified.",
            "Simulation Team must supply baseline volume, service/capacity assumptions and unavailable parameters separately.",
            "No waiting-time substitution, borrowing from NHAMCS 2022, patient-weather joins or scenario multipliers.",
        ],
    }


def export_parameters(source_path: str | Path = DEFAULT_INPUT,
                      output_path: str | Path = DEFAULT_OUTPUT) -> dict:
    source, destination = Path(source_path).resolve(), Path(output_path).resolve()
    raw_root = (ROOT / "data/raw").resolve()
    if destination == source or raw_root == destination or raw_root in destination.parents:
        raise ValueError("Parameter destination must be distinct from source and outside data/raw")
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError("Refusing to overwrite an unrelated output") from None
        if not isinstance(existing, dict) or not {"schema_version", "hourly_arrivals", "triage", "unavailable_parameters"}.issubset(existing):
            raise ValueError("Refusing to overwrite an unrelated output")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    result = build_parameters(frame)
    result["source"].update(file_name=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(text.encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = export_parameters(args.input, args.output)
    print(f"Wrote {args.output}: {result['hourly_arrivals']['denominator']} arrivals; "
          f"{result['triage']['denominator']} triage records")


if __name__ == "__main__":
    main()
