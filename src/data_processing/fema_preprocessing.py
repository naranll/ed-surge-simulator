"""Create the minimal simulation handoff table from a verified offline FEMA snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from .fema_audit import ENTITY, audit_snapshot
except ImportError:  # Direct script execution.
    from fema_audit import ENTITY, audit_snapshot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/raw/fema/harris_20261016/manifest.json"
DEFAULT_OUTPUT = ROOT / "data/processed/fema_harris_declarations.csv"

SOURCE_FIELDS = [
    "id", "femaDeclarationString", "disasterNumber", "incidentId",
    "incidentType", "declarationType", "declarationTitle", "state",
    "fipsStateCode", "fipsCountyCode", "placeCode", "designatedArea",
    "tribalRequest", "declarationDate", "incidentBeginDate",
    "incidentEndDate", "ihProgramDeclared", "iaProgramDeclared",
    "paProgramDeclared", "hmProgramDeclared", "lastRefresh", "hash",
]
DATE_FIELDS = ["declarationDate", "incidentBeginDate", "incidentEndDate"]
OUTPUT_COLUMNS = [
    "output_row", "source_id", "fema_declaration_string", "disaster_number",
    "incident_id", "incident_type", "declaration_type", "declaration_title",
    "state", "fips_state_code", "fips_county_code", "county_fips",
    "place_code", "designated_area", "tribal_request",
    "declaration_date_raw", "declaration_date", "declaration_date_status",
    "incident_begin_date_raw", "incident_begin_date", "incident_begin_date_status",
    "incident_end_date_raw", "incident_end_date", "incident_end_date_status",
    "ih_program_declared", "ia_program_declared", "pa_program_declared",
    "hm_program_declared", "source_last_refresh", "source_record_hash",
    "source_record_sha256", "source_page_file", "source_page_sha256",
    "source_page_retrieved_utc", "source_snapshot_manifest",
    "source_snapshot_sha256", "source_api_url", "source_api_version",
    "source_filter", "geographic_scope", "record_semantics",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_snapshot(manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Load only an independently verified, complete local snapshot."""
    path = Path(manifest_path).resolve()
    audit = audit_snapshot(path)
    failures = [item for item in audit["independent_checks"] if item["status"] != "PASS"]
    if failures:
        raise ValueError(f"Snapshot failed independent checks: {failures}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["_manifest_sha256"] = _sha256(path.read_bytes())
    if manifest.get("status") != "complete":
        raise ValueError("Snapshot manifest is not complete")
    records: list[dict] = []
    for page in manifest["pages"]:
        page_path = (path.parent / page["file"]).resolve()
        if page_path.parent != path.parent:
            raise ValueError("Page path escapes snapshot directory")
        payload = json.loads(page_path.read_text(encoding="utf-8"))
        for index, record in enumerate(payload[ENTITY], start=1):
            if any(field not in record for field in SOURCE_FIELDS):
                missing = [field for field in SOURCE_FIELDS if field not in record]
                raise ValueError(f"Source record lacks required fields: {missing}")
            item = dict(record)
            item["_source_page_file"] = page["file"]
            item["_source_page_sha256"] = page["sha256"]
            item["_source_page_retrieved_utc"] = page["retrieved_utc"]
            item["_source_page_row"] = index
            item["_source_record_sha256"] = _sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
            records.append(item)
    if len(records) != manifest["retrieved_records"]:
        raise ValueError("Loaded record count differs from reconciled manifest count")
    return records, manifest


def _date_parts(value: object) -> tuple[str, str, str]:
    """Return source text, ISO calendar date, and an explicit validation status."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "", "", "missing_source"
    raw = str(value)
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return raw, "", "invalid_source_date"
    return raw, parsed.strftime("%Y-%m-%d"), "valid_source_date"


def preprocess_declarations(records: list[dict], manifest: dict,
                            manifest_path: str | Path) -> pd.DataFrame:
    """Select handoff fields without filtering or consolidating declarations."""
    rows = []
    manifest_path = Path(manifest_path).resolve()
    snapshot_hash = manifest.get("_manifest_sha256")
    if not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64:
        raise ValueError("Verified manifest SHA-256 is required")
    for record in records:
        dates = {field: _date_parts(record[field]) for field in DATE_FIELDS}
        row = {
            "source_id": record["id"],
            "fema_declaration_string": record["femaDeclarationString"],
            "disaster_number": record["disasterNumber"],
            "incident_id": "" if record["incidentId"] is None else str(record["incidentId"]),
            "incident_type": record["incidentType"],
            "declaration_type": record["declarationType"],
            "declaration_title": record["declarationTitle"],
            "state": record["state"],
            "fips_state_code": record["fipsStateCode"],
            "fips_county_code": record["fipsCountyCode"],
            "county_fips": str(record["fipsStateCode"]) + str(record["fipsCountyCode"]),
            "place_code": record["placeCode"],
            "designated_area": record["designatedArea"],
            "tribal_request": record["tribalRequest"],
            "declaration_date_raw": dates["declarationDate"][0],
            "declaration_date": dates["declarationDate"][1],
            "declaration_date_status": dates["declarationDate"][2],
            "incident_begin_date_raw": dates["incidentBeginDate"][0],
            "incident_begin_date": dates["incidentBeginDate"][1],
            "incident_begin_date_status": dates["incidentBeginDate"][2],
            "incident_end_date_raw": dates["incidentEndDate"][0],
            "incident_end_date": dates["incidentEndDate"][1],
            "incident_end_date_status": dates["incidentEndDate"][2],
            "ih_program_declared": record["ihProgramDeclared"],
            "ia_program_declared": record["iaProgramDeclared"],
            "pa_program_declared": record["paProgramDeclared"],
            "hm_program_declared": record["hmProgramDeclared"],
            "source_last_refresh": record["lastRefresh"],
            "source_record_hash": record["hash"],
            "source_record_sha256": record["_source_record_sha256"],
            "source_page_file": record["_source_page_file"],
            "source_page_sha256": record["_source_page_sha256"],
            "source_page_retrieved_utc": record["_source_page_retrieved_utc"],
            "source_snapshot_manifest": str(manifest_path),
            "source_snapshot_sha256": snapshot_hash,
            "source_api_url": manifest["source_url"],
            "source_api_version": manifest["version"],
            "source_filter": manifest["filter"],
            "geographic_scope": "explicit_harris_county",
            "record_semantics": "source_declaration_not_unique_physical_disaster",
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Snapshot contains no declarations")
    if frame["source_id"].isna().any() or frame["source_id"].duplicated().any():
        raise ValueError("Source IDs must be populated and unique")
    frame = frame.sort_values("source_id", kind="mergesort").reset_index(drop=True)
    frame.insert(0, "output_row", pd.Series(range(1, len(frame) + 1), dtype="int64"))
    frame["disaster_number"] = pd.to_numeric(frame["disaster_number"], errors="raise").astype("int64")
    for field in ["tribal_request", "ih_program_declared", "ia_program_declared",
                  "pa_program_declared", "hm_program_declared"]:
        if not frame[field].map(lambda value: isinstance(value, bool)).all():
            raise ValueError(f"{field} must contain source booleans")
        frame[field] = frame[field].astype("boolean")
    for field in ["fips_state_code", "fips_county_code", "county_fips", "place_code"]:
        frame[field] = frame[field].astype("string")
    if not (frame["state"].eq("TX") & frame["fips_state_code"].eq("48")
            & frame["fips_county_code"].eq("201") & frame["county_fips"].eq("48201")
            & frame["designated_area"].eq("Harris (County)")
            & frame["tribal_request"].eq(False)).all():
        raise ValueError("Output contains a non-explicit-Harris geographic record")
    if len(frame) != manifest["retrieved_records"] or set(frame["source_id"]) != {r["id"] for r in records}:
        raise ValueError("Output does not reconcile exactly to source declarations")
    return frame[OUTPUT_COLUMNS]


def write_output(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path).resolve()
    raw_root = (ROOT / "data/raw").resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if raw_root == destination or raw_root in destination.parents:
        raise ValueError("Processed output cannot be written under data/raw")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records, manifest = load_verified_snapshot(args.manifest)
    frame = preprocess_declarations(records, manifest, args.manifest)
    destination = write_output(frame, args.output)
    print(json.dumps({"output": str(destination), "records": len(frame),
                      "unique_source_ids": int(frame.source_id.nunique()),
                      "sha256": _sha256(destination.read_bytes())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
