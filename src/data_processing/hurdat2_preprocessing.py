"""Offline production exports for HURDAT2 tracks and reviewed FEMA candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from .hurdat2_matching_audit import parse_hurdat2, sha256
except ImportError:  # Direct script execution.
    from hurdat2_matching_audit import parse_hurdat2, sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HURDAT_MANIFEST = ROOT / "data/raw/hurdat2/manifest.json"
DEFAULT_MATCH_AUDIT = ROOT / "reports/hurdat2_fema_matching_audit.json"
DEFAULT_TRACK_OUTPUT = ROOT / "data/processed/hurdat2_atlantic_tracks.csv"
DEFAULT_CROSSWALK_OUTPUT = ROOT / "data/processed/fema_hurdat2_candidate_crosswalk.csv"

TRACK_COLUMNS = [
    "track_row", "storm_id", "storm_name", "date_raw", "time_raw",
    "timestamp_utc", "record_identifier", "status", "latitude_raw",
    "longitude_raw", "latitude", "longitude", "maximum_sustained_wind_raw",
    "maximum_sustained_wind_kt", "maximum_sustained_wind_status",
    "central_pressure_raw", "central_pressure_mb", "central_pressure_status",
    "source_line_number", "source_line_sha256", "source_file", "source_url",
    "source_retrieved_utc", "source_file_sha256", "source_manifest",
    "source_manifest_sha256",
]

CROSSWALK_COLUMNS = [
    "crosswalk_row", "fema_source_id", "fema_declaration",
    "fema_declaration_type", "fema_title", "fema_incident_begin",
    "fema_incident_end", "candidate_storm_id", "storm_name", "storm_year",
    "storm_lifetime_start_utc", "storm_lifetime_end_utc", "temporal_overlap",
    "minimum_recorded_distance_to_houston_km", "nearest_recorded_timestamp_utc",
    "geographic_relevance", "confidence", "reason", "relationship_status",
    "distance_semantics", "source_matching_audit", "source_matching_audit_sha256",
    "hurdat2_source_file_sha256",
]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_track_source(manifest_path: str | Path) -> tuple[pd.DataFrame, dict, Path]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = (manifest_path.parent / manifest["file"]).resolve()
    if source.parent != manifest_path.parent:
        raise ValueError("HURDAT2 source escapes its manifest directory")
    if sha256(source) != manifest["sha256"]:
        raise ValueError("HURDAT2 source SHA-256 differs from manifest")
    fixes, storms = parse_hurdat2(source)
    years = [int(storms.year.min()), int(storms.year.max())]
    if years != manifest["covered_years"]:
        raise ValueError("Parsed HURDAT2 years differ from manifest")
    return fixes, manifest, manifest_path


def build_track_export(fixes: pd.DataFrame, manifest: dict,
                       manifest_path: str | Path) -> pd.DataFrame:
    """Retain every parsed track row and raw values; never interpolate."""
    if fixes.empty:
        raise ValueError("No HURDAT2 track rows")
    frame = fixes.copy(deep=True).reset_index(drop=True)
    frame.insert(0, "track_row", pd.Series(range(1, len(frame) + 1), dtype="int64"))
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["maximum_sustained_wind_status"] = frame.maximum_sustained_wind_kt.isna().map(
        {True: "missing_source_sentinel", False: "reported"})
    frame["central_pressure_status"] = frame.central_pressure_mb.isna().map(
        {True: "missing_source_sentinel", False: "reported"})
    manifest_path = Path(manifest_path).resolve()
    frame["source_file"] = manifest["file"]
    frame["source_url"] = manifest["source_url"]
    frame["source_retrieved_utc"] = manifest["retrieved_utc"]
    frame["source_file_sha256"] = manifest["sha256"]
    frame["source_manifest"] = str(manifest_path)
    frame["source_manifest_sha256"] = file_sha256(manifest_path)
    for column in ["maximum_sustained_wind_kt", "central_pressure_mb"]:
        frame[column] = pd.array(frame[column], dtype="Int64")
    if frame.source_line_number.duplicated().any() or frame.track_row.duplicated().any():
        raise ValueError("Track rows do not have unique source/output positions")
    return frame[TRACK_COLUMNS]


def build_crosswalk_export(audit_path: str | Path, hurdat_manifest: dict) -> pd.DataFrame:
    """Export reviewed candidate relationships without promoting confidence."""
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = audit.get("candidate_matches")
    if not isinstance(candidates, list):
        raise ValueError("Matching audit lacks candidate relationships")
    if len(candidates) != 19 or len({row["fema_source_id"] for row in candidates}) != 18:
        raise ValueError("Reviewed crosswalk must contain 19 relationships across 18 declarations")
    frame = pd.DataFrame(candidates)
    if set(frame.confidence) - {"high", "medium", "unresolved"}:
        raise ValueError("Unexpected confidence category")
    frame["confidence"] = frame.confidence.map(
        {"high": "High", "medium": "Medium", "unresolved": "Unresolved"})
    frame["relationship_status"] = "candidate_not_confirmed_physical_event"
    frame["distance_semantics"] = "nearest_recorded_center_to_houston_reference_context_only"
    frame["source_matching_audit"] = str(audit_path)
    frame["source_matching_audit_sha256"] = file_sha256(audit_path)
    frame["hurdat2_source_file_sha256"] = hurdat_manifest["sha256"]
    frame = frame.sort_values(["fema_declaration", "candidate_storm_id"], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "crosswalk_row", pd.Series(range(1, len(frame) + 1), dtype="int64"))
    if frame.duplicated(["fema_source_id", "candidate_storm_id"]).any():
        raise ValueError("Duplicate FEMA/storm candidate relationship")
    return frame[CROSSWALK_COLUMNS]


def write_outputs(tracks: pd.DataFrame, crosswalk: pd.DataFrame,
                  track_output: str | Path, crosswalk_output: str | Path) -> tuple[Path, Path]:
    paths = [Path(track_output).resolve(), Path(crosswalk_output).resolve()]
    raw_root = (ROOT / "data/raw").resolve()
    if paths[0] == paths[1] or any(path.exists() for path in paths):
        raise FileExistsError("Output paths must be distinct and must not already exist")
    if any(raw_root == path or raw_root in path.parents for path in paths):
        raise ValueError("Processed output cannot be written under data/raw")
    for frame, path in zip([tracks, crosswalk], paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
    return paths[0], paths[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hurdat-manifest", type=Path, default=DEFAULT_HURDAT_MANIFEST)
    parser.add_argument("--matching-audit", type=Path, default=DEFAULT_MATCH_AUDIT)
    parser.add_argument("--track-output", type=Path, default=DEFAULT_TRACK_OUTPUT)
    parser.add_argument("--crosswalk-output", type=Path, default=DEFAULT_CROSSWALK_OUTPUT)
    args = parser.parse_args()
    fixes, manifest, manifest_path = load_track_source(args.hurdat_manifest)
    tracks = build_track_export(fixes, manifest, manifest_path)
    crosswalk = build_crosswalk_export(args.matching_audit, manifest)
    track_path, crosswalk_path = write_outputs(tracks, crosswalk, args.track_output, args.crosswalk_output)
    print(json.dumps({
        "tracks": {"path": str(track_path), "rows": len(tracks), "sha256": file_sha256(track_path)},
        "crosswalk": {"path": str(crosswalk_path), "rows": len(crosswalk),
                      "fema_declarations": int(crosswalk.fema_source_id.nunique()),
                      "sha256": file_sha256(crosswalk_path)}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
