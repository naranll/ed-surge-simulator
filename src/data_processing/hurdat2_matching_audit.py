"""Minimal HURDAT2 parser and FEMA candidate-matching audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HURDAT_MANIFEST = ROOT / "data/raw/hurdat2/manifest.json"
DEFAULT_FEMA = ROOT / "data/processed/fema_harris_declarations.csv"
DEFAULT_REPORT = ROOT / "reports/hurdat2_fema_matching_audit.json"
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
RELATED_PATTERN = re.compile(r"HURRICANE|TROPICAL STORM", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(token: str) -> float:
    value = token.strip().upper()
    if not re.fullmatch(r"\d+(?:\.\d+)?[NSEW]", value):
        raise ValueError(f"Invalid HURDAT2 coordinate: {token!r}")
    number = float(value[:-1])
    return -number if value[-1] in "SW" else number


def parse_hurdat2(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse only identifiers, time, position, status, wind, and pressure."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    fixes: list[dict] = []
    storms: list[dict] = []
    index = 0
    while index < len(lines):
        header = [part.strip() for part in lines[index].split(",")]
        if len(header) < 3 or not re.fullmatch(r"AL\d{6}", header[0]):
            raise ValueError(f"Invalid header at line {index + 1}")
        storm_id, storm_name, expected = header[0], header[1], int(header[2])
        index += 1
        first = len(fixes)
        for _ in range(expected):
            if index >= len(lines):
                raise ValueError(f"Truncated storm {storm_id}")
            raw_line = lines[index]
            source_line_number = index + 1
            fields = [part.strip() for part in raw_line.split(",")]
            if len(fields) < 8:
                raise ValueError(f"Invalid track row at line {index + 1}")
            timestamp = pd.to_datetime(fields[0] + fields[1], format="%Y%m%d%H%M", utc=True, errors="raise")
            wind = int(fields[6])
            pressure = int(fields[7])
            fixes.append({"storm_id": storm_id, "storm_name": storm_name,
                          "date_raw": fields[0], "time_raw": fields[1],
                          "timestamp_utc": timestamp, "record_identifier": fields[2],
                          "status": fields[3], "latitude_raw": fields[4],
                          "longitude_raw": fields[5], "latitude": coordinate(fields[4]),
                          "longitude": coordinate(fields[5]),
                          "maximum_sustained_wind_raw": fields[6],
                          "maximum_sustained_wind_kt": pd.NA if wind in (-99, -999) else wind,
                          "central_pressure_raw": fields[7],
                          "central_pressure_mb": pd.NA if pressure == -999 else pressure,
                          "source_line_number": source_line_number,
                          "source_line_sha256": hashlib.sha256(raw_line.encode("ascii")).hexdigest()})
            index += 1
        storm_fixes = fixes[first:]
        storms.append({"storm_id": storm_id, "storm_name": storm_name,
                       "year": int(storm_id[-4:]), "fix_count": len(storm_fixes),
                       "lifetime_start_utc": min(row["timestamp_utc"] for row in storm_fixes),
                       "lifetime_end_utc": max(row["timestamp_utc"] for row in storm_fixes)})
    fix_frame = pd.DataFrame(fixes)
    storm_frame = pd.DataFrame(storms)
    if fix_frame.empty or storm_frame.empty or fix_frame.storm_id.nunique() != len(storm_frame):
        raise ValueError("HURDAT2 storm reconciliation failed")
    return fix_frame, storm_frame


def haversine_km(latitude: float, longitude: float) -> float:
    radius = 6371.0088
    lat1, lat2 = math.radians(HOUSTON_LAT), math.radians(latitude)
    dlat = lat2 - lat1
    dlon = math.radians(longitude - HOUSTON_LON)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(value))


def build_matches(fema: pd.DataFrame, fixes: pd.DataFrame, storms: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    required = {"source_id", "fema_declaration_string", "incident_type", "declaration_type",
                "declaration_title", "incident_begin_date", "incident_end_date"}
    if not required.issubset(fema.columns):
        raise ValueError(f"FEMA handoff lacks fields: {sorted(required - set(fema.columns))}")
    related = fema.loc[
        fema.incident_type.eq("Hurricane") | fema.declaration_title.str.contains(RELATED_PATTERN, na=False)
    ].copy()
    candidates: list[dict] = []
    unresolved: list[dict] = []
    for _, declaration in related.iterrows():
        begin = pd.to_datetime(declaration.incident_begin_date, errors="coerce", utc=True)
        end = pd.to_datetime(declaration.incident_end_date, errors="coerce", utc=True)
        year = begin.year if not pd.isna(begin) else None
        title = str(declaration.declaration_title).upper()
        named = storms.loc[storms.year.eq(year) & storms.storm_name.map(
            lambda name: bool(re.search(rf"(?<![A-Z]){re.escape(str(name).upper())}(?![A-Z])", title))
        )]
        if named.empty:
            unresolved.append({"fema_declaration": declaration.fema_declaration_string,
                               "title": declaration.declaration_title,
                               "reason": "No same-year HURDAT2 storm name appears in the FEMA title."})
            continue
        for _, storm in named.iterrows():
            track = fixes.loc[fixes.storm_id.eq(storm.storm_id)].copy()
            distances = track.apply(lambda row: haversine_km(row.latitude, row.longitude), axis=1)
            nearest_index = distances.idxmin()
            nearest = track.loc[nearest_index]
            distance = float(distances.loc[nearest_index])
            overlap = (not pd.isna(begin) and not pd.isna(end)
                       and begin <= storm.lifetime_end_utc.normalize()
                       and end >= storm.lifetime_start_utc.normalize())
            if distance <= 250:
                geography = "houston_proximate_recorded_track"
            elif distance <= 500:
                geography = "regional_recorded_track_250_to_500_km"
            else:
                geography = "not_houston_proximate_over_500_km"
            compound = len(named) > 1
            if overlap and distance <= 250 and not compound:
                confidence = "high"
            elif overlap and distance <= 500:
                confidence = "medium"
            else:
                confidence = "unresolved"
            reasons = ["same-year storm name appears in FEMA title",
                       "incident period overlaps HURDAT2 lifetime" if overlap else "no incident/lifetime overlap",
                       f"nearest recorded center {distance:.1f} km from Houston"]
            if compound:
                reasons.append("FEMA title names multiple storms; relationship is one declaration to multiple candidates")
            if "EVACUATION" in title:
                reasons.append("FEMA title describes evacuation, not local storm-center passage")
            candidates.append({
                "fema_source_id": declaration.source_id,
                "fema_declaration": declaration.fema_declaration_string,
                "fema_declaration_type": declaration.declaration_type,
                "fema_title": declaration.declaration_title,
                "fema_incident_begin": declaration.incident_begin_date,
                "fema_incident_end": declaration.incident_end_date,
                "candidate_storm_id": storm.storm_id,
                "storm_name": storm.storm_name,
                "storm_year": int(storm.year),
                "storm_lifetime_start_utc": storm.lifetime_start_utc.isoformat(),
                "storm_lifetime_end_utc": storm.lifetime_end_utc.isoformat(),
                "temporal_overlap": bool(overlap),
                "minimum_recorded_distance_to_houston_km": round(distance, 1),
                "nearest_recorded_timestamp_utc": nearest.timestamp_utc.isoformat(),
                "geographic_relevance": geography,
                "confidence": confidence,
                "reason": "; ".join(reasons),
            })
    return pd.DataFrame(candidates), unresolved


def run(hurdat_manifest: str | Path, fema_path: str | Path) -> dict:
    manifest_path = Path(hurdat_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = (manifest_path.parent / manifest["file"]).resolve()
    if source.parent != manifest_path.parent or sha256(source) != manifest["sha256"]:
        raise ValueError("HURDAT2 snapshot hash/path verification failed")
    fixes, storms = parse_hurdat2(source)
    observed_years = [int(storms.year.min()), int(storms.year.max())]
    if observed_years != manifest["covered_years"]:
        raise ValueError("Parsed year coverage differs from manifest")
    fema_path = Path(fema_path).resolve()
    fema = pd.read_csv(fema_path, dtype=str, keep_default_na=False)
    matches, unresolved = build_matches(fema, fixes, storms)
    related_count = int((fema.incident_type.eq("Hurricane")
                         | fema.declaration_title.str.contains(RELATED_PATTERN, na=False)).sum())
    matched_declarations = int(matches.fema_source_id.nunique()) if not matches.empty else 0
    return {
        "scope": "Minimal identity/temporal/Houston-point matching audit; no declaration merge or impact estimate.",
        "hurdat2": {"source_url": manifest["source_url"], "retrieved_utc": manifest["retrieved_utc"],
                    "sha256": manifest["sha256"], "covered_years": observed_years,
                    "parsed_storms": int(len(storms)), "parsed_track_records": int(len(fixes)),
                    "first_storm_id": storms.iloc[0].storm_id, "last_storm_id": storms.iloc[-1].storm_id},
        "fema": {"source_file": str(fema_path), "hurricane_related_declarations": related_count,
                 "matched_declarations": matched_declarations},
        "candidate_match_count": int(len(matches)),
        "confidence_counts": matches.confidence.value_counts().to_dict() if not matches.empty else {},
        "candidate_matches": matches.to_dict(orient="records"),
        "unresolved_declarations_without_named_candidate": unresolved,
        "method": {"name": "same-year whole-name occurrence in FEMA title",
                   "time": "inclusive FEMA incident-date/HURDAT2 lifetime overlap",
                   "geography": "minimum great-circle distance from recorded storm centers to Houston reference point 29.7604,-95.3698; <=250 km proximate, 250-500 km regional, >500 km not proximate",
                   "confidence": "high requires unique title candidate, overlap, and <=250 km; medium permits compound title or 250-500 km; all other candidates unresolved"},
        "limitations": [
            "Nearest recorded center is not proof of local wind, rain, flooding, surge, damage, or hospital exposure.",
            "Track positions are usually six-hourly; no interpolation was performed.",
            "FEMA titles and dates are administrative source fields; matches are candidates, not merged physical events.",
            "One declaration may match multiple storms and one storm may match EM and DR declarations.",
            "No ED demand multiplier, simulation logic, GIS polygon, or patient linkage was created."],
    }


def markdown(report: dict) -> str:
    lines = ["# Minimal HURDAT2 to OpenFEMA matching audit", "",
             f"HURDAT2: {report['hurdat2']['parsed_storms']} storms, {report['hurdat2']['parsed_track_records']} track records, years {report['hurdat2']['covered_years'][0]}–{report['hurdat2']['covered_years'][1]}.",
             f"FEMA: {report['fema']['hurricane_related_declarations']} hurricane/tropical-storm-related declarations; {report['fema']['matched_declarations']} have named candidates.", "",
             "| FEMA declaration | Candidate storm | Temporal overlap | Geographic relevance | Confidence | Reason |",
             "|---|---|---|---|---|---|"]
    for match in report["candidate_matches"]:
        lines.append(f"| {match['fema_declaration']} — {match['fema_title']} | {match['candidate_storm_id']} — {match['storm_name']} {match['storm_year']} | {match['temporal_overlap']} | {match['geographic_relevance']} ({match['minimum_recorded_distance_to_houston_km']} km) | {match['confidence']} | {match['reason']} |")
    lines += ["", "## Unresolved declarations without a named candidate", ""]
    if report["unresolved_declarations_without_named_candidate"]:
        lines += [f"- {item['fema_declaration']} — {item['reason']}" for item in report["unresolved_declarations_without_named_candidate"]]
    else:
        lines.append("- None.")
    lines += ["", "## Method and limitations", "", f"- {report['method']}"]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Rita, Ike, and Beryl are reported only when supported by exact name/year, time overlap, and the recorded-track distance rule."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hurdat-manifest", type=Path, default=DEFAULT_HURDAT_MANIFEST)
    parser.add_argument("--fema", type=Path, default=DEFAULT_FEMA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run(args.hurdat_manifest, args.fema)
    destination = args.report.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(destination), "parsed_storms": report["hurdat2"]["parsed_storms"],
                      "fema_related": report["fema"]["hurricane_related_declarations"],
                      "candidates": report["candidate_match_count"],
                      "confidence": report["confidence_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
