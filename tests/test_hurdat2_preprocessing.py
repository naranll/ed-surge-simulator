"""Focused offline tests for final HURDAT2 handoff exports."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_processing.hurdat2_matching_audit import parse_hurdat2
from src.data_processing.hurdat2_preprocessing import (
    TRACK_COLUMNS, CROSSWALK_COLUMNS, build_crosswalk_export,
    build_track_export, write_outputs,
)


HURDAT_FIXTURE = """AL012020, ALPHA, 3,
20200101, 0000,  , TD, 28.0N, 95.0W, 30, -999,
20200101, 0600, L, TS, 29.0N, 96.0W, 40, 1000,
20200101, 0905, I, HU, 29.5N, 95.5W, -99, 990,
AL022020, BETA, 1,
20200201, 1200,  , EX, 30.0N, 94.0W, 35, -999,
"""


def candidate(index: int, storm: str | None = None, confidence: str = "high") -> dict:
    storm = storm or f"AL{index + 1:02d}2020"
    return {
        "fema_source_id": f"fema-{index:02}",
        "fema_declaration": f"DR-{index}-TX",
        "fema_declaration_type": "DR",
        "fema_title": f"HURRICANE TEST {index}",
        "fema_incident_begin": "2020-01-01",
        "fema_incident_end": "2020-01-03",
        "candidate_storm_id": storm,
        "storm_name": f"TEST{index}",
        "storm_year": 2020,
        "storm_lifetime_start_utc": "2020-01-01T00:00:00+00:00",
        "storm_lifetime_end_utc": "2020-01-02T00:00:00+00:00",
        "temporal_overlap": True,
        "minimum_recorded_distance_to_houston_km": float(index),
        "nearest_recorded_timestamp_utc": "2020-01-01T06:00:00+00:00",
        "geographic_relevance": "houston_proximate_recorded_track",
        "confidence": confidence,
        "reason": "synthetic reviewed candidate",
    }


class HURDAT2PreprocessingTests(unittest.TestCase):
    def test_parser_preserves_identifiers_raw_values_timestamps_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "hurdat.txt"
            source.write_text(HURDAT_FIXTURE, encoding="ascii")
            fixes, storms = parse_hurdat2(source)
        self.assertEqual(len(fixes), 4)
        self.assertEqual(len(storms), 2)
        self.assertEqual(fixes.storm_id.tolist(), ["AL012020"] * 3 + ["AL022020"])
        self.assertEqual(fixes.record_identifier.tolist(), ["", "L", "I", ""])
        self.assertEqual(fixes.time_raw.tolist(), ["0000", "0600", "0905", "1200"])
        self.assertEqual(fixes.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%MZ").tolist()[2], "2020-01-01T09:05Z")
        self.assertEqual(fixes.latitude_raw.tolist()[0], "28.0N")
        self.assertEqual(fixes.longitude_raw.tolist()[1], "96.0W")
        self.assertEqual(fixes.latitude.tolist()[0], 28.0)
        self.assertEqual(fixes.longitude.tolist()[1], -96.0)
        self.assertEqual(fixes.maximum_sustained_wind_raw.tolist()[2], "-99")
        self.assertTrue(pd.isna(fixes.maximum_sustained_wind_kt.iloc[2]))
        self.assertTrue(pd.isna(fixes.central_pressure_mb.iloc[0]))
        self.assertEqual(storms.fix_count.tolist(), [3, 1])

    def test_track_export_preserves_all_rows_and_missing_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "hurdat.txt"
            source.write_text(HURDAT_FIXTURE, encoding="ascii")
            manifest_path = directory / "manifest.json"
            manifest = {"file": source.name, "source_url": "https://example.test/hurdat.txt",
                        "retrieved_utc": "2026-01-01T00:00:00Z", "sha256": "a" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fixes, _ = parse_hurdat2(source)
            output = build_track_export(fixes, manifest, manifest_path)
        self.assertEqual(len(output), 4)
        self.assertEqual(list(output.columns), TRACK_COLUMNS)
        self.assertEqual(output.track_row.tolist(), [1, 2, 3, 4])
        self.assertEqual(output.maximum_sustained_wind_status.tolist()[2], "missing_source_sentinel")
        self.assertEqual(output.central_pressure_status.tolist(),
                         ["missing_source_sentinel", "reported", "reported", "missing_source_sentinel"])
        self.assertEqual(output.maximum_sustained_wind_raw.tolist()[2], "-99")
        self.assertEqual(output.central_pressure_raw.tolist()[0], "-999")
        self.assertTrue(output.source_line_number.is_unique)

    def test_crosswalk_preserves_19_relationships_18_declarations_and_confidence(self):
        rows = [candidate(i, confidence=["high", "medium", "unresolved"][i % 3]) for i in range(18)]
        rows.append(candidate(0, storm="AL992020", confidence="medium"))
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.json"
            audit.write_text(json.dumps({"candidate_matches": rows}), encoding="utf-8")
            output = build_crosswalk_export(audit, {"sha256": "b" * 64})
        self.assertEqual(len(output), 19)
        self.assertEqual(output.fema_source_id.nunique(), 18)
        self.assertEqual((output.fema_source_id == "fema-00").sum(), 2)
        self.assertEqual(set(output.confidence), {"High", "Medium", "Unresolved"})
        self.assertTrue(output.relationship_status.eq("candidate_not_confirmed_physical_event").all())
        self.assertTrue(output.distance_semantics.str.endswith("context_only").all())
        self.assertEqual(list(output.columns), CROSSWALK_COLUMNS)

    def test_invalid_crosswalk_cardinality_or_confidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.json"
            audit.write_text(json.dumps({"candidate_matches": [candidate(1)]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_crosswalk_export(audit, {"sha256": "b" * 64})
            rows = [candidate(i) for i in range(18)] + [candidate(0, storm="AL992020", confidence="confirmed")]
            audit.write_text(json.dumps({"candidate_matches": rows}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_crosswalk_export(audit, {"sha256": "b" * 64})

    def test_exports_are_byte_reproducible_and_refuse_overwrite(self):
        rows = [candidate(i) for i in range(18)] + [candidate(0, storm="AL992020", confidence="medium")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "hurdat.txt"
            source.write_text(HURDAT_FIXTURE, encoding="ascii")
            manifest_path = directory / "manifest.json"
            manifest = {"file": source.name, "source_url": "https://example.test/hurdat.txt",
                        "retrieved_utc": "2026-01-01T00:00:00Z", "sha256": "a" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fixes, _ = parse_hurdat2(source)
            tracks = build_track_export(fixes, manifest, manifest_path)
            audit = directory / "audit.json"
            audit.write_text(json.dumps({"candidate_matches": rows}), encoding="utf-8")
            crosswalk = build_crosswalk_export(audit, {"sha256": "b" * 64})
            first = write_outputs(tracks, crosswalk, directory / "tracks1.csv", directory / "cross1.csv")
            second = write_outputs(tracks, crosswalk, directory / "tracks2.csv", directory / "cross2.csv")
            self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
            self.assertEqual(first[1].read_bytes(), second[1].read_bytes())
            with self.assertRaises(FileExistsError):
                write_outputs(tracks, crosswalk, first[0], directory / "unused.csv")


if __name__ == "__main__":
    unittest.main()
