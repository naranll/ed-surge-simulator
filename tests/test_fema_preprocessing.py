"""Focused offline tests for the minimal FEMA declaration handoff table."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_processing.fema_preprocessing import (
    OUTPUT_COLUMNS, _date_parts, preprocess_declarations, write_output,
)


def source_record(index: int, **changes) -> dict:
    record = {
        "id": f"id-{index:03}", "femaDeclarationString": f"DR-{index}-TX",
        "disasterNumber": index, "incidentId": "incident-shared",
        "incidentType": "Biological" if index == 2 else "Hurricane",
        "declarationType": "DR", "declarationTitle": "SYNTHETIC INCIDENT",
        "state": "TX", "fipsStateCode": "48", "fipsCountyCode": "201",
        "placeCode": "99201", "designatedArea": "Harris (County)",
        "tribalRequest": False, "declarationDate": "2020-01-03T00:00:00.000Z",
        "incidentBeginDate": "2020-01-01T00:00:00.000Z",
        "incidentEndDate": "2020-01-04T00:00:00.000Z",
        "ihProgramDeclared": True, "iaProgramDeclared": False,
        "paProgramDeclared": True, "hmProgramDeclared": False,
        "lastRefresh": "2026-01-01T00:00:00.000Z", "hash": f"hash-{index}",
        "_source_record_sha256": f"sha-{index}", "_source_page_file": "page_0001.json",
        "_source_page_sha256": "page-sha", "_source_page_retrieved_utc": "2026-01-01T00:00:00Z",
    }
    record.update(changes)
    return record


def manifest(count: int) -> dict:
    return {"source_url": "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries",
            "version": "v2", "filter": "state eq 'TX' and fipsCountyCode eq '201'",
            "retrieved_records": count, "_manifest_sha256": "a" * 64}


class FEMAPreprocessingTests(unittest.TestCase):
    def test_preserves_all_rows_categories_ids_flags_and_stable_order(self):
        records = [source_record(2), source_record(1)]
        original = copy.deepcopy(records)
        frame = preprocess_declarations(records, manifest(2), "snapshot/manifest.json")
        self.assertEqual(records, original)
        self.assertEqual(frame.source_id.tolist(), ["id-001", "id-002"])
        self.assertEqual(frame.output_row.tolist(), [1, 2])
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.incident_id.tolist(), ["incident-shared"] * 2)
        self.assertIn("Biological", frame.incident_type.tolist())
        self.assertEqual(frame.ih_program_declared.tolist(), [True, True])
        self.assertEqual(list(frame.columns), OUTPUT_COLUMNS)

    def test_geographic_codes_are_strings_and_exact_harris_scope_is_enforced(self):
        frame = preprocess_declarations([source_record(1)], manifest(1), "snapshot/manifest.json")
        self.assertEqual(frame.loc[0, "fips_state_code"], "48")
        self.assertEqual(frame.loc[0, "fips_county_code"], "201")
        self.assertEqual(frame.loc[0, "county_fips"], "48201")
        self.assertEqual(str(frame.fips_county_code.dtype), "string")
        with self.assertRaises(ValueError):
            preprocess_declarations([source_record(1, fipsCountyCode="000")], manifest(1), "m.json")

    def test_date_values_remain_separate_and_missing_or_invalid_is_explicit(self):
        records = [source_record(1, declarationDate="2020-02-03T00:00:00.000Z",
                                 incidentBeginDate=None, incidentEndDate="bad-date")]
        frame = preprocess_declarations(records, manifest(1), "m.json")
        self.assertEqual(frame.declaration_date_raw[0], "2020-02-03T00:00:00.000Z")
        self.assertEqual(frame.declaration_date[0], "2020-02-03")
        self.assertEqual(frame.incident_begin_date[0], "")
        self.assertEqual(frame.incident_begin_date_status[0], "missing_source")
        self.assertEqual(frame.incident_end_date_raw[0], "bad-date")
        self.assertEqual(frame.incident_end_date_status[0], "invalid_source_date")
        self.assertEqual(_date_parts("2020-01-01T23:30:00-06:00")[1], "2020-01-02")

    def test_exact_reconciliation_rejects_count_or_id_problems(self):
        with self.assertRaises(ValueError):
            preprocess_declarations([source_record(1)], manifest(2), "m.json")
        with self.assertRaises(ValueError):
            preprocess_declarations([source_record(1), source_record(1)], manifest(2), "m.json")

    def test_repeatable_csv_and_no_overwrite(self):
        records = [source_record(2), source_record(1)]
        frame1 = preprocess_declarations(records, manifest(2), "m.json")
        frame2 = preprocess_declarations(list(reversed(records)), manifest(2), "m.json")
        with tempfile.TemporaryDirectory() as temporary:
            first = write_output(frame1, Path(temporary) / "first.csv")
            second = write_output(frame2, Path(temporary) / "second.csv")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = pd.read_csv(first, dtype=str, keep_default_na=False)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.fips_county_code.tolist(), ["201", "201"])
            with self.assertRaises(FileExistsError):
                write_output(frame1, first)


if __name__ == "__main__":
    unittest.main()
