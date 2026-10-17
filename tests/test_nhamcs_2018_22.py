"""Focused tests for the third-party NHAMCS 2018-2022 pipeline."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_processing.nhamcs_2018_22 import load_source, preprocess, run_pipeline


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "year": ["2018", "2022", "2022", "2022", "bad"],
        "visit_month": ["1", "12", "12", "12", "13"],
        "day_of_week": ["1", "7", "7", "7", "0"],
        "arrival_time": ["0", "141", "", "2360", "12.5"],
        "target_triage_acuity": ["1", "5", "5", "5", "9"],
        "wait_time_minutes": ["0", "1440", "1440", "1440", "-1"],
        "age": ["0", "94", "94", "94", "bad"],
        "sex": ["Female", "Male", "Male", "Male", "Unknown"],
        "ems_arrival": ["No", "Yes", "Yes", "Yes", "Blank"],
        "insurance": ["Private", "Medicare", "Medicare", "Medicare", "Unknown_Blank"],
        "is_injury_poison": ["No injury", "Injury", "Injury", "Injury", ""],
        "residence": ["Private residence", "Other", "Other", "Other", "Unknown"],
        "race": ["White", "Asian", "Asian", "Asian", "Unknown/Other"],
        "region": ["South", "West", "West", "West", "South"],
        "episode": ["Initial visit to this ED", "Follow-up visit to this ED", "Follow-up visit to this ED", "Follow-up visit to this ED", "Blank"],
    })


class NHAMCS2018To22Tests(unittest.TestCase):
    def test_rows_raw_values_and_order_are_preserved(self):
        raw = fixture()
        original = raw.copy(deep=True)
        output, metadata = preprocess(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(len(output), len(raw))
        self.assertEqual(output.source_row_number.tolist(), [1, 2, 3, 4, 5])
        for column in raw.columns:
            self.assertEqual(output[f"{column}_raw"].tolist(), raw[column].tolist())
        self.assertEqual(metadata["records"]["dropped"], 0)

    def test_hhmm_parsing_handles_boundaries_missing_and_invalid(self):
        raw = fixture()
        raw["arrival_time"] = ["0000", "2359", "", "1260", "2400"]
        output, _ = preprocess(raw)
        self.assertEqual(output.arrival_hhmm.iloc[:2].tolist(), ["0000", "2359"])
        self.assertEqual(output.arrival_hour.iloc[:2].tolist(), [0, 23])
        self.assertEqual(output.arrival_minute.iloc[:2].tolist(), [0, 59])
        self.assertEqual(output.arrival_minute_of_day.iloc[:2].tolist(), [0, 1439])
        self.assertEqual(output.arrival_time_status.tolist(), ["valid", "valid", "missing", "invalid", "invalid"])
        self.assertTrue(output.arrival_hhmm.iloc[2:].isna().all())

    def test_codes_wait_extremes_and_invalid_values_are_explicit(self):
        output, _ = preprocess(fixture())
        self.assertEqual(output.day_of_week_status.tolist(), ["valid"] * 4 + ["invalid"])
        self.assertEqual(output.triage_status.tolist(), ["valid"] * 4 + ["invalid"])
        self.assertEqual(output.wait_time_minutes_clean.iloc[:4].tolist(), [0.0, 1440.0, 1440.0, 1440.0])
        self.assertEqual(output.wait_time_status.tolist(), ["valid"] * 4 + ["invalid_negative"])
        self.assertNotIn("day_of_week_name", output.columns)
        self.assertNotIn("triage_label", output.columns)

    def test_duplicates_are_flagged_not_deleted(self):
        raw = fixture()
        raw.iloc[2] = raw.iloc[1]
        output, metadata = preprocess(raw)
        self.assertEqual(len(output), 5)
        self.assertEqual(output.exact_duplicate_flag.tolist(), [False, True, True, False, False])
        self.assertEqual(metadata["validation"]["exact_duplicate_groups"], 1)
        self.assertEqual(metadata["validation"]["exact_duplicate_excess_rows"], 1)

    def test_missing_values_are_not_imputed(self):
        raw = fixture()
        raw.loc[0, ["year", "visit_month", "day_of_week", "arrival_time", "target_triage_acuity", "wait_time_minutes", "age"]] = ""
        output, _ = preprocess(raw)
        for column in ["year_clean", "visit_month_clean", "day_of_week_code", "arrival_hhmm", "triage_acuity_code", "wait_time_minutes_clean", "age_clean"]:
            self.assertTrue(pd.isna(output.loc[0, column]))
        for column in ["year_status", "visit_month_status", "day_of_week_status", "arrival_time_status", "triage_status", "wait_time_status", "age_status"]:
            self.assertEqual(output.loc[0, column], "missing")

    def test_repeatable_files_and_manifest_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.csv"
            output = folder / "output.csv"
            report = folder / "report.json"
            fixture().to_csv(source, index=False, lineterminator="\n")
            run_pipeline(source, output, report)
            first = (output.read_bytes(), report.read_bytes())
            run_pipeline(source, output, report)
            self.assertEqual(first, (output.read_bytes(), report.read_bytes()))
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved["records"]["input"], saved["output"]["rows"])
            self.assertEqual(saved["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(saved["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            reloaded = pd.read_csv(output, dtype=str, keep_default_na=False)
            self.assertEqual(len(reloaded), len(fixture()))

    def test_missing_required_column_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Required columns missing: year"):
            preprocess(fixture().drop(columns="year"))


if __name__ == "__main__":
    unittest.main()
