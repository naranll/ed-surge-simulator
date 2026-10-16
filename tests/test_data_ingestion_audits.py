"""Small synthetic tests for Phase 1 source ingestion and descriptive audits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.io.stata import StataMissingValue

from src.data_processing.nhamcs_2022 import audit_nhamcs, load_nhamcs, write_audit_report as write_nhamcs_report
from src.data_processing.noaa_weather import audit_noaa, load_noaa_csv, write_audit_report as write_noaa_report
from src.data_processing.nhamcs_preprocessing import preprocess_nhamcs, write_outputs as write_nhamcs_outputs


class NHAMCSIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "IMMEDR": [1, 2, 9],
            "WAITTIME": [0, 34, None],
            "LOV": [42, 90, 120],
            "ARRTIME": ["0815", "1430", "2359"],
            "AGE": [22, 41, 77],
            "DISP": [1, 2, 9],
            "VISIT_DATE": ["2022-01-01", "2022-01-02", "2022-01-03"],
        })

    def test_csv_ingestion_audits_rows_candidates_and_unknown_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "synthetic_nhamcs.csv"
            self.frame.to_csv(source, index=False)
            loaded = load_nhamcs(source)
            report = audit_nhamcs(loaded, source)

        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["column_count"], 7)
        self.assertEqual(report["exact_duplicate_row_count"], 0)
        self.assertEqual(loaded["IMMEDR"].tolist(), [1, 2, 9])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["triage_urgency"])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["waiting_time"])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["length_of_visit"])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["arrival_time"])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["age"])
        self.assertTrue(report["candidate_fields_requiring_codebook_verification"]["disposition"])
        self.assertIn(
            9,
            [item["value"] for item in report["candidate_fields_requiring_codebook_verification"]["triage_urgency"][0]["sample_value_counts"]],
        )


class NHAMCSPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "IMMEDR": [0, 1, 2, 3, 4, 5, 7, -8, -9, StataMissingValue(102)],
            "WAITTIME": [0, 10, 20, 30, 40, 50, -7, -9, 60, StataMissingValue(103)],
            "LOV": [0, 10, 20, 30, 40, 50, 60, -9, 70, StataMissingValue(101)],
            "ARRTIME": [0, 815, "2359", "2400", "1260", "0815", -9, 12345, "0800", StataMissingValue(102)],
            "AGE": [0, 1, 20, 94, 93, 4, 2, 9, 30, 50],
            "ADISP": [-7, 1, 2, 3, 4, -8, -9, 1, 2, 3],
            "LOS": [3] * 10,
            "PATWT": [1.0] * 10,
            "CSTRATM": [1] * 10,
            "CPSUM": [1] * 10,
        })

    def test_documented_sentinels_and_raw_values_are_preserved(self) -> None:
        cleaned, _, _ = preprocess_nhamcs(self.frame)
        self.assertEqual(cleaned["WAITTIME"].iloc[6], -7)
        self.assertTrue(pd.isna(cleaned["WAITTIME_minutes"].iloc[6]))
        self.assertEqual(cleaned["WAITTIME_missing_reason"].iloc[6], "not_applicable_not_seen_by_provider")
        self.assertEqual(cleaned["IMMEDR_missing_reason"].iloc[7], "unknown")
        self.assertEqual(cleaned["ADISP_missing_reason"].iloc[0], "not_applicable_not_eligible_live_discharge")
        self.assertEqual(cleaned["IMMEDR"].iloc[9], ".a")
        self.assertEqual(cleaned["IMMEDR_missing_reason"].iloc[9], "stata_extended_missing_a")
        self.assertEqual(cleaned["LOS"].iloc[0], 3)
        self.assertEqual(cleaned["LOV_minutes"].iloc[0], 0)

    def test_triage_keeps_non_triage_unknown_and_urgency_distinct(self) -> None:
        _, distribution, metadata = preprocess_nhamcs(self.frame)
        counts = dict(zip(distribution["triage_category"], distribution["unweighted_count"]))
        self.assertEqual(counts["no_triage_for_visit"], 1)
        self.assertEqual(counts["esa_does_not_conduct_triage"], 1)
        self.assertEqual(counts["unknown"], 1)
        self.assertEqual(counts["blank"], 1)
        self.assertEqual(counts["stata_extended_missing_a"], 1)
        self.assertEqual(counts["immediate"], 1)
        self.assertTrue(metadata["triage_distribution_output"]["weighting"]["computed"])
        self.assertAlmostEqual(distribution["survey_weighted_estimated_visits"].sum(), 10)

    def test_arrival_time_validation_age_semantics_and_export(self) -> None:
        cleaned, distribution, metadata = preprocess_nhamcs(self.frame)
        self.assertEqual(cleaned["ARRTIME_HHMM"].iloc[0], "0000")
        self.assertEqual(cleaned["ARRTIME_HHMM"].iloc[1], "0815")
        self.assertEqual(cleaned["arrival_hour"].iloc[1], 8)
        self.assertEqual(cleaned["ARRTIME_missing_reason"].iloc[3], "invalid_hhmm")
        self.assertEqual(cleaned["ARRTIME_missing_reason"].iloc[4], "invalid_hhmm")
        self.assertEqual(cleaned["AGE_category_note"].iloc[0], "under_1_year")
        self.assertEqual(cleaned["AGE_category_note"].iloc[3], "94_years_or_older_topcoded")
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_nhamcs_outputs(cleaned, distribution, metadata,
                Path(temp_dir) / "clean.csv", Path(temp_dir) / "triage.csv", Path(temp_dir) / "meta.json")
            self.assertTrue(all(path.is_file() for path in paths))
            exported = pd.read_csv(paths[0], keep_default_na=False)
            self.assertEqual(exported.loc[9, "IMMEDR"], ".a")

    def test_weighted_estimates_are_unavailable_without_design_fields(self) -> None:
        frame = self.frame.drop(columns=["PATWT", "CSTRATM", "CPSUM"])
        _, distribution, metadata = preprocess_nhamcs(frame)
        self.assertTrue(distribution["survey_weighted_estimated_visits"].isna().all())
        self.assertFalse(metadata["triage_distribution_output"]["weighting"]["computed"])
        self.assertIn("not present", metadata["triage_distribution_output"]["weighting"]["reason_unavailable"])

class NHAMCSAuditRegressionTests(unittest.TestCase):
    def test_codebook_mappings_keep_ed_and_hospital_duration_separate(self) -> None:
        frame = pd.DataFrame({"LOV": [180, -9], "LOS": [2, -7], "ADISP": [1, -7]})
        original = frame.copy(deep=True)
        report = audit_nhamcs(frame)
        fields = report["candidate_fields_requiring_codebook_verification"]
        self.assertEqual([item["column"] for item in fields["length_of_visit"]], ["LOV"])
        self.assertEqual([item["column"] for item in fields["hospital_length_of_stay"]], ["LOS"])
        self.assertEqual([item["column"] for item in fields["disposition"]], ["ADISP"])
        self.assertEqual(set(fields["disposition"][0]),
                         {"column", "dtype", "missing_count", "sample_value_counts", "codebook_verification_required"})
        pd.testing.assert_frame_equal(frame, original)

    def test_los_alone_is_not_an_ed_visit_duration_candidate(self) -> None:
        fields = audit_nhamcs(pd.DataFrame({"LOS": [2]}))["candidate_fields_requiring_codebook_verification"]
        self.assertEqual(fields["length_of_visit"], [])
        self.assertEqual(fields["hospital_length_of_stay"][0]["column"], "LOS")

    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "IMMEDR": [1, 2, 9],
            "WAITTIME": [0, 34, None],
            "LOS": [42, 90, 120],
            "ARRTIME": ["0815", "1430", "2359"],
            "AGE": [22, 41, 77],
            "DISP": [1, 2, 9],
            "VISIT_DATE": ["2022-01-01", "2022-01-02", "2022-01-03"],
        })

    def test_stata_ingestion_and_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "synthetic_nhamcs.dta"
            report_path = Path(temp_dir) / "reports" / "audit.json"
            self.frame.to_stata(source, write_index=False, version=118)
            report = audit_nhamcs(load_nhamcs(source), source)
            saved = write_nhamcs_report(report, report_path)
            self.assertTrue(saved.is_file())
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["row_count"], 3)
            waittime_column = next(column for column in report["columns"] if column["name"] == "WAITTIME")
            self.assertEqual(waittime_column["missing_count"], 1)
            self.assertEqual(waittime_column["distinct_non_missing"], 2)
            self.assertEqual(waittime_column["stata_extended_missing_values"][0]["code"], ".")

    def test_extended_stata_missing_codes_are_reported_separately(self) -> None:
        frame = pd.DataFrame({
            "IMMEDR": [1, StataMissingValue(102), StataMissingValue(103)],
        })
        report = audit_nhamcs(frame)
        column = report["columns"][0]
        self.assertEqual(column["missing_count"], 2)
        self.assertEqual(column["distinct_non_missing"], 1)
        self.assertEqual(
            {item["code"] for item in column["stata_extended_missing_values"]},
            {".a", ".b"},
        )
        triage_values = report["candidate_fields_requiring_codebook_verification"]["triage_urgency"][0]["sample_value_counts"]
        self.assertIn(".a", [item["value"] for item in triage_values])


class NOAAIngestionTests(unittest.TestCase):
    def test_csv_audit_reports_metadata_missing_dates_and_no_heatwave_indicator(self) -> None:
        synthetic = pd.DataFrame({
            "STATION": ["USW00012960"] * 3,
            "DATE": ["2022-08-01", "2022-08-03", "2022-08-03"],
            "LATITUDE": [29.98438] * 3,
            "LONGITUDE": [-95.36072] * 3,
            "ELEVATION": [27.5] * 3,
            "NAME": ["HOUSTON INTERCONTINENTAL AIRPORT, TX US"] * 3,
            "TMAX": [350, 360, 360],
            "TMIN": [260, 270, 270],
            "PRCP": [0, None, None],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "USW00012960.csv"
            report_path = Path(temp_dir) / "reports" / "noaa.json"
            synthetic.to_csv(source, index=False)
            loaded = load_noaa_csv(source)
            report = audit_noaa(loaded, source)
            saved = write_noaa_report(report, report_path)
            self.assertTrue(saved.is_file())

        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["station_check"]["all_rows_match_expected_station"], True)
        self.assertTrue(report["station_metadata"]["LATITUDE"]["available"])
        self.assertEqual(report["date_continuity"]["missing_calendar_date_count"], 1)
        self.assertEqual(report["duplicate_station_date_key_count"], 1)
        self.assertEqual(report["observation_fields"]["TMAX"]["numeric_distribution"]["mean"], 356.6666666666667)
        self.assertFalse(report["heatwave_indicators_calculated"])

    def test_required_field_missing_is_reported(self) -> None:
        report = audit_noaa(pd.DataFrame({"DATE": ["2022-01-01"]}))
        self.assertEqual(report["missing_observation_fields"], ["TMAX", "TMIN", "PRCP"])
        self.assertEqual(report["date_continuity"]["status"], "assessed")


if __name__ == "__main__":
    unittest.main()
