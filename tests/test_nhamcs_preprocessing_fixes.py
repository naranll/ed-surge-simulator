"""Regressions from the independent Phase 2A validation."""

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
from pandas.io.stata import StataMissingValue

from src.data_processing.nhamcs_2022 import load_nhamcs
from src.data_processing.nhamcs_preprocessing import preprocess_nhamcs, write_outputs


def visits(codes):
    count = len(codes)
    return pd.DataFrame({
        "IMMEDR": codes, "WAITTIME": [12] * count, "LOV": [160] * count,
        "ARRTIME": ["0000"] * count, "AGE": [94] * count, "ADISP": [-7] * count,
        "PATWT": pd.Series([1.0] * count, dtype="float32"),
        "CSTRATM": [800] * count, "CPSUM": [999] * count,
    })


class NHAMCSFixTests(unittest.TestCase):
    def test_unexpected_codes_count_once_without_inventing_meanings(self):
        frame = visits([-7, -6, 9, 6, 2.5, "bad", 0, 7, -8, -9, None,
                        StataMissingValue(102), ".b", 1, 2, 3, 4, 5])
        cleaned, triage, meta = preprocess_nhamcs(frame)
        table = triage.set_index("triage_category")
        self.assertEqual(table.loc["other_or_undocumented", "unweighted_count"], 6)
        self.assertEqual(table.unweighted_count.sum(), len(frame))
        self.assertEqual(table.survey_weighted_estimated_visits.sum(), len(frame))
        self.assertEqual(cleaned.loc[0, "IMMEDR"], -7)
        self.assertEqual(cleaned.loc[0, "IMMEDR_missing_reason"], "undocumented_code_-7")
        self.assertTrue(pd.isna(cleaned.loc[0, "IMMEDR_category"]))
        self.assertNotIn("-7", meta["variable_rules"]["IMMEDR"])
        self.assertEqual(meta["triage_distribution_output"]["record_counts"]["urgency_levels_1_to_5"], 5)
        for name in ["no_triage_for_visit", "esa_does_not_conduct_triage", "unknown", "blank",
                     "source_null", "stata_extended_missing_a", "stata_extended_missing_b"]:
            self.assertEqual(table.loc[name, "unweighted_count"], 1)
            self.assertTrue(pd.isna(table.loc[name, "unweighted_percent_urgency_levels_1_to_5"]))

    def test_weighted_values_match_independent_decimal_reference(self):
        frame = visits([1, 1, 1, 2, 2, 3, 4, 5, 0, -8])
        frame["PATWT"] = pd.Series([1e8, 1, 3, 0.1, 12345.678, 98765.432, 3.14159, 1.7, 1.5, 2.5], dtype="float32")
        original = frame.copy(deep=True)
        cleaned, triage, meta = preprocess_nhamcs(frame)
        table = triage.set_index("triage_category")
        # Decimal sums of exact stored binary weights are independent of the
        # production float64 + math.fsum calculation, including denominator checks.
        def reference(mask):
            with localcontext() as context:
                context.prec = 100
                return float(sum((Decimal.from_float(float(x)) for x in frame.loc[mask, "PATWT"]), Decimal(0)))
        total = reference(frame.IMMEDR.notna())
        urgent_total = reference(frame.IMMEDR.isin([1, 2, 3, 4, 5]))
        for code, label in [(1, "immediate"), (2, "emergent"), (3, "urgent"), (4, "semi_urgent"), (5, "nonurgent")]:
            expected = reference(frame.IMMEDR.eq(code))
            self.assertEqual(table.loc[label, "survey_weighted_estimated_visits"], expected)
            self.assertAlmostEqual(table.loc[label, "survey_weighted_percent_all_records"], expected / total * 100, places=12)
            self.assertAlmostEqual(table.loc[label, "survey_weighted_percent_urgency_levels_1_to_5"], expected / urgent_total * 100, places=12)
        self.assertEqual(meta["triage_distribution_output"]["weighting"]["all_records_weight_sum"], total)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(cleaned.PATWT, original.PATWT)
        frame["CSTRATM"] *= 123
        frame["CPSUM"] *= 456
        _, changed, _ = preprocess_nhamcs(frame)
        pd.testing.assert_frame_equal(triage, changed)

    def test_provenance_custom_paths_and_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.csv"
            visits([1, -8, -7]).to_csv(source, index=False)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            cleaned, triage, meta = preprocess_nhamcs(load_nhamcs(source), source_path=source)
            paths = write_outputs(cleaned, triage, meta, folder / "custom.csv", folder / "triage.csv", folder / "manifest.json")
            saved = json.loads(paths[2].read_text())
            self.assertEqual(saved["source"]["sha256"], digest)
            self.assertEqual(saved["source"]["path"], source.resolve().as_posix())
            for key, path in zip(["cleaned_output", "triage_distribution_output", "metadata_output"], paths):
                self.assertEqual(saved[key]["path"], path.resolve().as_posix())
            self.assertEqual(saved["cleaned_output"]["row_count"], 3)
            schema = saved["cleaned_output"]["schema"]
            self.assertEqual([x["name"] for x in schema], list(cleaned.columns))
            read_types = {x["name"]: x["csv_read_dtype"] for x in schema}
            roundtrip = pd.read_csv(paths[0], dtype=read_types, keep_default_na=False, na_values=[""])
            self.assertEqual(roundtrip.ARRTIME_HHMM.iloc[0], "0000")
            self.assertFalse(saved["triage_distribution_output"]["weighting"]["reliability_established"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertIn("preprocessor_sha256", saved["software"])

    def test_nonfinite_weights_do_not_produce_estimates(self):
        frame = visits([1, 2])
        frame.loc[0, "PATWT"] = float("inf")
        _, triage, meta = preprocess_nhamcs(frame)
        self.assertFalse(meta["triage_distribution_output"]["weighting"]["computed"])
        self.assertTrue(triage.survey_weighted_estimated_visits.isna().all())
        self.assertEqual(triage.survey_weighted_estimated_visits.dtype, "float64")

    def test_schema_distinguishes_empty_raw_strings_and_nulls(self):
        frame = visits([1, 2, 3])
        frame["RAW_TEXT"] = ["", None, ".a"]
        cleaned, triage, meta = preprocess_nhamcs(frame)
        spec = next(c for c in meta["cleaned_output"]["schema"] if c["name"] == "RAW_TEXT")
        self.assertEqual(spec["null_count"], 1)
        self.assertEqual(spec["empty_string_count"], 1)
        self.assertEqual(spec["csv_null_count"], 2)
        self.assertEqual(cleaned.RAW_TEXT.iloc[0], "")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = write_outputs(cleaned, triage, meta, folder / "c.csv", folder / "t.csv", folder / "m.json")
            reloaded = pd.read_csv(paths[0], dtype={"RAW_TEXT": "string"}, keep_default_na=False, na_values=[""])
            self.assertEqual(int(reloaded.RAW_TEXT.isna().sum()), spec["csv_null_count"])
            self.assertEqual(reloaded.RAW_TEXT.iloc[2], ".a")


if __name__ == "__main__":
    unittest.main()
