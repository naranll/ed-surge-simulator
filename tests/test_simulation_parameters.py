"""Focused tests of the minimal observed-parameter contract."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_processing.simulation_parameters import build_parameters, export_parameters


def fixture():
    return pd.DataFrame({
        "arrival_hour": ["0", "23", "0", "", "25"],
        "arrival_time_status": ["valid", "valid", "valid", "missing", "invalid"],
        "triage_acuity_code": ["1", "5", "1", "2", ""],
        "triage_status": ["valid", "valid", "valid", "valid", "missing"],
        "wait_time_minutes_clean": ["10", "20", "30", "40", "50"],
    })


class ParameterTests(unittest.TestCase):
    def test_counts_exclusions_and_probabilities(self):
        frame = fixture()
        original = frame.copy(deep=True)
        result = build_parameters(frame)
        a, t = result['hourly_arrivals'], result['triage']
        self.assertEqual(a['hours'], list(range(24)))
        self.assertEqual(a['counts'], [2] + [0]*22 + [1])
        self.assertEqual(a['denominator'], 3)
        self.assertEqual(a['excluded_by_status'], {'missing': 1, 'invalid': 1})
        self.assertEqual(t['codes'], [1, 2, 3, 4, 5])
        self.assertEqual(t['counts'], [2, 1, 0, 0, 1])
        self.assertEqual(t['denominator'], 4)  # Missing arrival still contributes.
        for d in [a, t]:
            self.assertEqual(sum(d['counts']), d['denominator'])
            self.assertAlmostEqual(math.fsum(d['proportions']), 1.0, places=12)
            self.assertEqual(d['denominator']+d['excluded_records'], 5)
        pd.testing.assert_frame_equal(frame, original)

    def test_unavailable_values_and_waiting_times_do_not_supply_parameters(self):
        frame = fixture()
        before = build_parameters(frame)
        frame['wait_time_minutes_clean'] = '999999'
        self.assertEqual(before, build_parameters(frame))
        self.assertEqual(set(before['unavailable_parameters']), {
            'provider_service_times', 'admission_probabilities', 'boarding_durations'})
        for item in before['unavailable_parameters'].values():
            self.assertIsNone(item['value'])
            self.assertEqual(item['status'], 'unavailable_from_primary_dataset')

    def test_invalid_contract_and_empty_denominators_fail(self):
        for column, values in [('arrival_hour', ['24', '-1', '1.5', 'inf', '', 'x']),
                               ('triage_acuity_code', ['0', '6', '1.5', 'inf', '', 'x'])]:
            for value in values:
                with self.subTest(column=column, value=value):
                    frame = fixture(); frame.loc[0, column] = value
                    with self.assertRaises(ValueError): build_parameters(frame)
        for status in ['arrival_time_status', 'triage_status']:
            frame = fixture(); frame[status] = 'missing'
            with self.assertRaisesRegex(ValueError, 'No valid records'): build_parameters(frame)
            frame.loc[0, status] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'Unexpected'): build_parameters(frame)
        with self.assertRaisesRegex(ValueError, 'Required processed columns'):
            build_parameters(fixture().drop(columns='arrival_hour'))

    def test_csv_json_repeatability_and_source_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'visits.csv'
            output = Path(directory)/'params.json'
            fixture().to_csv(source, index=False)
            source_bytes = source.read_bytes()
            result = export_parameters(source, output)
            self.assertEqual(json.loads(output.read_text()), result)
            first = output.read_bytes()
            export_parameters(source, output)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(source_bytes, source.read_bytes())
            with self.assertRaises(ValueError): export_parameters(source, source)
            unrelated = Path(directory)/'other.json'
            unrelated.write_text('{}')
            with self.assertRaises(ValueError): export_parameters(source, unrelated)
            self.assertEqual(unrelated.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
