"""Hand-calculated fixtures; no access to prior project modules or datasets."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from export_parameters import arrival_hour, calculate, export, verify_reference


def sample():
    return b'arrival_time,target_triage_acuity,wait_time_minutes\n0000,1,500\n2359,5,0\n0030,1,20\n,2,30\n1260,3,99\n'


class ExportTests(unittest.TestCase):
    def test_midnight_boundaries_and_padding(self):
        for token, hour in [('0000', 0), ('0', 0), ('59', 0), ('100', 1), ('0906', 9), ('2359', 23), ('0.0', 0)]:
            with self.subTest(token=token): self.assertEqual(arrival_hour(token), (hour, 'valid'))

    def test_invalid_and_missing_times(self):
        for token in ['2400', '1260', '-9', '12.5', 'NaN', 'inf', '-inf', '.a', 'text', '1e99999']:
            with self.subTest(token=token): self.assertEqual(arrival_hour(token), (None, 'invalid'))
        for token in ['', '   ']: self.assertEqual(arrival_hour(token), (None, 'missing'))

    def test_independent_denominators_and_normalization(self):
        p = calculate(sample(), 'fixture.csv')
        h, t = p['hourly_arrivals'], p['triage']
        self.assertEqual(h['counts'], [2] + [0]*22 + [1])
        self.assertEqual(h['denominator'], 3)
        self.assertEqual(h['excluded_by_reason'], {'missing': 1, 'invalid': 1})
        self.assertEqual(t['counts'], [2, 1, 1, 0, 1])
        self.assertEqual(t['denominator'], 5)
        for d in [h, t]: self.assertAlmostEqual(math.fsum(d['proportions']), 1, places=12)
        for key, expected in [('hourly_shape', h), ('triage_mix', t)]:
            self.assertEqual(p[key]['value'], expected['proportions'])
            self.assertEqual(p[key]['status'], 'verified')
            self.assertEqual(p[key]['tag'], 'DATA')

    def test_invalid_triage_and_duplicates_retained(self):
        data = b'arrival_time,target_triage_acuity\n1200,1\n1200,1\n1200,0\n1200,7\n1200,1.5\n1200,\n'
        p = calculate(data, 'fixture.csv')
        self.assertEqual(p['source']['records'], 6)
        self.assertEqual(p['hourly_arrivals']['denominator'], 6)
        self.assertEqual(p['triage']['counts'], [2, 0, 0, 0, 0])
        self.assertEqual(p['triage']['excluded_by_reason'], {'missing': 1, 'invalid': 3})

    def test_zero_denominators_and_bad_csv_fail(self):
        for data in [b'', b'arrival_time,target_triage_acuity\n',
                     b'arrival_time,target_triage_acuity\n,1\n',
                     b'arrival_time,target_triage_acuity\n0,\n',
                     b'arrival_time,arrival_time\n0,0\n',
                     b'arrival_time,target_triage_acuity\n0,1,extra\n', b'wrong\n1\n']:
            with self.subTest(data=data):
                with self.assertRaises(ValueError): calculate(data, 'fixture.csv')

    def test_unavailable_parameters_never_use_waiting_time(self):
        p = calculate(sample(), 'fixture.csv')
        q = calculate(sample().replace(b'500', b'999'), 'fixture.csv')
        self.assertEqual(p['hourly_arrivals'], q['hourly_arrivals'])
        self.assertEqual(p['triage'], q['triage'])
        self.assertEqual(p['unavailable_parameters'], q['unavailable_parameters'])
        self.assertEqual(len(p['unavailable_parameters']), 4)
        self.assertTrue(all(v['value'] is None for v in p['unavailable_parameters'].values()))
        with self.assertRaisesRegex(ValueError, 'snapshot hash'): verify_reference(sample(), p)

    def test_export_determinism_and_no_source_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'raw.csv'; target = root/'outputs/params.json'
            source.write_bytes(sample())
            p = export(source, target); first = target.read_bytes()
            export(source, target)
            self.assertEqual(first, target.read_bytes())
            self.assertEqual(json.loads(first), p)
            self.assertEqual(source.read_bytes(), sample())
            with self.assertRaises(ValueError): export(source, source)
            with self.assertRaises(ValueError): export(source, root/'scenario_parameters.json')
            with self.assertRaises(ValueError): export(source, target, reference=True)
            self.assertEqual(first, target.read_bytes())


if __name__ == '__main__': unittest.main()
