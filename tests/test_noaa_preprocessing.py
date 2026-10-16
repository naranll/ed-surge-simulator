"""Synthetic NOAA preprocessing regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.data_processing.noaa_preprocessing import prepare_weather_fields, parse_attributes, run_pipeline, sha256, ROOT
from src.data_processing.heatwaves import fit_calendar_thresholds, classify_heatwaves


def config():
    return json.loads((ROOT/'configs/noaa_heatwave.json').read_text())


def raw_frame(dates, values=None):
    n = len(dates)
    return pd.DataFrame({'STATION': ['USW00012960']*n, 'DATE': list(dates),
                         'TMAX': values or ['350']*n, 'TMIN': ['-10']*n, 'PRCP': ['0']*n,
                         'TMAX_ATTRIBUTES': [',,X']*n, 'TMIN_ATTRIBUTES': [',,0']*n,
                         'PRCP_ATTRIBUTES': ['T,,W,2400']*n})


class WeatherTests(unittest.TestCase):
    def test_units_flags_trace_time_and_raw_preservation(self):
        raw = raw_frame(['2020-01-01', '2020-01-02'], ['  428', '-38'])
        raw.loc[0, 'TMIN_ATTRIBUTES'] = ',S,0'
        original = raw.copy(deep=True)
        clean = prepare_weather_fields(raw, 'ghcn_native_tenths')
        pd.testing.assert_frame_equal(raw, original)
        pd.testing.assert_frame_equal(clean[raw.columns], raw)
        self.assertEqual(clean.tmax_c.tolist(), [42.8, -3.8])
        self.assertTrue(pd.isna(clean.tmin_c[0]))
        self.assertEqual(clean.tmin_c[1], -1)
        self.assertEqual(clean.tmin_missing_reason[0], 'noaa_quality_flag_S')
        self.assertTrue(clean.prcp_is_trace.all())
        self.assertEqual(clean.prcp_observation_time.tolist(), ['2400']*2)
        self.assertEqual(clean.prcp_mm.tolist(), [0, 0])
        self.assertEqual(parse_attributes('H,,X')[:3], ('H', '', 'X'))

    def test_missing_invalid_and_unknown_units(self):
        raw = raw_frame(pd.date_range('2020-01-01', periods=5).strftime('%Y-%m-%d'), ['', '-9999', 'nan', '1.5', '400'])
        raw.loc[4, 'TMAX_ATTRIBUTES'] = 'bad'
        clean = prepare_weather_fields(raw, 'ghcn_native_tenths')
        self.assertTrue(clean.tmax_c.isna().all())
        with self.assertRaises(ValueError):
            prepare_weather_fields(raw, 'unknown')

    def test_duplicate_invalid_dates_rejected_without_mutation(self):
        for dates in [['2020-01-01']*2, ['not-date']]:
            with self.assertRaises(ValueError):
                prepare_weather_fields(raw_frame(dates), 'ghcn_native_tenths')

    def test_repeatable_export_and_unit_hash_guard(self):
        c = config(); c.update(baseline_start='2001-01-01', baseline_end='2001-12-31', window_half_width=1)
        raw = raw_frame(pd.date_range('2001-01-01', '2001-12-31').strftime('%Y-%m-%d'))
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); source = d/'source.csv'; cfg = d/'config.json'
            raw.to_csv(source, index=False)
            c['verified_source_sha256'] = sha256(source); cfg.write_text(json.dumps(c))
            m = run_pipeline(source, cfg, d/'out', d/'meta.json')
            before = {p: p.read_bytes() for p in [d/'meta.json', *list((d/'out').glob('*.csv'))]}
            run_pipeline(source, cfg, d/'out', d/'meta.json')
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            self.assertEqual(m['outputs']['cleaned']['rows'], 365)
            c['verified_source_sha256'] = 'wrong'; cfg.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                run_pipeline(source, cfg, d/'out', d/'meta.json')


if __name__ == '__main__':
    unittest.main()
