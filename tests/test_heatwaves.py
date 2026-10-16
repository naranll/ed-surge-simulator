"""Hand-checkable calendar, percentile, and event tests."""
import json
import unittest
import pandas as pd
from src.data_processing.noaa_preprocessing import ROOT
from src.data_processing.heatwaves import fit_calendar_thresholds, classify_heatwaves


def config():
    c = json.loads((ROOT/'configs/noaa_heatwave.json').read_text())
    c.update(baseline_start='2001-01-01', baseline_end='2001-12-31', window_half_width=1)
    return c


class HeatwaveTests(unittest.TestCase):
    def test_hand_quantile_wrap_leap_and_no_future_leakage(self):
        c = config()
        frame = pd.DataFrame({'date': pd.date_range('2001-01-01', '2001-12-31').strftime('%Y-%m-%d'), 'tmax_c': [0.0]*365})
        frame.loc[0, 'tmax_c'] = 10; frame.loc[1, 'tmax_c'] = 20; frame.loc[364, 'tmax_c'] = 30
        thresholds, coverage = fit_calendar_thresholds(frame, c)
        t = thresholds.set_index('calendar_day')
        self.assertEqual(t.loc['01-01', 'threshold_c'], 29)
        self.assertEqual(coverage['expected_days'], 365)
        self.assertTrue(t.drop('02-29').sample_count.eq(3).all())
        self.assertEqual(t.loc['02-29', 'threshold_c'], (t.loc['02-28','threshold_c']+t.loc['03-01','threshold_c'])/2)
        appended = pd.concat([frame, pd.DataFrame({'date':['2021-01-01'], 'tmax_c':[999.]})], ignore_index=True)
        pd.testing.assert_frame_equal(thresholds, fit_calendar_thresholds(appended, c)[0])

    def test_leap_samples_excluded_and_baseline_coverage_enforced(self):
        c = config(); c.update(baseline_start='2000-01-01', baseline_end='2000-12-31')
        frame = pd.DataFrame({'date': pd.date_range('2000-01-01','2000-12-31').strftime('%Y-%m-%d'), 'tmax_c':[10.]*366})
        frame.loc[59,'tmax_c'] = 999
        t, coverage = fit_calendar_thresholds(frame,c)
        self.assertTrue(t.threshold_c.eq(10).all()); self.assertEqual(coverage['expected_days'],366)
        with self.assertRaises(ValueError):
            fit_calendar_thresholds(frame.drop(index=10), c)
        frame.loc[10, 'tmax_c'] = float('nan')
        with self.assertRaises(ValueError):
            fit_calendar_thresholds(frame,c)

    def classify(self, values, dates=None, duration=2):
        c = config(); c['minimum_duration'] = duration
        dates = dates or pd.date_range('2020-01-01', periods=len(values)).strftime('%Y-%m-%d').tolist()
        frame = pd.DataFrame({'date':dates, 'tmax_c':values})
        t = pd.DataFrame({'calendar_day': pd.date_range('2000-01-01','2000-12-31').strftime('%m-%d'), 'threshold_c':[30.]*366})
        return classify_heatwaves(frame,t,c)

    def test_strict_ties_duration_confirmation_and_determinism(self):
        a, events, _ = self.classify([30,31,33,30])
        self.assertEqual(a.exceeds_threshold.tolist(),[False,True,True,False])
        self.assertEqual(events.duration_days.tolist(),[2])
        self.assertEqual(events.confirmed_on[0],'2020-01-03')
        self.assertEqual(events.cumulative_exceedance_c_days[0],4)
        self.assertEqual(events.boundary_status[0],'complete')
        pd.testing.assert_frame_equal(events,self.classify([30,31,33,30])[1])
        self.assertTrue(self.classify([30,31,33,30],duration=3)[1].empty)
        self.assertEqual(len(self.classify([30,31,33,32,30],duration=3)[1]),1)

    def test_missing_date_invalid_value_and_short_unknown_runs(self):
        a,e,gaps = self.classify([31,32],['2020-01-01','2020-01-03'])
        self.assertEqual(gaps,['2020-01-02']); self.assertTrue(e.empty)
        self.assertTrue(a.heatwave_day.isna().all())
        a,e,_ = self.classify([30,31,None,32,30])
        self.assertTrue(e.empty); self.assertTrue(a.heatwave_day.iloc[1:4].isna().all())
        self.assertTrue(pd.isna(a.exceeds_threshold[2]))

    def test_boundaries_and_leap_day_count(self):
        _,e,_ = self.classify([31,32])
        self.assertEqual(e.boundary_status[0],'both_censored')
        _,e,_ = self.classify([30,31,32,33,30],['2020-02-27','2020-02-28','2020-02-29','2020-03-01','2020-03-02'])
        self.assertEqual(e.duration_days[0],3)

    def test_absolute_threshold_does_not_change_primary(self):
        c = config(); c.update(definition='absolute',absolute_threshold_c=35)
        frame = pd.DataFrame({'date':['2020-01-01','2020-01-02'],'tmax_c':[36,37]})
        t = pd.DataFrame({'calendar_day':['01-01','01-02'],'threshold_c':[40,40]})
        a,e,_ = classify_heatwaves(frame,t,c)
        self.assertTrue(a.heatwave_day.all()); self.assertEqual(e.definition[0],'absolute')


if __name__ == '__main__':
    unittest.main()
