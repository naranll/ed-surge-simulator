"""Project-defined retrospective meteorological episodes; no ED-risk estimation."""
from __future__ import annotations

import hashlib
import json
import math
import pandas as pd

EVENT_COLUMNS = ['event_id', 'station_id', 'config_id', 'definition', 'start_date',
                 'end_date', 'confirmed_on', 'duration_days', 'max_tmax_c',
                 'mean_tmax_c', 'max_exceedance_c', 'cumulative_exceedance_c_days',
                 'boundary_status']


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def validate_config(c):
    policies = {'comparison': '>', 'quantile_interpolation': 'linear',
                'leap_day_policy': 'exclude_samples_interpolate_adjacent_thresholds',
                'quality_eligibility': 'blank_quality_flag_only',
                'missing_data_policy': 'no_interpolation_break_runs',
                'boundary_handling': 'censor_adjacent_unknown_or_series_edge', 'season': 'all'}
    for k, v in policies.items():
        if c[k] != v:
            raise ValueError(f'Unsupported {k}: {c[k]}')
    if not 0 < c['percentile'] < 1 or not 0 < c['minimum_baseline_coverage'] <= 1:
        raise ValueError('Invalid percentile/coverage')
    if type(c['window_half_width']) is not int or not 0 <= c['window_half_width'] <= 182:
        raise ValueError('Invalid calendar window')
    if type(c['minimum_duration']) is not int or c['minimum_duration'] < 1:
        raise ValueError('Invalid duration')
    if c['definition'] not in ('percentile', 'absolute'):
        raise ValueError('Unknown definition')
    if c['definition'] == 'absolute' and not math.isfinite(float(c['absolute_threshold_c'])):
        raise ValueError('Absolute threshold must be finite')
    start, end = pd.Timestamp(c['baseline_start']), pd.Timestamp(c['baseline_end'])
    if start > end or start.strftime('%m-%d') != '01-01' or end.strftime('%m-%d') != '12-31':
        raise ValueError('Baseline must contain complete calendar years')


def fit_calendar_thresholds(frame, c):
    """Pool eligible baseline values on a circular nonleap calendar."""
    validate_config(c)
    dates = pd.to_datetime(frame['date'])
    expected = pd.date_range(c['baseline_start'], c['baseline_end'])
    b = frame.loc[dates.between(expected[0], expected[-1])].copy()
    coverage = {'expected_days': len(expected), 'observed_days': len(b),
                'valid_tmax_days': int(b.tmax_c.notna().sum()),
                'missing_dates': expected.difference(pd.DatetimeIndex(b.date)).strftime('%Y-%m-%d').tolist()}
    if coverage['valid_tmax_days'] / len(expected) < c['minimum_baseline_coverage']:
        raise ValueError(f'Insufficient baseline coverage: {coverage}')
    calendar = pd.date_range('2001-01-01', '2001-12-31').strftime('%m-%d').tolist()
    positions = {d: i for i, d in enumerate(calendar)}
    md = pd.to_datetime(b.date).dt.strftime('%m-%d')
    b = b.loc[md.ne('02-29')].copy()
    pos = pd.to_datetime(b.date).dt.strftime('%m-%d').map(positions)
    years = expected[-1].year - expected[0].year + 1
    n_expected = years * (2 * c['window_half_width'] + 1)
    rows = []
    for i, day in enumerate(calendar):
        distance = (pos - i).abs()
        pool = b.loc[(distance <= c['window_half_width']) | (distance >= 365-c['window_half_width'])]
        valid = pool.loc[pool.tmax_c.notna()]
        if len(valid) / n_expected < c['minimum_baseline_coverage']:
            raise ValueError(f'Insufficient baseline window: {day}')
        rows.append({'calendar_day': day, 'threshold_c': float(valid.tmax_c.quantile(c['percentile'], interpolation='linear')),
                     'sample_count': len(valid), 'expected_sample_count': n_expected,
                     'contributing_years': int(pd.to_datetime(valid.date).dt.year.nunique()),
                     'threshold_method': 'pooled_linear_quantile', 'config_id': config_id(c)})
    thresholds = pd.DataFrame(rows)
    feb = thresholds.set_index('calendar_day').threshold_c
    leap = {'calendar_day': '02-29', 'threshold_c': float((feb['02-28']+feb['03-01'])/2),
            'sample_count': 0, 'expected_sample_count': 0, 'contributing_years': 0,
            'threshold_method': 'adjacent_threshold_mean_no_direct_samples', 'config_id': config_id(c)}
    thresholds = pd.concat([thresholds, pd.DataFrame([leap])], ignore_index=True).sort_values('calendar_day').reset_index(drop=True)
    return thresholds, coverage


def classify_heatwaves(frame, thresholds, c):
    """Classify on a complete analysis calendar, export only original source rows.

    Unknown days interrupt runs. Short runs adjacent to unknowns remain unknown;
    qualifying observed runs are confirmed but their total boundaries may be censored.
    """
    validate_config(c)
    out = frame.copy()
    if out.date.duplicated().any():
        raise ValueError('Duplicate dates are ambiguous; no automatic deduplication')
    if out.empty:
        raise ValueError('No source rows')
    daily = out.set_index(pd.to_datetime(out.date)).sort_index()
    full = pd.date_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(full)
    lookup = thresholds.set_index('calendar_day').threshold_c
    daily['threshold_c'] = [lookup.get(d.strftime('%m-%d'), float('nan')) for d in full]
    if c['definition'] == 'absolute':
        daily['threshold_c'] = float(c['absolute_threshold_c'])
    valid = daily.tmax_c.notna() & daily.threshold_c.notna()
    daily['exceeds_threshold'] = (daily.tmax_c > daily.threshold_c).astype('boolean').mask(~valid)
    daily['heatwave_day'] = pd.Series(False, index=full, dtype='boolean').mask(~valid)
    daily['heatwave_event_id'] = pd.Series(pd.NA, index=full, dtype='string')
    daily['event_confirmed_on'] = pd.Series(pd.NA, index=full, dtype='string')
    daily['classification_status'] = 'assessed_non_event'
    daily.loc[~valid, 'classification_status'] = 'unassessable_exposure_or_threshold'
    events = []
    i = 0
    while i < len(daily):
        if not valid.iloc[i] or not daily.exceeds_threshold.iloc[i]:
            i += 1
            continue
        j = i + 1
        while j < len(daily) and valid.iloc[j] and daily.exceeds_threshold.iloc[j]:
            j += 1
        left = i == 0 or not valid.iloc[i-1]
        right = j == len(daily) or not valid.iloc[j]
        idx = full[i:j]
        boundary = 'both_censored' if left and right else 'left_censored' if left else 'right_censored' if right else 'complete'
        if j-i >= c['minimum_duration']:
            event_id = f"{c['station']}_{config_id(c)}_{full[i]:%Y%m%d}"
            confirmed = full[i+c['minimum_duration']-1].strftime('%Y-%m-%d')
            daily.loc[idx, 'heatwave_day'] = True
            daily.loc[idx, 'heatwave_event_id'] = event_id
            daily.loc[idx, 'event_confirmed_on'] = confirmed
            daily.loc[idx, 'classification_status'] = 'confirmed_event_' + boundary
            vals = daily.loc[idx, 'tmax_c'].astype('float64')
            excess = vals - daily.loc[idx, 'threshold_c']
            events.append(dict(zip(EVENT_COLUMNS, [event_id, c['station'], config_id(c), c['definition'],
                full[i].strftime('%Y-%m-%d'), full[j-1].strftime('%Y-%m-%d'), confirmed, j-i,
                float(vals.max()), float(vals.mean()), float(excess.max()), float(excess.sum()), boundary])))
        elif left or right:
            daily.loc[idx, 'heatwave_day'] = pd.NA
            daily.loc[idx, 'classification_status'] = 'unconfirmed_short_run_' + boundary
        i = j
    columns = ['threshold_c', 'exceeds_threshold', 'heatwave_day', 'heatwave_event_id',
               'event_confirmed_on', 'classification_status']
    for column in columns:
        out[column] = daily.loc[pd.to_datetime(out.date), column].reset_index(drop=True).array
    out['config_id'] = config_id(c)
    events = pd.DataFrame(events, columns=EVENT_COLUMNS)
    events['duration_days'] = events.duration_days.astype('int64')
    for col in ['max_tmax_c', 'mean_tmax_c', 'max_exceedance_c', 'cumulative_exceedance_c_days']:
        events[col] = events[col].astype('float64')
    return out, events, full.difference(pd.DatetimeIndex(out.date)).strftime('%Y-%m-%d').tolist()
