"""NOAA preprocessing CLI. Native scaling requires an explicitly verified file hash."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import pandas as pd

try:
    from .heatwaves import fit_calendar_thresholds, classify_heatwaves, config_id
except ImportError:  # direct script CLI
    from heatwaves import fit_calendar_thresholds, classify_heatwaves, config_id

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = [
    'https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
    'https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf',
    'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00012960.dly',
    'https://doi.org/10.1175/JCLI-D-12-00383.1',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC3040608/']
LIMITATIONS = [
    'Project-defined all-season meteorological episodes, not clinically validated ED-risk thresholds.',
    'Single airport exposure; citywide representativeness and station homogeneity unestablished.',
    'No ED surge multipliers or NHAMCS patient-weather joins.',
    'Pre-2021 classifications use retrospective baseline information, not prospective forecasts.',
    'Event membership is retrospective; confirmed_on identifies the earliest observed confirmation day.',
    'Censored event duration/intensity describe observed segments, not unknown full episodes.',
    'Original download URL/request/date unknown; reviewed file values matched official native archive.',
    'Blank NOAA quality flags do not establish absence of all measurement errors.',
    'Trace PRCP retains reported zero with trace indicator; not a confirmed dry day.',
    'Cross-platform Ubuntu/LinuxONE execution has not been tested.']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_attributes(value):
    parts = str(value).split(',')
    if len(parts) not in (3, 4):
        return ('', '', '', '', 'malformed_attributes')
    flags = [p.strip() for p in parts[:3]]
    time = parts[3].strip() if len(parts) == 4 else ''
    if any(len(p) > 1 for p in flags):
        return (*flags, time, 'malformed_attributes')
    return (*flags, time, 'parsed')


def prepare_weather_fields(raw, source_units):
    """Retain all source cells as strings; add separate typed fields. No imputation."""
    if source_units != 'ghcn_native_tenths':
        raise ValueError('Only explicitly verified native tenths units are supported')
    required = {'STATION', 'DATE', 'TMAX', 'TMIN', 'PRCP',
                'TMAX_ATTRIBUTES', 'TMIN_ATTRIBUTES', 'PRCP_ATTRIBUTES'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing columns: {required-set(raw.columns)}')
    out = raw.copy(deep=True).reset_index(drop=True)
    generated = {'source_row', 'date', 'station_id', 'config_id', 'threshold_c', 'exceeds_threshold',
                 'heatwave_day', 'heatwave_event_id', 'event_confirmed_on', 'classification_status'}
    generated.update(f'{e}_{s}' for e in ['tmax', 'tmin', 'prcp'] for s in
                     ['c', 'mm', 'measurement_flag', 'quality_flag', 'source_flag', 'observation_time', 'missing_reason', 'validation_status'])
    generated.add('prcp_is_trace')
    if generated.intersection(out.columns):
        raise ValueError('Source columns collide with derived schema')
    out['source_row'] = pd.Series(range(1, len(out)+1), dtype='int64')
    parsed = pd.to_datetime(out.DATE, format='%Y-%m-%d', errors='coerce')
    if parsed.isna().any() or not parsed.dt.strftime('%Y-%m-%d').eq(out.DATE).all():
        raise ValueError('Invalid source dates; preserve input and resolve before classification')
    if out.duplicated(['STATION', 'DATE']).any():
        raise ValueError('Duplicate station/date keys; no automatic row removal')
    out['date'] = out.DATE.astype('string')
    out['station_id'] = out.STATION.astype('string')
    for element, unit in [('TMAX', 'c'), ('TMIN', 'c'), ('PRCP', 'mm')]:
        prefix = element.lower()
        attrs = pd.DataFrame([parse_attributes(x) for x in out[element+'_ATTRIBUTES']],
                             columns=['measurement_flag', 'quality_flag', 'source_flag', 'observation_time', 'attribute_status'])
        for field in attrs.columns[:-1]:
            out[prefix+'_'+field] = attrs[field].astype('string')
        values, reasons = [], []
        for raw_value, (_, attr) in zip(out[element], attrs.iterrows()):
            token = str(raw_value).strip()
            reason = 'eligible'
            try:
                n = float(token)
            except ValueError:
                n = float('nan')
            if not token:
                reason = 'missing_blank'
            elif token == '-9999':
                reason = 'missing_native_sentinel'
            elif not math.isfinite(n):
                reason = 'invalid_nonfinite_or_nonnumeric'
            elif n != int(n):
                reason = 'invalid_native_noninteger'
            elif attr.attribute_status != 'parsed':
                reason = 'malformed_attributes'
            elif attr.quality_flag:
                reason = 'noaa_quality_flag_' + attr.quality_flag
            elif element == 'PRCP' and n < 0:
                reason = 'invalid_negative_precipitation'
            values.append(n/10 if reason == 'eligible' else pd.NA)
            reasons.append(reason)
        out[prefix+'_'+unit] = pd.Series(values, dtype='Float64')
        out[prefix+'_validation_status'] = pd.Series(reasons, dtype='string')
        out[prefix+'_missing_reason'] = pd.Series(['' if x == 'eligible' else x for x in reasons], dtype='string')
    out['prcp_is_trace'] = out.prcp_measurement_flag.eq('T').astype('boolean')
    return out


def run_pipeline(source, config_path, output_dir=None, metadata_path=None):
    source, config_path = Path(source).resolve(), Path(config_path).resolve()
    output_dir = Path(output_dir or ROOT/'data/processed').resolve()
    metadata_path = Path(metadata_path or ROOT/'reports/noaa_preprocessing_metadata.json').resolve()
    c = json.loads(config_path.read_text(encoding='utf-8'))
    source_hash = sha256(source)
    if source_hash != c['verified_source_sha256']:
        raise ValueError('Input hash is not the unit-verified source. Independently verify units before updating configuration.')
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    if not raw.STATION.eq(c['station']).all() or raw.empty:
        raise ValueError('Unexpected station or empty input')
    clean = prepare_weather_fields(raw, c['source_units'])
    thresholds, coverage = fit_calendar_thresholds(clean, c)
    clean, events, missing_dates = classify_heatwaves(clean, thresholds, c)
    tables = {'cleaned': (clean, output_dir/'noaa_houston_cleaned.csv'),
              'events': (events, output_dir/'heatwave_events.csv'),
              'thresholds': (thresholds, output_dir/'noaa_baseline_thresholds.csv')}
    targets = [p for _, p in tables.values()] + [metadata_path]
    raw_root = (ROOT/'data/raw').resolve()
    if len(set(targets)) != len(targets) or any(p in (source, config_path) or raw_root in p.parents for p in targets):
        raise ValueError('Output destinations must be distinct and outside raw/source/config files')
    metadata = {'source_file': str(source), 'source_sha256': source_hash, 'source_rows': len(raw),
                'source_columns': list(raw.columns), 'configuration': c, 'config_id': config_id(c),
                'configuration_file': str(config_path), 'metadata_path': str(metadata_path),
                'baseline': coverage, 'missing_calendar_dates': missing_dates,
                'source_download_url': None, 'source_download_date': None,
                'references': REFERENCES, 'limitations': LIMITATIONS,
                'software': {'python': platform.python_version(), 'pandas': pd.__version__},
                'code_hashes': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('heatwaves.py')]},
                'null_encoding': 'CSV empty derived cells; source strings reload with keep_default_na=False',
                'raw_preservation': 'All original columns retained as source strings in original row order; source_row is 1-based.',
                'exclusions': {e: clean[e+'_validation_status'].value_counts().to_dict() for e in ['tmax', 'tmin', 'prcp']},
                'tmax_below_tmin_count': int((clean.tmax_c < clean.tmin_c).fillna(False).sum()),
                'classification_counts': clean.classification_status.value_counts().to_dict(),
                'outputs': {}}
    for key, (table, path) in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, lineterminator='\n')
        metadata['outputs'][key] = {'path': str(path), 'rows': len(table), 'sha256': sha256(path),
            'schema': {col: str(table[col].dtype) for col in table.columns},
            'null_counts': {col: int(table[col].isna().sum()) for col in table.columns}}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'data/raw/noaa/USW00012960.csv')
    p.add_argument('--config', type=Path, default=ROOT/'configs/noaa_heatwave.json')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/processed')
    p.add_argument('--metadata', type=Path, default=ROOT/'reports/noaa_preprocessing_metadata.json')
    a = p.parse_args()
    m = run_pipeline(a.input, a.config, a.output_dir, a.metadata)
    print(json.dumps({'source_rows': m['source_rows'], 'events': m['outputs']['events']['rows'], 'metadata': str(a.metadata)}))


if __name__ == '__main__':
    main()
