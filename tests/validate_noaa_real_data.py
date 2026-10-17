"""Independent references from raw strings; production used only for regeneration.

Run from repository root: .venv/Scripts/python.exe -B tests/validate_noaa_real_data.py
"""
import calendar
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from collections import Counter
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_processing.noaa_preprocessing import run_pipeline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    def check(name, condition, evidence):
        checks.append({'check': name, 'status': 'PASS' if condition else 'FAIL', 'evidence': evidence})
    source = ROOT/'data/raw/noaa/USW00012960.csv'
    meta_path = ROOT/'reports/noaa_preprocessing_metadata.json'
    m = json.loads(meta_path.read_text())
    c = m['configuration']
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    paths = {k: Path(v['path']) for k,v in m['outputs'].items()}
    clean_raw = pd.read_csv(paths['cleaned'], dtype=str, keep_default_na=False)
    clean = pd.read_csv(paths['cleaned'], low_memory=False)
    events = pd.read_csv(paths['events'], keep_default_na=False)
    thresholds = pd.read_csv(paths['thresholds']).set_index('calendar_day')
    check('Source rows and cells', len(raw)==20951 and raw.equals(clean_raw[list(raw.columns)])
          and clean.source_row.tolist()==list(range(1,20952)), f'{len(raw)} rows; {raw.size} original cells compared in source order')
    values = {}
    conversion_ok = True
    for field, unit in [('TMAX','c'),('TMIN','c'),('PRCP','mm')]:
        nums = []
        for token, attrs in zip(raw[field],raw[field+'_ATTRIBUTES']):
            q = attrs.split(',')[1]
            nums.append(float(token)/10 if token.strip() and token.strip()!='-9999' and not q else float('nan'))
        values[field] = nums
        actual = clean[field.lower()+'_'+unit].tolist()
        conversion_ok &= all((math.isnan(a) and math.isnan(b)) or math.isclose(a,b,abs_tol=1e-12,rel_tol=0) for a,b in zip(nums,actual))
    check('Independent native unit conversions', conversion_ok, 'All TMAX/TMIN/PRCP cells independently divided by 10 after raw quality checks; native source hash is documented')
    flagged = raw.TMIN_ATTRIBUTES.str.split(',').str[1].ne('')
    check('Flagged TMIN exclusion', int(flagged.sum())==1 and raw.loc[flagged,'DATE'].tolist()==['1977-12-08']
          and clean.loc[flagged,'tmin_c'].isna().all()
          and clean_raw.loc[flagged,'tmin_missing_reason'].tolist()==['noaa_quality_flag_S'], '1977-12-08 raw 183 retained; cleaned null with quality S reason')
    trace = raw.PRCP_ATTRIBUTES.str.split(',').str[0].eq('T')
    check('Trace and observation time', int(trace.sum())==2568 and clean.prcp_is_trace.tolist()==trace.tolist()
          and clean_raw.prcp_observation_time.eq('2400').all(), '2,568 traces; PRCP observation time 2400 retained')
    start,end = date.fromisoformat(c['baseline_start']),date.fromisoformat(c['baseline_end'])
    dates = [date.fromisoformat(x) for x in raw.DATE]
    baseline = [(d,v) for d,v in zip(dates,values['TMAX']) if start<=d<=end]
    check('Baseline coverage', len(baseline)==10958 and all(math.isfinite(v) for _,v in baseline), f'{len(baseline)} eligible calendar days')
    days = [(date(2001,1,1)+timedelta(days=i)).strftime('%m-%d') for i in range(365)]
    groups = {day: [] for day in days}
    for d,v in baseline:
        if d.strftime('%m-%d')!='02-29':
            groups[d.strftime('%m-%d')].append(v)
    references = {}; counts = []; differences = []
    for i,day in enumerate(days):
        pool = sorted(v for offset in range(-15,16) for v in groups[days[(i+offset)%365]])
        counts.append(len(pool))
        h=(len(pool)-1)*0.95; lo=math.floor(h); hi=math.ceil(h)
        references[day]=pool[lo]+(h-lo)*(pool[hi]-pool[lo])
        differences.append(abs(references[day]-thresholds.loc[day,'threshold_c']))
    references['02-29']=(references['02-28']+references['03-01'])/2
    check('Threshold sample coverage', counts==[930]*365 and thresholds.drop('02-29').sample_count.eq(930).all()
          and thresholds.loc['02-29','sample_count']==0, '365 pools of 930; Feb 29 uses no direct samples')
    check('Independent sorted linear quantiles', max(differences)<1e-10 and abs(references['02-29']-thresholds.loc['02-29','threshold_c'])<1e-10,
          f'Max threshold difference {max(differences):.3g} C; leap interpolation checked')
    expected_thresholds=[references[d.strftime('%m-%d')] for d in dates]
    above=[v>t for v,t in zip(values['TMAX'],expected_thresholds)]
    check('Daily thresholds and strict exceedance', all(abs(a-b)<1e-10 for a,b in zip(clean.threshold_c,expected_thresholds))
          and clean.exceeds_threshold.tolist()==above, 'Every observed day independently compared with strict >')
    runs=[]; i=0
    while i<len(dates):
        if not above[i]:
            i+=1; continue
        j=i+1
        while j<len(dates) and dates[j]==dates[j-1]+timedelta(days=1) and above[j]:
            j+=1
        if j-i>=2:
            runs.append((i,j))
        i=j
    event_ok=len(runs)==len(events); membership=[False]*len(dates)
    for n,(i,j) in enumerate(runs):
        if n>=len(events):
            event_ok=False; break
        e=events.iloc[n]; temps=values['TMAX'][i:j]
        excess=[values['TMAX'][k]-expected_thresholds[k] for k in range(i,j)]
        event_ok &= e.start_date==str(dates[i]) and e.end_date==str(dates[j-1]) and e.duration_days==j-i and e.confirmed_on==str(dates[i+1])
        for actual,ref in [(e.max_tmax_c,max(temps)),(e.mean_tmax_c,math.fsum(temps)/len(temps)),
                           (e.max_exceedance_c,max(excess)),(e.cumulative_exceedance_c_days,math.fsum(excess))]:
            event_ok &= math.isclose(actual,ref,abs_tol=1e-9,rel_tol=0)
        expected_id=f"{c['station']}_{m['config_id']}_{dates[i]:%Y%m%d}"
        boundary='both_censored' if i==0 and j==len(dates) else 'left_censored' if i==0 else 'right_censored' if j==len(dates) else 'complete'
        event_ok &= e.event_id==expected_id and e.boundary_status==boundary
        event_ok &= clean_raw.heatwave_event_id.iloc[i:j].eq(e.event_id).all()
        event_ok &= clean_raw.event_confirmed_on.iloc[i:j].eq(str(dates[i+1])).all()
        for k in range(i,j): membership[k]=True
    check('Independent event runs and summaries', bool(event_ok), f'{len(runs)} events; {sum(membership)} confirmed days; durations, confirmation dates and float64 intensities reconciled')
    # Source has no gaps/invalid TMAX. A short run at an edge can remain unknown.
    membership_ok=True
    for k,expected in enumerate(membership):
        token=clean_raw.heatwave_day[k]
        if expected: membership_ok &= token=='True'
        elif token=='': membership_ok &= above[k] and (k==0 or k==len(dates)-1)
        else: membership_ok &= token=='False'
    check('Daily/event membership', membership_ok, 'Confirmed days agree; unknown boundary labels are not coerced to false')
    schema_ok=True
    for key,p in paths.items():
        spec=m['outputs'][key]
        f=pd.read_csv(p,dtype=str,keep_default_na=False)
        schema_ok &= list(f.columns)==list(pd.read_csv(p,nrows=0).columns) and set(f.columns)==set(spec['schema']) and len(f)==spec['rows'] and digest(p)==spec['sha256']
        for col,dtype in spec['schema'].items():
            if dtype in ('Float64','float64','int64','Int64'):
                numeric=pd.to_numeric(f[col].replace('',None),errors='raise')
                schema_ok &= all(math.isfinite(x) for x in numeric.dropna())
                schema_ok &= int(numeric.isna().sum())==spec['null_counts'][col]
    check('Schemas hashes and numeric finiteness', bool(schema_ok), 'All output columns, row counts, hashes, numeric types and numeric null counts checked')
    protected=json.loads((ROOT/'reports/noaa_protected_files_before.json').read_text())
    changed=[p for p,h in protected.items() if not (ROOT/p.replace('\\','/')).is_file() or digest(ROOT/p.replace('\\','/'))!=h]
    check('Raw and established NHAMCS files unchanged', not changed, {'changed':changed,'files_checked':len(protected)})
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp)
        regenerated=run_pipeline(source,ROOT/'configs/noaa_heatwave.json',d,d/'meta.json')
        same=all(digest(Path(regenerated['outputs'][key]['path']))==digest(p) for key,p in paths.items())
        a=json.loads(json.dumps(m)); b=json.loads(json.dumps(regenerated))
        for obj in (a,b):
            obj.pop('metadata_path')
            for spec in obj['outputs'].values(): spec.pop('path')
        check('Reproducibility', same and a==b, 'CSVs byte-identical; metadata identical after normalizing explicit output destinations')
    for name,evidence in [('Scientific scope','No ED-risk calibration, citywide exposure validation, or station homogenization assessment.'),
                          ('Portability and provenance','Windows validation only; original download request/date unknown; unit verification tied to source hash.')]:
        checks.append({'check':name,'status':'WARNING','evidence':evidence})
    counts=Counter(x['status'] for x in checks)
    summary={status: counts[status] for status in ['PASS','WARNING','FAIL']}
    report={'summary':summary,'checks':checks,'event_count':len(events),'confirmed_heatwave_days':sum(membership)}
    (ROOT/'reports/noaa_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# NOAA independent real-data validation','',f'Results: {summary}','', '| Check | Status | Evidence |','|---|---|---|']
    lines += [f"| {x['check']} | {x['status']} | {x['evidence']} |" for x in checks]
    lines += ['', 'References are independently calculated from raw strings using sorted-value quantiles and standard-library event loops. Production code is called only for temporary regeneration.', '', 'Reproduce: `.venv/Scripts/python.exe -B tests/validate_noaa_real_data.py`']
    (ROOT/'reports/noaa_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 1 if summary.get('FAIL',0) else 0


if __name__=='__main__':
    raise SystemExit(main())
