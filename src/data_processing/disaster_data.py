"""Acquire immutable OpenFEMA v2 Harris County snapshots. No event curation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries'
ENTITY = 'DisasterDeclarationsSummaries'
FILTER = "state eq 'TX' and fipsCountyCode eq '201'"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fetch_bytes(url, timeout):
    """Bound the whole request, including DNS/body reads, using a child process.

    urllib's socket timeout alone is not a total request deadline. The child uses
    only Python's standard library, writes no files, and is killed on timeout.
    """
    code = (
        "import sys,urllib.request; "
        "r=urllib.request.urlopen(urllib.request.Request(sys.argv[1],"
        "headers={'User-Agent':'ED-Surge-Simulator-source-audit/1.0','Accept':'application/json'}),"
        "timeout=float(sys.argv[2])); sys.stdout.buffer.write(r.read())"
    )
    try:
        result = subprocess.run([sys.executable, '-B', '-c', code, url, str(timeout)],
                                capture_output=True, timeout=timeout+2, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f'Total request deadline exceeded ({timeout+2}s)') from exc
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-1500:])
    return result.stdout


def acquire(destination, page_size=20, timeout=20, retries=2, fetch=fetch_bytes, sleep=time.sleep):
    """New directory only. Save exact successful response bytes and failure evidence.

    Completion requires stable advertised counts, unique nonempty IDs, echoed
    filter/offset, and a final count probe. No rows are dropped or merged.
    """
    if not 1 <= page_size <= 1000 or not 1 <= timeout <= 120 or not 0 <= retries <= 5:
        raise ValueError('page_size 1..1000; timeout 1..120; retries 0..5 required')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'source_url': ENDPOINT, 'entity': ENTITY, 'version': 'v2',
                'filter': FILTER, 'started_utc': utc_now(), 'status': 'incomplete',
                'page_size': page_size, 'socket_timeout_seconds': timeout,
                'total_request_deadline_seconds': timeout+2, 'retries_after_first': retries,
                'pages': [], 'attempts': [], 'count_probe': None,
                'scope': 'Explicit filtered county records only; statewide/tribal/unresolved relevance is not inferred.',
                'limitations': ['API pagination is not a transactional historical snapshot.',
                               'Stable counts/IDs cannot detect all concurrent source edits.',
                               'No declarations are merged into physical disasters.']}
    records = []
    expected = None

    def request(skip, top, filename):
        params = {'$filter': FILTER, '$top': str(top), '$skip': str(skip),
                  '$orderby': 'id asc', '$inlinecount': 'allpages'}
        url = ENDPOINT + '?' + urlencode(params)
        for attempt in range(retries+1):
            log = {'url': url, 'attempt': attempt+1, 'started_utc': utc_now()}
            try:
                body = fetch(url, timeout)
                log.update(status='success', finished_utc=utc_now())
                manifest['attempts'].append(log)
                break
            except Exception as exc:
                log.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
                manifest['attempts'].append(log)
                print(f'Request {filename}, attempt {attempt+1}: {log["error"]}', file=sys.stderr, flush=True)
                if attempt == retries:
                    raise RuntimeError(f'Exhausted {retries+1} attempts for {url}') from exc
                sleep(min(2**attempt, 8))
        path = destination/filename
        with path.open('xb') as handle:
            handle.write(body)
        evidence = {'file': filename, 'url': url, 'parameters': params,
                    'retrieved_utc': utc_now(), 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
        # Record even malformed responses before validating their structure.
        if filename == 'count_probe.json':
            manifest['count_probe'] = evidence
        else:
            manifest['pages'].append(evidence)
        data = json.loads(body)
        rows, meta = data.get(ENTITY), data.get('metadata', {})
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError('Response lacks the expected record array')
        count = meta.get('count')
        if type(count) is not int or count < 0:
            raise ValueError('Missing or invalid metadata.count; cannot reconcile pagination')
        if meta.get('filter') != FILTER or meta.get('skip') != skip:
            raise ValueError('Response filter/offset differs from request')
        evidence.update(record_count=len(rows), advertised_count=count)
        return rows, count

    try:
        skip = 0
        for page in range(1000):
            rows, count = request(skip, page_size, f'page_{page+1:04}.json')
            if expected is None:
                expected = count
            if count != expected:
                raise ValueError('Advertised total changed during pagination; snapshot is incomplete')
            records.extend(rows)
            if len(rows) > page_size or len(records) > expected:
                raise ValueError('Response exceeds requested page size or advertised total')
            if len(records) == expected:
                break
            if not rows:
                raise ValueError('Empty page before advertised total reached')
            skip += len(rows)
        else:
            raise ValueError('Pagination safety limit reached')
        ids = [r.get('id') for r in records]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Missing or duplicate source IDs; records preserved but reconciliation failed')
        for r in records:
            if r.get('state') != 'TX' or r.get('fipsCountyCode') != '201':
                raise ValueError('A record violates the explicit county filter')
            if any(r.get(k) is not None and not isinstance(r[k], str) for k in ['fipsStateCode','fipsCountyCode','placeCode']):
                raise ValueError('Geographic identifier is not a source string; no implicit coercion')
        _, final_count = request(0, 1, 'count_probe.json')
        if final_count != expected:
            raise ValueError('Final count changed; snapshot not reconciled')
        manifest.update(status='complete', final_advertised_count=final_count)
    except Exception as exc:
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        manifest.update(finished_utc=utc_now(), retrieved_records=len(records), advertised_count=expected,
                        unique_source_ids=len({r.get('id') for r in records if isinstance(r.get('id'), str)}))
        with (destination/'manifest.json').open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New snapshot directory; existing directories are never overwritten')
    parser.add_argument('--page-size', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()
    destination = args.output or ROOT/'data/raw/fema'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = acquire(destination, args.page_size, args.timeout, args.retries)
    print(json.dumps({'manifest': str(destination/'manifest.json'), 'status': result['status'],
                      'records': result['retrieved_records'], 'expected': result['advertised_count'], 'error': result.get('error')}, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
