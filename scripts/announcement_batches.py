"""Durable RSS announcement queue. Public metadata only; no calendar-day dedup."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from periodic_review import Ledger, BRANCH, CATEGORIES

CATEGORIES = tuple(filter(None, os.environ.get('ARXIV_BATCH_CATEGORIES', ','.join(CATEGORIES)).split(',')))
if not CATEGORIES or any(not re.fullmatch(r'[a-z][\w.-]*', c) for c in CATEGORIES):
    raise ValueError('Invalid configured arXiv categories')
BJ = timezone(timedelta(hours=8))
A = '{http://www.w3.org/2005/Atom}'
AR = '{http://arxiv.org/schemas/atom}'
DC = '{http://purl.org/dc/elements/1.1/}'
SCOPE = hashlib.sha256(','.join(sorted(CATEGORIES)).encode()).hexdigest()[:12]
FOCUS = ('visual navigation', 'visual localization', 'visual localisation', 'place recognition',
         'loop closure', 'slam', 'structure from motion', 'structure-from-motion', 'sfm',
         '3d reconstruction', 'three-dimensional reconstruction', 'gaussian splatting', 'nerf',
         'scene graph', 'pose graph', 'lunar', 'moon', 'crater', 'planetary', 'rover', 'embodied',
         'vision-language-action', 'vision language action', 'robot navigation', 'robotic navigation',
         '3d mapping', '3d scene', 'depth estimation', 'visual odometry')


def batch_key(date):
    datetime.strptime(date, '%Y-%m-%d')
    return f'rss:{date}:{SCOPE}'


def parse_feed(xml):
    root = ET.fromstring(xml)  # malformed/network failures must never become empty success
    if root.tag != A+'feed' or 'updates on' not in root.findtext(A+'title', ''):
        raise RuntimeError('Invalid arXiv announcement feed')
    groups, seen = {}, set()
    for entry in root.findall(A+'entry'):
        if entry.findtext(AR+'announce_type', 'new') not in ('new', 'cross'):
            continue
        identifier = re.sub(r'v\d+$', '', entry.findtext(A+'id', '').removeprefix('oai:arXiv.org:'))
        if not re.fullmatch(r'\d{4}\.\d{4,5}', identifier):
            raise RuntimeError('Invalid announcement identifier')
        date = entry.findtext(A+'published', '')[:10]
        batch_key(date)  # RSS published label is the announcement date, not submission date
        if identifier in seen:
            continue
        seen.add(identifier)
        abstract = re.sub(r'^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*', '',
                          entry.findtext(A+'summary', '').strip(), flags=re.I)
        title = entry.findtext(A+'title', '').strip()
        if not title or not abstract.strip():
            raise RuntimeError('Announcement missing title/abstract')
        groups.setdefault(date, []).append({
            'arxiv_id': identifier, 'title': title, 'abstract': abstract.strip(),
            'authors': [x.strip() for x in entry.findtext(DC+'creator', '').split(',') if x.strip()],
            'categories': [x.get('term') for x in entry.findall(A+'category') if x.get('term')],
            'publish_date': date, 'source': 'arxiv', 'url': f'https://arxiv.org/abs/{identifier}',
            'paper_url': f'https://arxiv.org/abs/{identifier}', 'pdf_url': f'https://arxiv.org/pdf/{identifier}'})
    return groups


def select_candidates(rows, channel):
    if channel == 'email':
        return rows[:100]  # preserve the email fork's official feed-order recall
    def priority(p):
        t, a = p['title'].lower(), p['abstract'].lower()
        return sum(3*(term in t)+(term in a) for term in FOCUS)
    ordered = sorted(rows, key=priority, reverse=True)
    selected, seen = [], set()
    for category in CATEGORIES:
        bucket = [p for p in ordered if category in p['categories'] and p['arxiv_id'] not in seen][:8]
        selected.extend(bucket)
        seen.update(p['arxiv_id'] for p in bucket)
    selected.extend(p for p in ordered if p['arxiv_id'] not in seen)
    return selected[:100]


def plan(state, now, force=False):
    if force and state.get('batches'):
        latest = max(state['batches'].values(), key=lambda b: b['date'])
        return latest, 'papers', ''
    pending = sorted((b for b in state.get('batches', {}).values() if not b.get('completed_at')),
                     key=lambda b: b['date'])
    if pending:
        return pending[0], 'papers', ''
    batches = sorted(state.get('batches', {}).values(), key=lambda b: b['date'])
    if now.astimezone(BJ).hour < 20 or not batches:
        return None, 'wait', ''
    latest = batches[-1]
    if (now.astimezone(BJ).date()-datetime.strptime(latest['date'], '%Y-%m-%d').date()).days > 14:
        return None, 'wait', ''
    review = f"review:{now.astimezone(BJ).date()}:{latest['id']}"
    if review in state.get('completed', {}):
        return None, 'wait', ''
    return latest, 'review', review


def valid_proof(proof, channel, run_id):
    base = (proof.get('schema_version') == 1 and str(proof.get('run_id')) == str(run_id)
            and isinstance(proof.get('paper_count'), int) and not isinstance(proof['paper_count'], bool)
            and proof['paper_count'] > 0)
    if channel == 'email':
        return base and proof.get('smtp_accepted') is True
    ids = proof.get('message_ids')
    return (base and proof.get('api_accepted') is True and proof.get('complete') is True
            and isinstance(ids, list) and bool(ids) and all(isinstance(x,str) and x for x in ids))


def apply_proof(state, proof):
    key = proof.get('announcement_batch_id')
    if key and key in state.get('batches', {}):
        if proof.get('announcement_mode') == 'review':
            review = proof.get('announcement_review_key')
            if review:
                state.setdefault('completed', {})[review] = {'delivered_at': proof['delivered_at']}
        elif proof.get('announcement_mode') == 'papers':
            state['batches'][key]['completed_at'] = proof['delivered_at']


def import_proofs(ledger, channel):
    workflow, name, filename = ('main.yml', 'email-smtp-receipt', 'smtp-receipt.json') if channel == 'email' else (
        'daily-push.yml', 'feishu-delivery-receipt', 'feishu-receipt.json')
    state, _ = ledger.load()
    receipts, examined = [], []
    runs = ledger.api(f'actions/workflows/{workflow}/runs?per_page=50')['workflow_runs']
    for run in runs:
        if str(run['id']) == os.environ['GITHUB_RUN_ID'] or run['status'] != 'completed':
            continue
        if str(run['id']) in state.get('imported', []):
            continue
        if datetime.fromisoformat(run['created_at'].replace('Z','+00:00')) < datetime.now(timezone.utc)-timedelta(days=7):
            continue
        for a in ledger.api(f"actions/runs/{run['id']}/artifacts?per_page=100")['artifacts']:
            if a['name'] != name or a.get('expired'):
                continue
            blob = ledger.api(f"actions/artifacts/{a['id']}/zip", binary=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                info = archive.getinfo(filename)
                if info.file_size > 256000:
                    raise RuntimeError('Oversized announcement receipt')
                proof = json.loads(archive.read(info))
            if valid_proof(proof, channel, run['id']):
                receipts.append(proof)
        examined.append(str(run['id']))
    if examined:
        def mutation(s):
            for proof in receipts:
                apply_proof(s, proof)
            s['imported'] = list(dict.fromkeys(s.get('imported', [])+examined))[-500:]
        ledger.update(mutation)


def save_snapshot(ledger, date, rows, channel):
    key = batch_key(date)
    state, _ = ledger.load()
    if key in state.get('batches', {}):
        return
    path = f'announcement-batches/{date}-{SCOPE}.json'
    snapshot = {'schema_version': 1, 'batch_id': key, 'announcement_date': date,
                'candidates': select_candidates(rows, channel)}
    if not snapshot['candidates']:
        raise RuntimeError('Cannot register empty batch')
    try:
        old = ledger.api(f'contents/{path}?ref={BRANCH}')
    except urllib.error.HTTPError as exc:
        if exc.code != 404: raise
        old = None
    if old is None:
        ledger.api(f'contents/{path}', {'branch': BRANCH, 'message': f'Cache public announcement {date}',
                   'content': base64.b64encode(json.dumps(snapshot,ensure_ascii=False).encode()).decode()}, method='PUT')
    ledger.update(lambda s: s.setdefault('batches', {}).setdefault(key, {
        'id': key, 'date': date, 'path': path, 'count': len(snapshot['candidates'])}))


def set_output(key, value):
    with Path(os.environ['GITHUB_OUTPUT']).open('a',encoding='utf8') as f:
        f.write(f'{key}={value}\n')


def prepare(channel):
    ledger = Ledger(state_file='announcement-ledger.json')
    # Initialize the shared state branch before saving a snapshot.
    state, sha = ledger.load()
    if sha is None: ledger.update(lambda s: s.setdefault('batches', {}))
    import_proofs(ledger, channel)
    query = '+'.join(CATEGORIES)
    req = urllib.request.Request(f'https://rss.arxiv.org/atom/{query}', headers={'User-Agent':'arxiv-batch-queue/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response:
        xml = response.read(8_000_001)
    if len(xml) > 8_000_000: raise RuntimeError('Announcement feed exceeds limit')
    groups = parse_feed(xml)
    for date, rows in groups.items():
        save_snapshot(ledger, date, rows, channel)
    state, _ = ledger.load()
    batch, mode, review = plan(state, datetime.now(BJ), os.environ.get('ARXIV_FORCE_SEND') == 'true')
    if batch is None:
        set_output('should_run', 'false')
        print('No pending public announcement batch; quiet retry, no empty message')
        return
    obj = ledger.api(f"contents/{batch['path']}?ref={BRANCH}")
    if obj.get('encoding') == 'none': obj = ledger.api(f"git/blobs/{obj['sha']}")
    snapshot = json.loads(base64.b64decode(obj['content']))
    if snapshot.get('batch_id') != batch['id'] or not 1 <= len(snapshot.get('candidates', [])) <= 100:
        raise RuntimeError('Invalid persisted announcement snapshot')
    path = Path('data/selected-announcement.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(snapshot,ensure_ascii=False),encoding='utf8')
    values = {'ARXIV_BATCH_PATH':str(path),'ARXIV_BATCH_ID':batch['id'],'ARXIV_BATCH_DATE':batch['date'],
              'ARXIV_BATCH_MODE':mode,'ARXIV_REVIEW_KEY':review}
    if mode == 'review': values['PAPERFLOW_DAILY_LIMIT'] = '5'
    with Path(os.environ['GITHUB_ENV']).open('a',encoding='utf8') as f:
        for k,v in values.items(): f.write(f'{k}={v}\n')
    set_output('should_run','true')
    print(f"Selected {mode}: public announcement {batch['date']}, {batch['count']} candidates; not submission date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', required=True, choices=('email','feishu'))
    parser.add_argument('--record')
    args = parser.parse_args()
    if args.record:
        proof = json.loads(Path(args.record).read_text('utf8'))
        if not valid_proof(proof,args.channel,os.environ['GITHUB_RUN_ID']):
            raise RuntimeError('No real complete delivery acknowledgement')
        Ledger(state_file='announcement-ledger.json').update(lambda s: apply_proof(s,proof))
    else:
        prepare(args.channel)


if __name__ == '__main__': main()
