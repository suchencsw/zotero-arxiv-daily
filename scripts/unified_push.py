"""Canonical cloud producer: compute once, persist encrypted, deliver twice."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from openai import OpenAI

import announcement_batches as announcements
import periodic_review as periods
import unified_delivery as delivery
import unified_recommendations as recommendations
from unified_model_cache import ModelCache


class Store:
    def __init__(self):self.ledger=periods.Ledger(state_file='unified-ledger.json')
    def load(self):return self.ledger.load()[0]
    def update(self,mutation):return self.ledger.update(mutation)
    def read(self,path):
        obj=self.ledger.api(f'contents/{path}?ref={periods.BRANCH}')
        if obj.get('encoding')=='none':obj=self.ledger.api('git/blobs/'+obj['sha'])
        return json.loads(base64.b64decode(obj['content']))
    def write_new(self,path,value):
        try:self.read(path);return
        except urllib.error.HTTPError as exc:
            if exc.code!=404:raise
        self.ledger.api('contents/'+path,{'branch':periods.BRANCH,'message':'Cache encrypted unified result or public announcement',
            'content':base64.b64encode(json.dumps(value,ensure_ascii=False).encode()).decode()},method='PUT')


def initialize(store):
    state=store.load()
    if state.get('unified_initialized'):return
    legacy,unused=periods.Ledger().load()
    daily,unused=periods.Ledger(state_file='announcement-ledger.json').load()
    def migrate(s):
        s['unified_initialized']=True
        s['activated_on']=legacy.get('activated_on',s['activated_on'])
        s.setdefault('sent',{}).update(legacy.get('sent',{}))
        s.setdefault('completed',{}).update(legacy.get('completed',{}))
        jobs=s.setdefault('jobs',{})
        for b in daily.get('batches',{}).values():
            if b.get('completed_at') or b.get('covered_at') or b.get('review_only'):continue
            key='daily:'+b['id']
            jobs.setdefault(key,{'batch_id':key,'kind':'daily','date':b['date'],'snapshot_path':b['path'],
                                 'label':'历史公告补发 · '+b['date'],'mode':'daily'})
    store.update(migrate)


def import_receipts(store):
    """Recover channel acknowledgements even if the prior state write failed."""
    import io
    import zipfile
    state=store.load();checked=[];proofs=[]
    for workflow in ('main.yml','periodic-review.yml'):
        runs=store.ledger.api(f'actions/workflows/{workflow}/runs?per_page=10')['workflow_runs']
        for run in runs:
            identifier=str(run['id'])
            if run['status']!='completed' or identifier==os.environ.get('GITHUB_RUN_ID') or identifier in state.get('unified_imported',[]):continue
            for artifact in store.ledger.api(f"actions/runs/{identifier}/artifacts?per_page=100")['artifacts']:
                if artifact['name']!='unified-channel-receipts' or artifact.get('expired'):continue
                raw=store.ledger.api(f"actions/artifacts/{artifact['id']}/zip",binary=True)
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    member=archive.getinfo('unified-receipts.json')
                    if member.file_size>2_000_000:raise RuntimeError('Receipt artifact too large')
                    for proof in json.loads(archive.read(member)):
                        if str(proof.get('run_id'))!=identifier:continue
                        if proof.get('channel')=='email' and proof.get('smtp_accepted') is True:proofs.append(proof)
                        elif proof.get('channel')=='feishu' and proof.get('api_accepted') is True and proof.get('complete') is True and proof.get('message_ids'):proofs.append(proof)
            checked.append(identifier)
    if checked:
        def merge(s):
            for p in proofs:
                s.setdefault('deliveries',{}).setdefault(p['bundle_id'],{}).setdefault(p['channel'],p)
            s['unified_imported']=list(dict.fromkeys(s.get('unified_imported',[])+checked))[-200:]
        store.update(merge)


def register_daily(store):
    query='+'.join(announcements.CATEGORIES)
    request=urllib.request.Request('https://rss.arxiv.org/atom/'+query,headers={'User-Agent':'unified-zotero-arxiv/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response:xml=response.read(8_000_001)
    if len(xml)>8_000_000:raise RuntimeError('RSS exceeds bound')
    groups=announcements.parse_feed(xml)
    for day,rows in groups.items():
        key='daily:'+announcements.batch_key(day)
        path=f'unified-announcements/{day}-{announcements.SCOPE}.json'
        store.write_new(path,{'schema_version':1,'announcement_date':day,'candidates':rows})
        def register(s,key=key,path=path,day=day):
            job=s.setdefault('jobs',{}).setdefault(key,{'batch_id':key,'kind':'daily','mode':'daily',
                'date':day,'snapshot_path':path,'label':'公告批次 · '+day})
            if not job.get('bundle_path') and not job.get('completed_at'):
                job['snapshot_path']=path  # upgrade legacy first100 to the complete current feed
        store.update(register)


def add_evening_review(store):
    state=store.load();now=datetime.now(periods.BJ)
    if now.hour<20:return
    if any(j['kind']=='daily' and not j.get('completed_at') and not j.get('covered_at') for j in state.get('jobs',{}).values()):return
    if any(str(p.get('delivered_at',''))[:10]==now.date().isoformat()
           for record in state.get('deliveries',{}).values() for p in record.values()):return
    daily=[j for j in state.get('jobs',{}).values() if j['kind']=='daily' and j.get('snapshot_path')]
    if not daily:return
    latest=max(daily,key=lambda j:j['date'])
    if (now.date()-date.fromisoformat(latest['date'])).days>14:return
    key=f'history:{now.date()}:{latest["batch_id"]}'
    store.update(lambda s:s.setdefault('jobs',{}).setdefault(key,dict(latest,batch_id=key,kind='history',mode='history',
          label='历史批次复盘 · '+latest['date'],service_date=str(now.date()))))


def pending_jobs(state,kind,now=None):
    now=now or datetime.now(periods.BJ)
    pending=[j for j in state.get('jobs',{}).values() if not j.get('completed_at') and not j.get('covered_at')]
    # Finish partially delivered frozen results before any new calculation.
    partial=[j for j in pending if j.get('bundle_id') and (state.get('deliveries',{}).get(j['bundle_id'])
             or state.get('feishu_pages',{}).get(j['bundle_id']))]
    if not partial:
        for proofs in state.get('deliveries',{}).values():
            stamps=[]
            for p in proofs.values():
                if p.get('paper_count',0)<=0:continue
                try:stamps.append(datetime.fromisoformat(p['delivered_at'].replace('Z','+00:00')).astimezone(periods.BJ))
                except (ValueError,KeyError,TypeError):continue
            if stamps and min(stamps).date()==now.date():return []
    remainder=[j for j in pending if j not in partial and ((kind=='daily' and j['kind'] in ('daily','history')) or (kind!='daily' and j['kind'] not in ('daily','history')))]
    remainder.sort(key=lambda j:j['date'])
    if kind=='daily' and remainder:remainder=[remainder[-1]]+remainder[:-1]
    return partial+remainder


def profile_requires_refresh(bundle,state,current_version):
    started=bool(state.get('deliveries',{}).get(bundle['bundle_id']) or state.get('feishu_pages',{}).get(bundle['bundle_id']))
    return not started and (str(bundle['zotero_version'])!=str(current_version)
                            or bundle.get('policy_version')!=recommendations.POLICY_VERSION)


def build_bundle(job,store,cache,client,deadline):
    print('Reading fresh Zotero for '+job['batch_id'],flush=True)
    refs,version,saved_ids,saved_titles=periods.fresh_zotero()
    print(f'Fresh Zotero: {len(refs)} references, library {version}',flush=True)
    state=store.load()
    if job.get('bundle_path'):
        old=delivery.decrypt_bundle(store.read(job['bundle_path']),os.environ['UNIFIED_STATE_KEY'])
        if not profile_requires_refresh(old,state,version):return old
    if job['kind'] in ('daily','history'):
        raw=store.read(job['snapshot_path'])['candidates']
    else:raise ValueError('Weekly/monthly/yearly recall is disabled by user preference')
    eligible=periods.exclude_saved_and_sent(raw,saved_ids,saved_titles,state)
    if not eligible and job['kind']=='history':
        # A labelled review may revisit previously recommended papers, never newly saved items.
        eligible=periods.exclude_saved_and_sent(raw,saved_ids,saved_titles,dict(state,sent={}))
    now=datetime.now(timezone.utc);selected=[];short=[];profile={'reference_digest':recommendations.digest(refs)}
    decisions={};ranked=[]
    if eligible:
        print(f'Embedding complete announcement pool: {len(raw)} raw, {len(eligible)} eligible',flush=True)
        vectors=cache.vectors(eligible,lambda texts:recommendations.embed_texts(client,texts),deadline)
        reference_vectors=recommendations.embed_texts(client,[r['abstract'] for r in refs])
        ranked,profile=recommendations.rank_pool(eligible,vectors,refs,reference_vectors,now)
        short=recommendations.shortlist(ranked,100)
        print(f'Shortlist: {len(short)}, now checking in similarity order',flush=True)
        limit=5 if job['kind']=='history' else 30
        model_cache=ModelCache(cache.db,os.environ['UNIFIED_STATE_KEY'])
        selected,decisions=recommendations.select_papers(short,profile,client,limit=limit,minimum=5,cache=model_cache)
        selected=recommendations.generate_summaries(selected,client,cache=model_cache)
        print(f'Model cache: {model_cache.hits} hits, {model_cache.misses} misses; checked {len(decisions)} of {len(short)} candidates',flush=True)
    # A fresh, complete version check is mandatory before freezing an unsent bundle.
    final_refs,final_version,_,_=periods.fresh_zotero()
    if final_version!=version or recommendations.digest(final_refs)!=recommendations.digest(refs):
        raise RuntimeError('Zotero changed while ranking; retry against the newest library')
    public_papers=[]
    for p in selected:
        public_papers.append({k:p.get(k) for k in ('arxiv_id','title','authors','abstract','categories','url','pdf_url',
            'score','recent_similarity','similarity_rank','tldr','reason','cited_titles','recommendation_status','relation')})
    bundle=delivery.seal_bundle({'schema_version':1,'batch_id':job['batch_id'],'label':job['label'],'mode':job['mode'],
        'created_at':datetime.now(timezone.utc).isoformat(),'synced_at':now.isoformat(),
        'profile_digest':profile['reference_digest'],'zotero_version':str(version),'zotero_count':len(refs),
        'latest_saved_at':max(r['date_added'] for r in refs),'recent_saved_count':sum(
            (now-datetime.fromisoformat(r['date_added'].replace('Z','+00:00'))).total_seconds()<=30*86400 for r in refs),
        'raw_count':len(raw),'eligible_count':len(eligible),'candidate_count':len(short),
        'diagnostic_count':sum(p['recommendation_status']=='diagnostic' for p in selected),
        'model':recommendations.MODEL,'embedding_model':recommendations.EMBED_MODEL,'policy_version':recommendations.POLICY_VERSION,
        'strategy':'rank_log_base_30pct+recent30d_half14_70pct;recall_global50_recent40_coverage10',
        'papers':public_papers})
    audit={'batch_id':bundle['batch_id'],'bundle_id':bundle['bundle_id'],'profile_digest':bundle['profile_digest'],
           'zotero_version':version,'zotero_count':len(refs),'latest_saved_at':bundle['latest_saved_at'],
           'raw_count':len(raw),'candidate_count':len(short),'selected_count':len(selected),
           'model_usage':client.totals() if hasattr(client,'totals') else {},
           'selected':[{'arxiv_id':p['arxiv_id'],'score':p['score'],'status':p['recommendation_status']} for p in selected],
           'screened':[{'arxiv_id':p['arxiv_id'],'score':p['score'],'recent_similarity':p['recent_similarity'],
                        'decision':decisions[p['arxiv_id']]['relation'],
                        'validation_error':decisions[p['arxiv_id']].get('validation_error')} for p in short if p['arxiv_id'] in decisions]}
    out=Path('outputs/unified');out.mkdir(parents=True,exist_ok=True)
    (out/(bundle['bundle_id'][:12]+'-audit.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),'utf8')
    return bundle


def save_bundle(store,job,bundle):
    path='unified-manifests/'+bundle['bundle_id']+'.json'
    store.write_new(path,delivery.encrypt_bundle(bundle,os.environ['UNIFIED_STATE_KEY']))
    store.update(lambda s:s['jobs'][job['batch_id']].update(bundle_id=bundle['bundle_id'],bundle_path=path,
                  profile_digest=bundle['profile_digest'],zotero_version=bundle['zotero_version']))


def deliver_bundle(store,job,bundle):
    state=store.load()
    def persist(current):
        def merge(s):
            s.setdefault('deliveries',{}).setdefault(bundle['bundle_id'],{}).update(current['deliveries'][bundle['bundle_id']])
        store.update(merge)
    def page_done(page,message):
        store.update(lambda s:s.setdefault('feishu_pages',{}).setdefault(bundle['bundle_id'],{}).update({page:message}))
    known=state.get('feishu_pages',{}).get(bundle['bundle_id'],{})
    delivery.deliver_pending(bundle,state,persist,{'email':delivery.send_email,
        'feishu':lambda b:delivery.send_feishu(b,known,page_done)})
    def complete(s):
        proofs=s.get('deliveries',{}).get(bundle['bundle_id'],{})
        if not all(c in proofs for c in ('email','feishu')):raise RuntimeError('Both channels must acknowledge the identical bundle')
        expected=[p['arxiv_id'] for p in bundle['papers']]
        if any(p.get('bundle_id')!=bundle['bundle_id'] or p.get('paper_ids')!=expected or p.get('profile_digest')!=bundle['profile_digest'] for p in proofs.values()):
            raise RuntimeError('Channel acknowledgement does not match the frozen ordered result')
        stamp=datetime.now(timezone.utc).isoformat()
        s['jobs'][job['batch_id']]['completed_at']=stamp
        s.setdefault('completed',{})[job['batch_id']]={'delivered_at':stamp,'paper_count':len(bundle['papers']),'bundle_id':bundle['bundle_id']}
        if job['kind']!='history':
            for identifier in delivery.formal_ids(bundle):s.setdefault('sent',{})[periods.fingerprint(identifier)]=stamp
    store.update(complete)
    print(f"Both channels accepted bundle {bundle['bundle_id'][:12]}: {len(bundle['papers'])} papers; Zotero {bundle['zotero_count']}/{bundle['zotero_version']}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=('daily',),default='daily')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',help='Persist the encrypted verified bundle without sending')
    parser.add_argument('--max-batches',type=int,default=1)
    args=parser.parse_args()
    os.environ['PAPERFLOW_LLM_MODEL']=recommendations.MODEL
    os.environ['PAPERFLOW_RELEVANCE_REVIEW_MODEL']=''
    os.environ['PAPERFLOW_COMPACT_REFERENCES']='true'
    os.environ['PAPERFLOW_REFERENCE_CONTEXT_LIMIT']='20'
    for key in ('OPENAI_API_KEY','ZOTERO_ID','ZOTERO_KEY','UNIFIED_STATE_KEY'):
        if not os.environ.get(key):raise RuntimeError('Missing configured credential: '+key)
    store=Store();initialize(store);import_receipts(store)
    register_daily(store);add_evening_review(store)
    state=store.load();jobs=pending_jobs(state,args.kind)
    if not jobs:print('No pending announcement or review; quiet retry');return
    client=recommendations.MeteredClient(OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=os.environ.get('OPENAI_BASE_URL','https://api.siliconflow.cn/v1'),timeout=180,max_retries=2))
    deadline=time.monotonic()+70*60
    cache=periods.PublicCache('outputs/unified-cache/public.sqlite')
    try:
        for job in jobs[:max(1,min(3,args.max_batches))]:
            bundle=build_bundle(job,store,cache,client,deadline)
            if args.dry_run:
                print(f"Verified only: {bundle['batch_id']}, bundle {bundle['bundle_id'][:12]}, {len(bundle['papers'])} papers; no message sent")
                continue
            save_bundle(store,job,bundle)
            if args.prepare_only:
                print(f"Prepared encrypted bundle {bundle['bundle_id'][:12]}; cloud delivery can reuse it without another LLM call")
                continue
            deliver_bundle(store,job,bundle)
    finally:
        cache.db.close()


if __name__=='__main__':main()
