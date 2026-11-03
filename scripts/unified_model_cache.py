"""Authenticated cache; any saved-reference, prompt or model change invalidates gates."""
import json
import os
from datetime import datetime,timedelta

import unified_delivery as D
import unified_recommendations as R
from periodic_relevance import SYSTEM_PROMPT


class ModelCache:
    def __init__(self,db,key):
        self.db=db;self.key=key;self.hits=0;self.misses=0
        self.db.execute('CREATE TABLE IF NOT EXISTS unified_private_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    def _get(self,key):
        row=self.db.execute('SELECT value FROM unified_private_cache WHERE key=?',(key,)).fetchone()
        if row is None:self.misses+=1;return None
        self.hits+=1
        return D.decrypt_payload(json.loads(row[0]),self.key)
    def _put(self,key,value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO unified_private_cache VALUES (?,?)',
                            (key,json.dumps(D.encrypt_payload(value,self.key))))
    def gate_key(self,p,state):
        references=[r['key'] for r in (p.get('zotero_recent_matches') or [])[:2]+(p.get('zotero_matches') or [])[:3]]
        return R.digest({'kind':'gate-v1','model':R.MODEL,'prompt':SYSTEM_PROMPT,
            'context_limit':os.environ.get('PAPERFLOW_REFERENCE_CONTEXT_LIMIT','40'),
            'scope':os.environ.get('PAPERFLOW_RESEARCH_SCOPE',''),'profile':state['reference_digest'],
            'recent_context':[r['key'] for r in state['similarity_corpus'] if datetime.fromisoformat(r['date_added'].replace('Z','+00:00'))>=
                (datetime.fromisoformat(state['last_synced_at'])-timedelta(days=30))],
            'references':references,'id':p['arxiv_id'],'title':p['title'],'abstract':p['abstract']})
    def get_gate(self,p,state):return self._get(self.gate_key(p,state))
    def put_gate(self,p,state,decision):
        # A malformed model answer must be retried next run rather than cached as a real rejection.
        if not decision.get('validation_error'):self._put(self.gate_key(p,state),decision)
    def summary_key(self,p):
        return R.digest({'kind':'summary-v1','model':R.MODEL,'prompt':R.SUMMARY_PROMPT,
                         'title':p['title'],'abstract':p['abstract']})
    def get_summary(self,p):return self._get(self.summary_key(p))
    def put_summary(self,p,value):self._put(self.summary_key(p),value)
    def transfer_key(self,p,state):
        return R.digest({'kind':'transfer-review','gate':self.gate_key(p,state),'prompt':R.TRANSFER_REVIEW_PROMPT})
    def get_transfer(self,p,state):return self._get(self.transfer_key(p,state))
    def put_transfer(self,p,state,value):self._put(self.transfer_key(p,state),value)
