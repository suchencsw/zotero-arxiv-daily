from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
import os
import re
import hashlib
from datetime import datetime,timezone
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: (x.added_date,getattr(x,'key','')),reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        now = datetime.now(timezone.utc)
        if os.environ.get('EMAIL_RECENT_WEIGHTING') == 'recent_mixture':
            age=np.array([max(0,(now-(c.added_date.replace(tzinfo=timezone.utc) if c.added_date.tzinfo is None else c.added_date)).total_seconds()/86400) for c in corpus])
            recent=time_decay_weight*np.where(age<=30,2**(-age/14),0)
            if recent.sum(): time_decay_weight=.3*time_decay_weight+.7*recent/recent.sum()
        normalize=lambda text: re.sub(r'\s+',' ',text).strip()[:8000]
        sim = self.get_similarity_score([normalize(c.abstract) for c in candidates], [normalize(c.abstract) for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        references=[{'key':getattr(c,'key','') or f'ref_{i}','title':c.title,'abstract':c.abstract,
                     'date_added':c.added_date.replace(tzinfo=timezone.utc).isoformat() if c.added_date.tzinfo is None else c.added_date.isoformat(),
                     'weight':float(w)} for i,(c,w) in enumerate(zip(corpus,time_decay_weight))]
        self.profile_state={'similarity_corpus':references,'last_synced_at':now.isoformat(),'recent_days':30}
        self.profile_state['reference_digest']=hashlib.sha256(''.join(r['key']+normalize(r['abstract']) for r in references).encode()).hexdigest()
        for i,(s,c) in enumerate(zip(scores,candidates)):
            c.score = s
            c.zotero_matches=[{**references[int(j)],'similarity':float(sim[i,j])} for j in np.argsort(-sim[i])[:3]]
            recent_ids=[j for j,r in enumerate(references) if (now-datetime.fromisoformat(r['date_added'])).total_seconds()<=30*86400]
            recent_ids=sorted(recent_ids,key=lambda j: -sim[i,j])[:2]
            c.zotero_recent_matches=[{**references[j],'similarity':float(sim[i,j])} for j in recent_ids]
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
