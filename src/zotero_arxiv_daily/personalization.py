"""Email adapter for the same grounded relevance gate as PaperFlow."""
import re
from .grounded_relevance import screen_zotero_relevance


def screen_ranked(papers, state, client, *, limit):
    rows=[]
    for i,p in enumerate(papers,1):
        match=re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?',p.url)
        if not match: raise RuntimeError('Real arXiv ID required for relevance review')
        p.similarity_rank=i
        rows.append({'arxiv_id':match.group(1),'title':p.title,'abstract':p.abstract,
                     'zotero_matches':p.zotero_matches,'zotero_recent_matches':p.zotero_recent_matches})
    decisions=screen_zotero_relevance(rows,state,client=client)
    formal,diagnostic=[],[]
    refs={r['key']:r for r in state['similarity_corpus']}
    for p,row in zip(papers,rows):
        d=decisions[row['arxiv_id']];p.relevance_decision=d
        p.reference_titles=[refs[k]['title'] for k in d.get('reference_keys',[]) if k in refs]
        p.recommendation_status='recommended' if d['relation'] in {'direct','method'} else 'diagnostic'
        (formal if p.recommendation_status=='recommended' else diagnostic).append(p)
    selected=formal[:limit]
    # Minimum five are an explicitly labelled diagnostic panel, never fake matches.
    if len(selected)<min(5,limit): selected.extend(diagnostic[:min(5,limit)-len(selected)])
    return sorted(selected,key=lambda p: -p.score)
