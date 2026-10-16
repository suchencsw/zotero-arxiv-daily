"""SMTP acknowledgement records, deliberately excluding credentials and addresses."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def write_delivery_receipt(papers, candidate_count, corpus_count, *, profile=None):
    path = os.environ.get("EMAIL_DELIVERY_RECEIPT")
    if not path:
        return
    now = datetime.now(timezone.utc)
    recovery = os.environ.get("EMAIL_RECOVERY_BATCH", "")
    receipt = {
        "schema_version": 1,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "smtp_accepted": True,
        "delivered_at": now.isoformat(),
        "beijing_date": now.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "kind": "recovery" if recovery else ("papers" if papers else "empty"),
        "recovery_batch": recovery or None,
        "announcement_batch_id": os.environ.get("ARXIV_BATCH_ID"),
        "announcement_date": os.environ.get("ARXIV_BATCH_DATE"),
        "announcement_mode": os.environ.get("ARXIV_BATCH_MODE"),
        "announcement_review_key": os.environ.get("ARXIV_REVIEW_KEY"),
        "candidate_count": candidate_count,
        "paper_count": len(papers),
        "recommended_count": sum(getattr(p,'recommendation_status','recommended')!='diagnostic' for p in papers),
        "diagnostic_count": sum(getattr(p,'recommendation_status','')=='diagnostic' for p in papers),
        "zotero_reference_digest": (profile or {}).get('reference_digest'),
        "zotero_last_synced_at": (profile or {}).get('last_synced_at'),
        "weighting": os.environ.get('EMAIL_RECENT_WEIGHTING','rank_log'),
        "zotero_corpus_count": corpus_count,
        "papers": [{"title": p.title, "url": p.url, "score": float(p.score),
                    "recommendation_status": getattr(p,'recommendation_status','recommended'),
                    "similarity_rank":getattr(p,'similarity_rank',None)} for p in papers],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)

