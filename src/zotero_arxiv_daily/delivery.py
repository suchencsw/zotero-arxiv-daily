"""SMTP acknowledgement records, deliberately excluding credentials and addresses."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def write_delivery_receipt(papers, candidate_count, corpus_count):
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
        "candidate_count": candidate_count,
        "paper_count": len(papers),
        "zotero_corpus_count": corpus_count,
        "papers": [{"title": p.title, "url": p.url, "score": p.score} for p in papers],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)

