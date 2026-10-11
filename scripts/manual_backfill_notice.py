"""One-off audited notification recovery; never changes the daily ranker."""
import html
import io
import json
import os
import re
import smtplib
import ssl
import zipfile
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from email_delivery_guard import github_get

ROOT = Path(__file__).resolve().parents[1]


def load_notice(date):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("Target date must be YYYY-MM-DD")
    datetime.strptime(date, "%Y-%m-%d")
    manifest = json.loads((ROOT / "recovery" / f"delivery-audit-{date[:7]}.json").read_text("utf8"))
    notice = manifest["notices"][date]
    if manifest.get("schema_version") != 1 or notice.get("paper_count") != 0:
        raise ValueError("Only audited empty-notification recovery is supported")
    if notice.get("kind") != "missed_empty_notification_backfill" or not notice.get("body"):
        raise ValueError("Invalid audited notice")
    return notice


def already_accepted(proof, date, run_id):
    return (proof.get("schema_version") == 1 and proof.get("smtp_accepted") is True
            and proof.get("target_date") == date
            and str(proof.get("run_id")) == str(run_id)
            and proof.get("kind") == "missed_empty_notification_backfill")


def check_previous(date):
    repo = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Invalid repository")
    current = os.environ["GITHUB_RUN_ID"]
    runs = github_get(repo, "actions/workflows/manual-backfill-notice.yml/runs?per_page=100")["workflow_runs"]
    for run in runs:
        if str(run["id"]) == current:
            continue
        artifacts = github_get(repo, f"actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
        for a in artifacts:
            if a["name"] != "email-backfill-notice-receipt" or a.get("expired"):
                continue
            blob = github_get(repo, f"actions/artifacts/{a['id']}/zip", binary=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                f = archive.getinfo("backfill-notice-receipt.json")
                if f.file_size > 256000:
                    raise RuntimeError("Invalid recovery receipt")
                proof = json.loads(archive.read(f))
            if already_accepted(proof, date, run["id"]):
                return True
    return False


def main():
    date = os.environ["BACKFILL_TARGET_DATE"]
    notice = load_notice(date)
    if check_previous(date):
        print("Audited notification already SMTP-accepted; not resending")
        return
    required = ("SENDER", "RECEIVER", "SENDER_PASSWORD")
    if not all(os.environ.get(k) for k in required):
        raise RuntimeError("Configured email secrets are missing")
    now = datetime.now(timezone.utc)
    msg = EmailMessage()
    msg["From"] = os.environ["SENDER"]
    msg["To"] = os.environ["RECEIVER"]
    msg["Subject"] = f"arXiv 日报漏发补报 · {date}（附本月发送核对）"
    msg["Date"] = format_datetime(now)
    msg["Message-ID"] = f"<zotero-arxiv-audited-notice-{date}@paperflow.local>"
    msg.set_content(notice["body"])
    rendered = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body style="font:15px/1.8 system-ui;background:#f5f7fb;padding:24px">'
                '<main style="max-width:720px;margin:auto;background:white;border:1px solid #dce3ef;border-radius:12px;padding:28px">'
                f'<h2 style="color:#245bd6">{html.escape(date)} · 日报状态补发</h2>'
                '<p style="color:#68758a">历史日期补报 · 不重复发送已补发论文</p>'
                f'<div style="white-space:pre-wrap">{html.escape(notice["body"])}</div></main></body></html>')
    msg.add_alternative(rendered, subtype="html")
    server = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=30, context=ssl.create_default_context())
    try:
        server.login(os.environ["SENDER"], os.environ["SENDER_PASSWORD"])
        if server.send_message(msg):
            raise RuntimeError("SMTP refused recipient")
        proof = {"schema_version": 1, "run_id": os.environ["GITHUB_RUN_ID"],
                 "smtp_accepted": True, "delivered_at": datetime.now(timezone.utc).isoformat(),
                 "target_date": date, "kind": notice["kind"], "paper_count": 0,
                 "message_id": msg["Message-ID"]}
        output = ROOT / "outputs" / "backfill-notice-receipt.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(proof, indent=2), encoding="utf8")
        print(f"SMTP accepted audited empty-notification backfill for {date}")
    finally:
        try:
            server.quit()
        except Exception:
            pass  # QUIT failure does not undo a prior DATA acknowledgement.


if __name__ == "__main__":
    main()
