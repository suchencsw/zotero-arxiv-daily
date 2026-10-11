"""Suppress retries only after a real SMTP acknowledgement for the Beijing date."""
import io
import json
import os
import re
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")


def decide_delivery(runs, fetch_receipt, *, now, current_run, recovery=""):
    today = now.astimezone(BEIJING).date()
    empty_sent = False
    for run in runs:
        if str(run["id"]) == str(current_run) or run.get("status") != "completed":
            continue
        created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        if created < now - timedelta(days=3):
            continue
        proof = fetch_receipt(run["id"])
        if not proof or proof.get("schema_version") != 1 or proof.get("smtp_accepted") is not True:
            continue
        if str(proof.get("run_id")) != str(run["id"]):
            continue
        try:
            sent_at = datetime.fromisoformat(proof["delivered_at"].replace("Z", "+00:00"))
            if sent_at.tzinfo is None or sent_at.astimezone(BEIJING).date() != today:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        count = proof.get("paper_count")
        if not isinstance(count, int) or isinstance(count, bool):
            continue
        kind = proof.get("kind")
        if recovery:
            if kind == "recovery" and proof.get("recovery_batch") == recovery and count > 0:
                return False, empty_sent, str(run["id"])
        elif kind == "papers" and count > 0:
            return False, empty_sent, str(run["id"])
        elif kind == "empty" and count == 0:
            empty_sent = True
    return True, empty_sent, None


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected and urlsplit(new_url).netloc != urlsplit(request.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def github_get(repository, path, *, binary=False):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                 "Accept": "application/vnd.github+json", "User-Agent": "zotero-email-delivery-guard"},
    )
    with urllib.request.build_opener(SafeRedirect()).open(request, timeout=30) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise RuntimeError("Delivery proof exceeds size limit")
    return payload if binary else json.loads(payload)


def receipt_for_run(repository, run_id):
    artifacts = github_get(repository, f"actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
    matches = [a for a in artifacts if a["name"] == "email-smtp-receipt" and not a.get("expired")]
    if not matches:
        return None
    blob = github_get(repository, f"actions/artifacts/{matches[0]['id']}/zip", binary=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        files = [f for f in archive.infolist() if f.filename == "smtp-receipt.json"]
        if len(files) != 1 or files[0].file_size > 256_000:
            raise RuntimeError("Invalid SMTP receipt archive")
        return json.loads(archive.read(files[0]))


def main():
    repository = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Invalid repository name")
    recovery = os.environ.get("EMAIL_RECOVERY_BATCH", "")
    if recovery and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", recovery):
        raise ValueError("Recovery date must be YYYY-MM-DD")
    if os.environ.get("EMAIL_FORCE_SEND") == "true" or os.environ.get("EMAIL_DRY_RUN") == "true":
        decision = True, False, None
    else:
        runs = github_get(repository, "actions/workflows/main.yml/runs?per_page=100")["workflow_runs"]
        decision = decide_delivery(
            runs, lambda run: receipt_for_run(repository, run),
            now=datetime.now(timezone.utc), current_run=os.environ["GITHUB_RUN_ID"], recovery=recovery,
        )
    should_run, empty_sent, proof_run = decision
    outputs = f"should_run={str(should_run).lower()}\nempty_already_sent={str(empty_sent).lower()}\n"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(outputs)
    print(outputs.strip())
    print(f"Matching SMTP proof run: {proof_run}" if proof_run else "No matching SMTP proof; retry allowed")


if __name__ == "__main__":
    main()
