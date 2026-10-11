import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("backfill_notice", SCRIPTS / "manual_backfill_notice.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_audited_date_supported():
    assert module.load_notice("2026-10-10")["paper_count"] == 0
    with pytest.raises(KeyError):
        module.load_notice("2026-10-09")


@pytest.mark.parametrize("date", ["../2026-10-10", "2026-10-10;", "2026-10-32", "2026-1-1"])
def test_invalid_date_rejected(date):
    with pytest.raises(ValueError):
        module.load_notice(date)


def test_receipt_requires_real_ack_and_matching_date_run():
    p = {"schema_version": 1, "smtp_accepted": True, "target_date": "2026-10-10",
         "run_id": "123", "kind": "missed_empty_notification_backfill"}
    assert module.already_accepted(p, "2026-10-10", 123)
    assert not module.already_accepted(p, "2026-10-11", 123)
    assert not module.already_accepted(p, "2026-10-10", 124)
    assert not module.already_accepted({**p, "smtp_accepted": False}, "2026-10-10", 123)
