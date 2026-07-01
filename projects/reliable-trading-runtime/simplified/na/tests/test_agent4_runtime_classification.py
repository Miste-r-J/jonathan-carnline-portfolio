from __future__ import annotations

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer


def test_missing_stop_broker_reject_classification() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    code = s._nt_broker_reject_code(status="REJECTED", reason="missing_stop_price", msg=None)
    assert code == "nt_missing_stop_price"


def test_required_fields_reject_maps_to_nt_schema() -> None:
    s = LiveCSVStreamer.__new__(LiveCSVStreamer)
    code = s._nt_broker_reject_code(
        status="LOCKOUT",
        reason="missing_stop_price",
        msg={"required_fields": ["stop_price", "target_price"], "schema_version": "2"},
    )
    assert code == "nt_schema"

