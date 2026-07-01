from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from na.discord_addons.cli.stream_live_csv import LiveCSVStreamer


MODEL_SHA = "116037797a3c96d5ffd1d5c5fd1db8d45b773f29365e036b84d59517708954e3"


def _make_lock_streamer(tmp_path: Path, *, alive_pids: set[int] | None = None) -> LiveCSVStreamer:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.run_id = "RUNID"
    streamer.model_sha = MODEL_SHA
    streamer.model_path = "model.joblib"
    streamer.active_preset_name = "es_maxpack_10_full_send_prop_safe_pnl"
    streamer.primary_preset = streamer.active_preset_name
    streamer.instrument_alias = "ES"
    streamer.nt_port = 5018
    streamer.nt_exec_policy = "paper"
    streamer.out_dir = tmp_path
    streamer._instance_lock_acquired = False
    streamer._instance_lock_paths_cache = None
    streamer._lock_slug = LiveCSVStreamer._lock_slug.__get__(streamer, LiveCSVStreamer)
    streamer._instance_lock_key = LiveCSVStreamer._instance_lock_key.__get__(streamer, LiveCSVStreamer)
    streamer._instance_lock_payload = LiveCSVStreamer._instance_lock_payload.__get__(streamer, LiveCSVStreamer)
    streamer._instance_lock_owner_summary = LiveCSVStreamer._instance_lock_owner_summary.__get__(streamer, LiveCSVStreamer)
    streamer._instance_lock_action_hint = LiveCSVStreamer._instance_lock_action_hint.__get__(streamer, LiveCSVStreamer)
    streamer._replace_existing_owner_process = LiveCSVStreamer._replace_existing_owner_process.__get__(streamer, LiveCSVStreamer)
    streamer._acquire_instance_lock = LiveCSVStreamer._acquire_instance_lock.__get__(streamer, LiveCSVStreamer)
    streamer._release_instance_lock = LiveCSVStreamer._release_instance_lock.__get__(streamer, LiveCSVStreamer)
    local_lock = tmp_path / "local.runlock"
    global_lock = tmp_path / "global.runlock"
    streamer._instance_lock_paths = lambda: (local_lock, global_lock)
    alive = set(alive_pids or set())
    streamer._pid_is_alive = lambda pid: pid in alive
    streamer.replace_existing_run = False
    streamer.nt_host = "127.0.0.1"
    return streamer


def _write_lock(path: Path, *, pid: int, lock_key: str, port: int = 5018) -> None:
    payload = {
        "pid": pid,
        "run_id": "OWNER_RUN",
        "ts": "2026-04-16T04:39:18.966548-06:00",
        "model": MODEL_SHA,
        "preset": "es_maxpack_10_full_send_prop_safe_pnl",
        "instrument": "ES",
        "nt_port": port,
        "nt_exec_policy": "paper",
        "lock_key": lock_key,
        "cmd_summary": f"port={port} instrument=ES policy=paper",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_acquire_instance_lock_replaces_stale_lock_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    streamer = _make_lock_streamer(tmp_path, alive_pids=set())
    current_key = streamer._instance_lock_key()
    local_lock, global_lock = streamer._instance_lock_paths()
    _write_lock(local_lock, pid=424242, lock_key=current_key)
    _write_lock(global_lock, pid=424242, lock_key=current_key)

    streamer._acquire_instance_lock()

    assert streamer._instance_lock_acquired is True
    local_payload = json.loads(local_lock.read_text(encoding="utf-8"))
    global_payload = json.loads(global_lock.read_text(encoding="utf-8"))
    assert int(local_payload["pid"]) == os.getpid()
    assert int(global_payload["pid"]) == os.getpid()
    out = capsys.readouterr().out
    assert "INSTANCE_LOCK_STALE|path=" in out
    assert "INSTANCE_LOCK_ACQUIRED|local=" in out


def test_acquire_instance_lock_reports_live_owner_for_matching_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    streamer = _make_lock_streamer(tmp_path, alive_pids={20980})
    current_key = streamer._instance_lock_key()
    local_lock, _ = streamer._instance_lock_paths()
    _write_lock(local_lock, pid=20980, lock_key=current_key)

    def _replace_existing_owner_process(**kwargs) -> bool:
        return False

    streamer._replace_existing_owner_process = _replace_existing_owner_process

    with pytest.raises(RuntimeError, match=r"RUN_LOCK_FAIL\|reason=lock_held") as exc_info:
        streamer._acquire_instance_lock()

    msg = str(exc_info.value)
    assert "owner_run_id=OWNER_RUN" in msg
    assert "owner_preset=es_maxpack_10_full_send_prop_safe_pnl" in msg
    assert "owner_instrument=ES" in msg
    assert "owner_model=116037797a3c" in msg
    assert "hint=Get-Process -Id 20980; netstat -ano | findstr :5018" in msg
    out = capsys.readouterr().out
    assert "RUN_LOCK_OWNER|reason=lock_held|path=" in out
    assert "RUN_LOCK_ACTION|hint=Get-Process -Id 20980; netstat -ano | findstr :5018" in out


def test_acquire_instance_lock_replaces_same_key_live_owner_automatically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    streamer = _make_lock_streamer(tmp_path, alive_pids={20980})
    calls: list[tuple[int, int]] = []

    def _replace_existing_owner_process(**kwargs) -> bool:
        calls.append((int(kwargs["pid"]), int(kwargs["port"])))
        return True

    streamer._replace_existing_owner_process = _replace_existing_owner_process
    local_lock, _ = streamer._instance_lock_paths()
    _write_lock(local_lock, pid=20980, lock_key=streamer._instance_lock_key())

    streamer._acquire_instance_lock()

    assert calls == [(20980, 5018)]
    assert streamer._instance_lock_acquired is True
    payload = json.loads(local_lock.read_text(encoding="utf-8"))
    assert int(payload["pid"]) == os.getpid()
    out = capsys.readouterr().out
    assert "INSTANCE_LOCK_ACQUIRED|local=" in out


def test_acquire_instance_lock_reports_port_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    streamer = _make_lock_streamer(tmp_path, alive_pids={20980})
    local_lock, _ = streamer._instance_lock_paths()
    _write_lock(local_lock, pid=20980, lock_key=streamer._instance_lock_key(), port=5020)

    with pytest.raises(RuntimeError, match=r"RUN_LOCK_FAIL\|reason=port_mismatch") as exc_info:
        streamer._acquire_instance_lock()

    msg = str(exc_info.value)
    assert "lock_port=5020" in msg
    assert "hint=Get-Process -Id 20980; netstat -ano | findstr :5020" in msg
    out = capsys.readouterr().out
    assert "RUN_LOCK_OWNER|reason=port_mismatch|path=" in out


def test_acquire_instance_lock_reports_lock_key_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    streamer = _make_lock_streamer(tmp_path, alive_pids={20980})
    local_lock, _ = streamer._instance_lock_paths()
    _write_lock(local_lock, pid=20980, lock_key="other-model|other-preset|ES|5018")

    with pytest.raises(RuntimeError, match=r"RUN_LOCK_FAIL\|reason=lock_key_mismatch") as exc_info:
        streamer._acquire_instance_lock()

    msg = str(exc_info.value)
    assert "lock_key=other-model|other-preset|ES|5018" in msg
    assert "hint=Get-Process -Id 20980; netstat -ano | findstr :5018" in msg
    out = capsys.readouterr().out
    assert "RUN_LOCK_OWNER|reason=lock_key_mismatch|path=" in out


def test_write_status_is_noop_when_state_missing_during_init(tmp_path: Path) -> None:
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer._write_status = LiveCSVStreamer._write_status.__get__(streamer, LiveCSVStreamer)
    streamer._last_status_write_ts = 0.0
    streamer.status_path = tmp_path / "status.json"

    streamer._write_status(force=True)

    assert not streamer.status_path.exists()
