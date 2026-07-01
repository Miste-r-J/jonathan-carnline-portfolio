from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer


class _Dummy:
    pass


def test_qa_emit_defaults_from_init() -> None:
    dummy = _Dummy()
    dummy.lookback_bars = 900
    state = LiveCSVStreamer._init_qa_emit_state(dummy, "qa_full_trade")
    assert state is not None
    assert int(state["open_index"]) == int(state["warmup_bars"]) + 5
    assert int(state["stop_update_index"]) < int(state["open_index"])


def test_qa_emit_open_index_moves_to_tail_when_processed_rows_known() -> None:
    import pandas as pd
    from types import SimpleNamespace

    class DummyStreamer:
        qa_emit_signals = "model_master_hold"
        tick_size = 0.25
        instrument_risk_params = SimpleNamespace(min_stop_ticks=6)
        _signal_to_order_header_written = True

        def __init__(self) -> None:
            self._qa_emit_state = LiveCSVStreamer._init_qa_emit_state(self, "model_master_hold")

        def _round_to_tick(self, x):
            return float(x)

        def _grade_from_prob(self, prob: float, side: str) -> str:
            return "A+"

        def _init_signal_to_order_log(self) -> None:
            return None

    dummy = DummyStreamer()
    assert dummy._qa_emit_state is not None
    row = pd.Series({"Close": 6862.5})
    ts_dt = pd.Timestamp("2026-02-17T14:25:00-07:00")
    _ = LiveCSVStreamer._qa_scripted_events(
        dummy,
        row=row,
        bar_index=0,
        ts_dt=ts_dt,
        prob_override=0.55,
        gates_override=None,
        open_ready=True,
        processed_rows=840,
    )
    assert int(dummy._qa_emit_state["open_index"]) >= 830
