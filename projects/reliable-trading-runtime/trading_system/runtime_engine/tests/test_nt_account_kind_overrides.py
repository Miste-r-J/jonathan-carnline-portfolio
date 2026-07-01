from trading_system.runtime_engine.integrations.cli.live_trading_runtime import LiveCSVStreamer


def test_account_kind_override_demo_treated_as_live():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    assert streamer._account_kind("DEMO6730705") == "live"
    assert streamer._account_kind("DEMO6927902") == "live"
    assert streamer._account_kind("DEMO7321116") == "live"
    assert streamer._account_kind("DEMO7989060") == "live"
    assert streamer._account_kind("DEMO8142346") == "live"
    assert streamer._account_kind("DEMO123") == "demo"


def test_account_kind_prop_firm_account_treated_as_live():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    assert streamer._account_kind("MFFUEVFLX447934002") == "live"
    assert streamer._account_kind("MFFUEVRPD447934003") == "live"


def test_account_allowed_with_expected_live_respects_override():
    streamer = LiveCSVStreamer.__new__(LiveCSVStreamer)
    streamer.nt_account_expected_kind = "live"
    streamer.nt_allowed_accounts = []
    streamer.nt_allow_unlisted_accounts = False
    streamer.nt_account_allow_prefix = None
    streamer.nt_account_allow_regex = None

    assert streamer._account_allowed("DEMO6730705") is True
    assert streamer._account_allowed("DEMO6927902") is True
    assert streamer._account_allowed("DEMO7321116") is True
    assert streamer._account_allowed("DEMO7989060") is True
    assert streamer._account_allowed("DEMO8142346") is True
    assert streamer._account_allowed("MFFUEVFLX447934002") is True
    assert streamer._account_allowed("MFFUEVRPD447934003") is True
    assert streamer._account_allowed("DEMO123") is False
