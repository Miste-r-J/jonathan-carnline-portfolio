"""Stop/target sign validation used by the offset-mode pre-submission reject.

LONG requires stop < entry < target; SHORT requires target < entry < stop.
A missing target is a legal stop-only bracket; a missing stop is never valid.
"""
from trading_system.runtime_engine.integrations.cli.live_trading_runtime import bracket_geometry_valid


def test_valid_long_bracket():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=99.0, target_price=102.0) is True


def test_invalid_long_stop_above_entry():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=101.0, target_price=102.0) is False


def test_invalid_long_target_below_entry():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=99.0, target_price=99.5) is False


def test_valid_short_bracket():
    assert bracket_geometry_valid("SHORT", entry_ref=100.0, stop_price=102.0, target_price=98.0) is True


def test_invalid_short_stop_below_entry():
    assert bracket_geometry_valid("SHORT", entry_ref=100.0, stop_price=99.0, target_price=98.0) is False


def test_invalid_short_target_above_entry():
    assert bracket_geometry_valid("SHORT", entry_ref=100.0, stop_price=102.0, target_price=101.0) is False


def test_long_stop_only_bracket_valid():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=99.0, target_price=None) is True


def test_short_stop_only_bracket_valid():
    assert bracket_geometry_valid("SHORT", entry_ref=100.0, stop_price=101.0, target_price=None) is True


def test_long_stop_only_inverted_invalid():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=101.0, target_price=None) is False


def test_missing_stop_is_invalid():
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=None, target_price=102.0) is False


def test_missing_entry_is_invalid():
    assert bracket_geometry_valid("LONG", entry_ref=None, stop_price=99.0, target_price=102.0) is False


def test_equal_stop_and_entry_invalid():
    # stop == entry is not strictly below; must reject (no zero-distance stop).
    assert bracket_geometry_valid("LONG", entry_ref=100.0, stop_price=100.0, target_price=102.0) is False


def test_unknown_side_invalid():
    assert bracket_geometry_valid("", entry_ref=100.0, stop_price=99.0, target_price=102.0) is False


def test_offset_mode_reference_classifies_by_sign():
    # In offset mode entry_ref is the *expected* entry, which differs from the
    # eventual fill, but the sign relationship still classifies correctly: a stop
    # below the reference and target above is a valid long regardless of the gap.
    assert bracket_geometry_valid("LONG", entry_ref=5000.0, stop_price=4992.0, target_price=5012.0) is True
    assert bracket_geometry_valid("SHORT", entry_ref=5000.0, stop_price=5008.0, target_price=4988.0) is True
