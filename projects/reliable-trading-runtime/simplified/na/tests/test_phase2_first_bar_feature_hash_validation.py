from __future__ import annotations


def test_feature_hash_validator_raises_on_first_mismatch() -> None:
    from na.bot.exceptions import FeatureMismatchError
    from na.bot.feature_hash import FeatureHashOnceValidator, compute_feature_hash

    required = ["f1", "f2", "f3"]
    expected = compute_feature_hash(required)
    v = FeatureHashOnceValidator(expected_hash=expected, required_features=required)

    try:
        v.validate(["f1", "f2"])  # missing f3
        assert False, "Expected FeatureMismatchError"
    except FeatureMismatchError:
        pass


def test_feature_hash_validator_only_runs_once_after_success() -> None:
    from na.bot.feature_hash import FeatureHashOnceValidator, compute_feature_hash

    required = ["f1", "f2", "f3"]
    expected = compute_feature_hash(required)
    v = FeatureHashOnceValidator(expected_hash=expected, required_features=required)

    live_hash = v.validate(["f1", "f2", "f3", "extra"])
    assert live_hash == expected
    assert v.validated is True

    # Once validated, it must not re-validate on subsequent calls.
    # (Feature columns are assumed invariant during a live run; this avoids per-bar overhead.)
    assert v.validate(["f1"]) is None

