from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .exceptions import FeatureMismatchError


def compute_feature_hash(feature_names: Sequence[str]) -> str:
    """
    Deterministic schema hash matching simplified/tools/train_phase2_grid.py:_feature_hash().

    - Sorts feature names (order-independent)
    - Joins with '|'
    - sha256 hexdigest truncated to 16 chars
    """
    joined = "|".join(sorted(map(str, feature_names)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def validate_feature_schema_and_hash(
    live_columns: Sequence[str],
    *,
    required_features: Sequence[str],
    expected_hash: str,
) -> str:
    required_list = [str(x) for x in required_features]
    required_set = set(required_list)
    live_set = {str(c) for c in live_columns}

    missing = [c for c in required_list if c not in live_set]
    if missing:
        raise FeatureMismatchError(f"Feature schema mismatch: missing {len(missing)} required feature(s): {missing}")

    live_required = [c for c in live_columns if str(c) in required_set]
    live_hash = compute_feature_hash(live_required)
    if live_hash != str(expected_hash):
        raise FeatureMismatchError(
            "Feature hash mismatch. Model expects "
            f"{expected_hash}, got {live_hash}. Aborting."
        )
    return live_hash


@dataclass
class FeatureHashOnceValidator:
    expected_hash: str
    required_features: Sequence[str]
    validated: bool = False

    def validate(self, live_columns: Sequence[str]) -> str | None:
        if self.validated:
            return None
        live_hash = validate_feature_schema_and_hash(
            live_columns,
            required_features=self.required_features,
            expected_hash=self.expected_hash,
        )
        self.validated = True
        return live_hash
