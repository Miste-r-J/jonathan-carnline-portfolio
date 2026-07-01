from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictionState:
    last_bar_ts: Optional[str] = None
    last_features_hash: Optional[str] = None
    last_prediction_id: Optional[str] = None
    model_config_snapshot: Dict[str, Any] = field(default_factory=dict)
    session_stats: Dict[str, Any] = field(default_factory=dict)
    feature_hashes: List[str] = field(default_factory=list)

    def update(self, *, bar_ts: str, features_hash: str, prediction_id: str) -> None:
        self.last_bar_ts = bar_ts
        self.last_features_hash = features_hash
        self.last_prediction_id = prediction_id
        if features_hash:
            self.feature_hashes.append(features_hash)
