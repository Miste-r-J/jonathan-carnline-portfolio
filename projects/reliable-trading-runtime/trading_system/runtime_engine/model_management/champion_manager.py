from __future__ import annotations

"""
Tiny champion-manager shim used by the standalone streamer.

The real system stores promotion history in a database.  We keep a filesystem-based
reader so operators can drop ``<symbol>.json`` files into the champion state
directory and have the streamer pick them up automatically.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ChampionRecord:
    symbol: str
    model_path: str
    model_id: Optional[int] = None
    metadata: dict[str, object] = None  # type: ignore[assignment]


class ChampionManager:
    def __init__(
        self,
        state_dir: str | Path,
        *,
        ev_margin: float = 0.0,
        promotion_min_trades: int = 0,
    ) -> None:
        self.state_dir = Path(state_dir or "artifacts/champions_state").expanduser()
        self.ev_margin = float(ev_margin)
        self.promotion_min_trades = int(promotion_min_trades)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load_current(self, instrument: str) -> Optional[ChampionRecord]:
        """
        Load ``<symbol>.json`` from ``state_dir`` if present.

        Expected schema:
            {"symbol": "ES", "model_path": "artifacts/es.joblib", "model_id": 123}
        """
        symbol = str(instrument or "").upper()
        if not symbol:
            return None
        candidate = self.state_dir / f"{symbol}.json"
        if not candidate.exists():
            return None
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            return None
        model_path = payload.get("model_path")
        if not model_path:
            return None
        return ChampionRecord(
            symbol=symbol,
            model_path=str(model_path),
            model_id=payload.get("model_id"),
            metadata={k: v for k, v in payload.items() if k not in {"model_path", "model_id", "symbol"}},
        )


__all__ = ["ChampionManager", "ChampionRecord"]
