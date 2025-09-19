"""Core trading system logic for orchestrating gold trades.

This module introduces :class:`GoldTradingSystem`, a light-weight manager that
keeps track of closed positions alongside performance metrics.  The class is
kept intentionally independent from the MT5 connector so it can be exercised in
unit tests without requiring a terminal connection.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, time as dt_time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["GoldTradingSystem"]


class GoldTradingSystem:
    """Stateful manager for strategy level bookkeeping.

    Parameters
    ----------
    notifier:
        Optional callable that gets invoked whenever a position is closed.
        The callable receives the dictionary that is appended to
        ``trading_history``.
    max_history_records:
        Maximum number of closed trade records to keep in memory.  When the
        limit is exceeded the oldest entries are discarded to avoid unbounded
        growth.
    """

    def __init__(
        self,
        *,
        notifier: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_history_records: int = 1000,
    ) -> None:
        self.notifier = notifier
        self._max_history_records = max_history_records if max_history_records > 0 else 0
        self.trading_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_trades": 0,
            "net_profit": 0.0,
            "wins": 0,
            "losses": 0,
            "largest_win": None,
            "largest_loss": None,
            "average_profit": 0.0,
            "win_rate": 0.0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_trade_payload(self, trade: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        if trade is None:
            return {}
        if isinstance(trade, dict):
            return dict(trade)

        # Attribute-style objects (e.g. MT5 position namedtuples)
        candidate_keys = {
            "symbol",
            "direction",
            "type",
            "entry_price",
            "open_price",
            "price_open",
            "exit_price",
            "price_close",
            "close_price",
            "profit",
            "net_profit",
            "ticket",
            "order",
            "position",
            "volume",
            "close_time",
            "time_close",
            "close_at",
        }
        payload: Dict[str, Any] = {}
        for key in candidate_keys:
            if hasattr(trade, key):
                payload[key] = getattr(trade, key)
        return payload

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime()  # type: ignore[no-any-return]
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, date):
            return datetime.combine(value, dt_time.min)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            for parser in (datetime.fromisoformat, self._try_parse_datetime):
                try:
                    parsed = parser(value)
                except Exception:
                    continue
                if parsed:
                    return parsed
        return datetime.now()

    @staticmethod
    def _try_parse_datetime(value: str) -> Optional[datetime]:
        fmt_candidates = (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%d",
        )
        for fmt in fmt_candidates:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _append_to_history(self, record: Dict[str, Any]) -> None:
        self.trading_history.append(record)
        if self._max_history_records and len(self.trading_history) > self._max_history_records:
            # Drop oldest entries while keeping list semantics
            del self.trading_history[:-self._max_history_records]

    def _update_performance_metrics(self, trade: Dict[str, Any]) -> None:
        metrics = self.performance_metrics
        profit = float(trade.get("profit", 0.0) or 0.0)

        metrics["total_trades"] += 1
        metrics["net_profit"] += profit

        if profit > 0:
            metrics["wins"] += 1
            metrics["largest_win"] = (
                profit
                if metrics["largest_win"] is None
                else max(metrics["largest_win"], profit)
            )
        elif profit < 0:
            metrics["losses"] += 1
            metrics["largest_loss"] = (
                profit
                if metrics["largest_loss"] is None
                else min(metrics["largest_loss"], profit)
            )

        metrics["average_profit"] = (
            metrics["net_profit"] / metrics["total_trades"]
            if metrics["total_trades"]
            else 0.0
        )
        completed = metrics["wins"] + metrics["losses"]
        metrics["win_rate"] = metrics["wins"] / completed if completed else 0.0

    def _notify_position_closed(self, trade: Dict[str, Any]) -> None:
        if self.notifier is None:
            logger.info(
                "Position closed | %s %s profit=%s",
                trade.get("symbol"),
                trade.get("direction"),
                trade.get("profit"),
            )
            return

        try:
            self.notifier(trade)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Notifier raised an exception while processing closed trade notification")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_position_closed(
        self,
        trade: Optional[Dict[str, Any]] = None,
        *,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        profit: Optional[float] = None,
        ticket: Optional[int] = None,
        close_time: Any = None,
        volume: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Bookkeeping hook invoked when a position closes.

        Parameters are intentionally permissive to accommodate multiple call
        styles.  ``trade`` may be a mapping or an attribute style object as
        returned by MetaTrader5, while keyword arguments allow overriding /
        supplementing the extracted information.
        """

        payload = self._normalize_trade_payload(trade)

        provided_fields = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "ticket": ticket,
            "close_time": close_time,
            "volume": volume,
        }
        for key, value in provided_fields.items():
            if value is not None:
                payload[key] = value

        if "direction" not in payload and "type" in payload:
            payload["direction"] = payload["type"]

        if "entry_price" not in payload:
            for alt in ("price_open", "open_price"):
                if alt in payload:
                    payload["entry_price"] = payload[alt]
                    break
        if "exit_price" not in payload:
            for alt in ("price_close", "close_price"):
                if alt in payload:
                    payload["exit_price"] = payload[alt]
                    break
        if "profit" not in payload and "net_profit" in payload:
            payload["profit"] = payload["net_profit"]
        if "ticket" not in payload:
            for alt in ("order", "position"):
                if alt in payload:
                    payload["ticket"] = payload[alt]
                    break
        if "close_time" not in payload:
            for alt in ("time_close", "close_at"):
                if alt in payload:
                    payload["close_time"] = payload[alt]
                    break

        close_dt = self._coerce_datetime(payload.get("close_time"))
        close_date = close_dt.date()

        direction_value = payload.get("direction")
        direction_value = (str(direction_value).upper() if direction_value is not None else None)
        entry_value = self._to_float(payload.get("entry_price"))
        exit_value = self._to_float(payload.get("exit_price"))
        profit_value = self._to_float(payload.get("profit")) or 0.0
        ticket_value = self._to_int(payload.get("ticket"))
        volume_value = self._to_float(payload.get("volume"))

        record: Dict[str, Any] = {
            "date": close_date,
            "closed_at": close_dt,
            "symbol": payload.get("symbol"),
            "direction": direction_value,
            "entry_price": entry_value,
            "exit_price": exit_value,
            "profit": profit_value,
            "ticket": ticket_value,
        }
        if volume_value is not None:
            record["volume"] = volume_value
        if extra:
            record.update(extra)

        self._append_to_history(record)

        # Preserve downstream behaviour: existing performance updates and
        # notifications should execute after recording the trade.
        self._update_performance_metrics(record)
        self._notify_position_closed(record)

    def generate_daily_report(self, target_date: Optional[Union[date, datetime]] = None) -> Dict[str, Any]:
        """Aggregate closed trades for ``target_date``.

        Parameters
        ----------
        target_date:
            Date for which the report is requested.  ``datetime`` values are
            normalised to a date to support callers passing timestamps.
        """

        if target_date is None:
            target = datetime.now().date()
        elif isinstance(target_date, datetime):
            target = target_date.date()
        else:
            target = target_date

        trades = [trade for trade in self.trading_history if trade.get("date") == target]
        profit_sum = sum(float(trade.get("profit") or 0.0) for trade in trades)
        wins = [trade for trade in trades if float(trade.get("profit") or 0.0) > 0.0]
        losses = [trade for trade in trades if float(trade.get("profit") or 0.0) < 0.0]
        count = len(trades)

        return {
            "date": target,
            "trades": trades,
            "count": count,
            "total_profit": profit_sum,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / count if count else 0.0,
            "average_profit": profit_sum / count if count else 0.0,
        }

    def trim_history(self, keep_last: int) -> None:
        """Reduce ``trading_history`` length to ``keep_last`` records."""

        if keep_last <= 0:
            self.trading_history.clear()
            return
        if len(self.trading_history) > keep_last:
            del self.trading_history[:-keep_last]
        self._max_history_records = keep_last

    def extend_history(self, trades: Iterable[Dict[str, Any]]) -> None:
        """Bulk add existing trade records into the history."""

        for trade in trades:
            if "date" not in trade and "closed_at" in trade:
                closed_at = self._coerce_datetime(trade["closed_at"])
                trade = dict(trade)
                trade["date"] = closed_at.date()
            self._append_to_history(dict(trade))
