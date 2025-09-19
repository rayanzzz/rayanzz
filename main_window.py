"""Simplified trading UI logic with sentiment analysis helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional


class SimpleVar:
    """A lightweight stand-in for ``tk.StringVar`` used in tests."""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def set(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value


class SimpleListbox:
    """A minimal listbox implementation that mimics Tk's API for tests."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self._selection: tuple[int, ...] = ()

    def set_items(self, items: list[str]) -> None:
        self._items = list(items)

    def select(self, index: Optional[int]) -> None:
        if index is None:
            self._selection = ()
        else:
            self._selection = (index,)

    def curselection(self) -> tuple[int, ...]:
        return self._selection

    def get(self, index: int) -> str:
        return self._items[index]


class SimpleMessageBox:
    """Collects info/error calls instead of showing GUI dialogs."""

    def __init__(self) -> None:
        self.infos: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []

    def showinfo(self, title: str, message: str) -> None:
        self.infos.append((title, message))

    def showerror(self, title: str, message: str) -> None:
        self.errors.append((title, message))


class SentimentAnalyzer:
    """Minimal stub used for dependency injection in tests."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

    def analyze_text(self, text: str) -> Dict[str, Any]:
        # Provide deterministic response so tests can inspect formatting.
        return {"label": "neutral", "score": 0.0}


class MainWindow:
    """Subset of the GUI class focusing on news sentiment handling."""

    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        messagebox: Optional[SimpleMessageBox] = None,
        news_listbox: Optional[SimpleListbox] = None,
    ) -> None:
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.messagebox = messagebox or SimpleMessageBox()
        self.news_listbox = news_listbox or SimpleListbox()
        self.sentiment_var = SimpleVar()
        self.sentiment_metrics: Dict[str, SimpleVar] = {"Confidence": SimpleVar()}
        self._log_history: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Utilities
    def _await_if_needed(self, value: Any) -> Any:
        return value

    def _add_log(self, level: str, message: str) -> None:
        self._log_history.append((level, message))

    # ------------------------------------------------------------------
    # News sentiment
    def analyze_news_impact(self) -> None:
        try:
            selection = self.news_listbox.curselection()
            if not selection:
                self.messagebox.showinfo("Info", "Select a news item first.")
                return

            text = self.news_listbox.get(selection[0])

            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = SentimentAnalyzer(self.config)

            result: Dict[str, Any] = {}
            try:
                res = self.sentiment_analyzer.analyze_text(text)
                result = self._await_if_needed(res) or {}
            except Exception as exc:  # pragma: no cover - defensive branch
                self._add_log("WARNING", f"Sentiment analyze fallback: {exc}")
                result = {"label": "neutral", "score": 0.0}

            label = str(result.get("label", "neutral")).capitalize()
            score = float(result.get("score", 0.0))
            self.sentiment_var.set(label)
            # Ensure two fixed decimal places, matching the dialog formatting.
            self.sentiment_metrics["Confidence"].set(f"{score:.2f}")
            self.messagebox.showinfo(
                "Sentiment", f"{label}  (confidence={score:.2f})"
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            self._add_log("ERROR", f"Analyze news error: {exc}")
            self.messagebox.showerror("Error", str(exc))
