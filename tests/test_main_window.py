"""Tests for the lightweight MainWindow sentiment helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_window import MainWindow, SimpleListbox, SimpleMessageBox


class _RecordingAnalyzer:
    """Sentiment analyzer stub recording the analysed text."""

    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.calls: list[str] = []

    def analyze_text(self, text: str) -> Dict[str, Any]:
        self.calls.append(text)
        return self.result


class _AsyncAnalyzer(_RecordingAnalyzer):
    """Async variant mirroring coroutine-based analyzers in production."""

    async def analyze_text(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
        await asyncio.sleep(0)
        return super().analyze_text(text)


def _setup_window(news: list[str]) -> tuple[MainWindow, SimpleListbox, SimpleMessageBox]:
    listbox = SimpleListbox()
    listbox.set_items(news)
    messagebox = SimpleMessageBox()
    window = MainWindow(messagebox=messagebox, news_listbox=listbox)
    return window, listbox, messagebox


def test_analyze_news_requires_selection() -> None:
    window, _, messagebox = _setup_window(["Important headline"])

    window.analyze_news_impact()

    assert messagebox.infos == [("Info", "Select a news item first.")]
    assert messagebox.errors == []


@pytest.mark.parametrize(
    "analyzer_cls",
    (_RecordingAnalyzer, _AsyncAnalyzer),
    ids=("sync", "async"),
)
def test_analyze_news_formats_sentiment(analyzer_cls: type[_RecordingAnalyzer]) -> None:
    window, listbox, messagebox = _setup_window(["Gold surges on market optimism [Reuters]"])
    analyzer = analyzer_cls({"label": "bullish", "score": 0.7345})
    window.sentiment_analyzer = analyzer

    listbox.select(0)
    window.analyze_news_impact()

    assert analyzer.calls == ["Gold surges on market optimism [Reuters]"]
    assert window.sentiment_var.get() == "Bullish"
    # Confidence is displayed with two decimal places even when the analyzer
    # returns more precision.
    assert window.sentiment_metrics["Confidence"].get() == "0.73"
    assert messagebox.infos == [("Sentiment", "Bullish  (confidence=0.73)")]
