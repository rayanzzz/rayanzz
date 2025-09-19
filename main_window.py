"""Simplified trading UI logic with sentiment analysis helpers."""
from __future__ import annotations

import asyncio
import inspect
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
        """Synchronously resolve ``value`` when it is awaitable.

        The helper needs to cope with a variety of awaitable types used by the
        larger project and the pared-down tests: raw coroutines, tasks and
        futures bound to specific event loops, as well as analysers that remain
        synchronous.  The implementation therefore

        1. returns the value unchanged for non-awaitables,
        2. short-circuits already finished futures,
        3. reuses the awaitable's event loop when it is idle,
        4. mirrors ``asyncio.run`` semantics by spinning up a temporary loop, and
        5. executes work thread-safely when the awaitable is tied to a running
           loop in another thread.

        Awaitables bound to the *currently* running loop cannot be resolved
        synchronously without risking a dead-lock; in that rare case we re-raise
        a descriptive ``RuntimeError`` so callers can decide how to proceed.
        """

        if not inspect.isawaitable(value):
            return value

        if isinstance(value, asyncio.Future):
            if value.done():
                return value.result()

            try:
                target_loop = value.get_loop()
            except RuntimeError:
                target_loop = None

            if target_loop is not None:
                if target_loop.is_running():
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None

                    if running_loop is target_loop:
                        raise RuntimeError(
                            "Cannot synchronously resolve awaitable attached to the current running event loop."
                        )

                    async def _await_on_running_loop() -> Any:
                        return await value

                    proxy = asyncio.run_coroutine_threadsafe(
                        _await_on_running_loop(), target_loop
                    )
                    return proxy.result()

                return self._run_awaitable_in_loop(value, loop=target_loop)

        return self._run_awaitable_in_loop(value)

    def _run_awaitable_in_loop(
        self, value: Any, *, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Any:
        """Execute ``value`` inside ``loop`` (or a temporary loop) and return the result."""

        created_loop = loop is None
        event_loop = loop or asyncio.new_event_loop()
        previous_loop: Optional[asyncio.AbstractEventLoop]
        need_restore = False
        try:
            try:
                previous_loop = asyncio.get_event_loop()
            except RuntimeError:
                previous_loop = None

            if previous_loop is not event_loop:
                asyncio.set_event_loop(event_loop)
                need_restore = True

            async def _consume() -> Any:
                return await value

            task = event_loop.create_task(_consume())
            event_loop.run_until_complete(task)
            return task.result()
        finally:
            if need_restore:
                if previous_loop is None:
                    asyncio.set_event_loop(None)
                else:
                    asyncio.set_event_loop(previous_loop)

            if created_loop:
                event_loop.close()

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
