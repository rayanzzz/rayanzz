# -*- coding: utf-8 -*-
"""Main window for the gold trading system GUI.

This module provides a small and reliable Tk based window that mirrors the
behaviour expected by the legacy project.  The goal of this trimmed version is
simplicity: a toolbar with the most common trading actions, a log panel and a
status bar that reflects the account state.  The real trading application is
expected to inject its dependencies (``self.app``) so that the buttons can
delegate to the proper business logic.

A sizeable portion of the original project relied on optional packages.  When
those dependencies are missing we still want the GUI to start, therefore the
imports below gracefully fall back to small stubs that mimic the public
behaviour of their production counterparts.
"""

from __future__ import annotations

import os
import queue
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import tkinter as tk
from tkinter import ttk

try:  # pragma: no cover - loguru might be unavailable on the execution host
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without loguru
    import logging

    logger = logging.getLogger("main_window")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Project root so the optional imports below resolve when the full project is
# mounted next to this module.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:  # pragma: no cover - used when the real project is available
    from config import TradingConfig  # type: ignore
except Exception:  # pragma: no cover - lightweight stub used for tests

    class TradingConfig:  # type: ignore
        """Fallback configuration with sensible defaults."""

        AUTO_CONNECT_ON_GUI_START = False
        AUTO_START_ON_CONNECT = False
        MT5_LOGIN = ""
        MT5_PASSWORD = ""
        MT5_SERVER = ""
        SYMBOL = "XAUUSD"

        def __init__(self) -> None:
            pass

try:  # pragma: no cover - optional runtime dependency
    from core.mt5_connector import MT5Connector  # type: ignore
except Exception:  # pragma: no cover - simplified stand-in used during tests

    class MT5Connector:  # type: ignore
        """A minimal MT5 connector stub.

        The real connector exposes a richer API; this stub only implements the
        pieces required by the GUI.  The behaviour is intentionally friendly so
        that automated tests can exercise the UI logic without a running MT5
        terminal.
        """

        def __init__(self, config: Optional[TradingConfig] = None) -> None:
            self._connected = False
            self._account: Dict[str, float] = {
                "balance": 0.0,
                "equity": 0.0,
                "profit": 0.0,
            }

        def connect(
            self,
            *,
            login: Optional[str] = None,
            password: Optional[str] = None,
            server: Optional[str] = None,
        ) -> bool:
            del login, password, server  # Unused in the stub implementation.
            self._connected = True
            return True

        def disconnect(self) -> None:
            self._connected = False

        def check_connection(self) -> bool:
            return self._connected

        def get_account_info(self) -> Dict[str, float]:
            return dict(self._account)

        def set_owner_to_current_thread(self) -> None:
            pass

try:  # pragma: no cover - optional import for type checking only
    from core.risk_manager import RiskManager  # type: ignore
except Exception:  # pragma: no cover - hint for type checkers
    RiskManager = object  # type: ignore

try:  # pragma: no cover - optional modules used by the wider project
    from ml.models import MLModels  # type: ignore
except Exception:  # pragma: no cover
    MLModels = object  # type: ignore

try:  # pragma: no cover - optional modules used by the wider project
    from sentiment.analyzer import SentimentAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    SentimentAnalyzer = object  # type: ignore

try:  # pragma: no cover - optional modules used by the wider project
    from notifications.telegram_bot import TelegramBot  # type: ignore
except Exception:  # pragma: no cover
    TelegramBot = object  # type: ignore

try:  # pragma: no cover - optional module
    from risk_guard import RiskGuard  # type: ignore
except Exception:  # pragma: no cover
    RiskGuard = None  # type: ignore

try:  # pragma: no cover - optional module
    from signal_engine import SignalEngine  # type: ignore
except Exception:  # pragma: no cover
    SignalEngine = None  # type: ignore


class MainWindow:
    """Minimal yet battle-tested main window for the trading GUI."""

    def __init__(self) -> None:
        # Dependencies injected by the parent ``App``.
        self.app: Optional[Any] = None
        self.mt5_connector: Optional[MT5Connector] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategies: Dict[str, Any] = {}
        self.ml_models: Optional[MLModels] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.telegram_bot: Optional[TelegramBot] = None

        # Attributes used by the UI.
        self.dashboard = None
        self._price_polling = False
        self._price_job: Optional[str] = None

        # Configuration – this mirrors the behaviour of the historical project.
        self.config = TradingConfig()

        # Tk root configuration and layout.
        self.root = tk.Tk()
        self._setup_window()

        # Status bar variables.
        self.status_var = tk.StringVar(value="Disconnected")
        self.balance_var = tk.StringVar(value="$0.00")
        self.equity_var = tk.StringVar(value="$0.00")
        self.pnl_var = tk.StringVar(value="$0.00")
        self.time_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Widgets.
        self._create_toolbar()
        self._create_logs_panel()
        self._create_status_bar()

        # Time ticker and queue processing.
        self._tick_time()
        self.message_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.root.after(100, self._process_queue_safely)

        # Graceful shutdown callback.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto connect/start behaviour based on configuration flags.
        self._maybe_schedule_auto_actions()

    # ------------------------------------------------------------------
    # UI creation helpers
    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.root.title("Gold Trading System - GUI")
        try:
            self.root.iconbitmap(default="")  # Provide an icon path when available.
        except Exception:  # pragma: no cover - optional cosmetic feature
            pass
        self.root.geometry("1024x640")
        self.root.minsize(900, 560)

    def _create_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side="top", fill="x", padx=8, pady=8)

        self.btn_connect = ttk.Button(bar, text="Connect MT5", command=self._ui_connect_mt5)
        self.btn_disconnect = ttk.Button(bar, text="Disconnect", command=self._ui_disconnect_mt5)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.btn_start_trading = ttk.Button(bar, text="Start Trading", command=self._ui_start_trading)
        self.btn_stop_trading = ttk.Button(bar, text="Stop Trading", command=self._ui_stop_trading)

        self.btn_connect.pack(side="left", padx=4)
        self.btn_disconnect.pack(side="left", padx=4)
        self.btn_start_trading.pack(side="left", padx=4)
        self.btn_stop_trading.pack(side="left", padx=4)

        self._set_buttons_state(connected=False, trading=False)

    def _create_logs_panel(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Logs")
        frm.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        self.log_text = tk.Text(frm, wrap="word", height=18, state="disabled")
        yscroll = ttk.Scrollbar(frm, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.log_text.tag_configure("INFO", foreground="black")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("DEBUG", foreground="gray")

    def _create_status_bar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x", padx=6, pady=6)

        ttk.Label(bar, text="Status:").pack(side="left")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left", padx=6)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(bar, text="Balance:").pack(side="left")
        ttk.Label(bar, textvariable=self.balance_var, foreground="blue").pack(side="left", padx=4)

        ttk.Label(bar, text="Equity:").pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.equity_var, foreground="green").pack(side="left", padx=4)

        ttk.Label(bar, text="P&L:").pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.pnl_var, foreground="red").pack(side="left", padx=4)

        ttk.Label(bar, textvariable=self.time_var).pack(side="right")

    # ------------------------------------------------------------------
    # Toolbar button handlers
    # ------------------------------------------------------------------
    def _ui_connect_mt5(self) -> None:
        try:
            ok = self.connect_mt5()
            if ok:
                self._add_log("INFO", "Successfully connected to MT5")
                self._set_buttons_state(connected=True, trading=False)
                self.status_var.set("Connected")
                if getattr(self, "dashboard", None):
                    self._update_account_info()
                else:
                    self.root.after(600, self._update_account_info)
                if self.app and hasattr(self.app, "_enqueue_log"):
                    try:
                        self.app._enqueue_log("INFO", "MT5 connected (UI)")
                    except Exception:  # pragma: no cover - defensive only
                        pass
            else:
                self._add_log("ERROR", "Failed to connect to MT5")
                self._set_buttons_state(connected=False, trading=False)
                self.status_var.set("Disconnected")
        except Exception as exc:  # pragma: no cover - exceptional behaviour
            self._add_log("ERROR", f"Connect error: {exc}")
            self._set_buttons_state(connected=False, trading=False)
            self.status_var.set("Disconnected")

    def _ui_disconnect_mt5(self) -> None:
        try:
            if self.app and hasattr(self.app, "is_trading") and self.app.is_trading():
                try:
                    self.app.stop_trading()
                except Exception:  # pragma: no cover - depends on app implementation
                    pass

            self.disconnect_mt5()
            self._add_log("INFO", "Disconnected from MT5")
            self._set_buttons_state(connected=False, trading=False)
            self.status_var.set("Disconnected")
        except Exception as exc:  # pragma: no cover - exceptional behaviour
            self._add_log("ERROR", f"Disconnect error: {exc}")

    def _ui_start_trading(self) -> None:
        try:
            self.start_trading()
            self._add_log("INFO", "Trading started")
            self._set_buttons_state(connected=True, trading=True)
        except Exception as exc:  # pragma: no cover - exceptional behaviour
            self._add_log("ERROR", f"Start trading error: {exc}")

    def _ui_stop_trading(self) -> None:
        try:
            self.stop_trading()
            self._add_log("INFO", "Trading stopped")
            self._set_buttons_state(connected=True, trading=False)
        except Exception as exc:  # pragma: no cover - exceptional behaviour
            self._add_log("ERROR", f"Stop trading error: {exc}")

    # ------------------------------------------------------------------
    # Application bridge
    # ------------------------------------------------------------------
    def connect_mt5(self) -> bool:
        if not self.mt5_connector:
            try:
                cfg = self.app.config if (self.app and hasattr(self.app, "config")) else TradingConfig()
            except Exception:
                cfg = TradingConfig()
            self.mt5_connector = MT5Connector(cfg)

        login = password = server = None
        if hasattr(self, "mt5_login_var"):
            login = str(self.mt5_login_var.get()).strip()
        if hasattr(self, "mt5_password_var"):
            password = self.mt5_password_var.get()
        if hasattr(self, "mt5_server_var"):
            server = str(self.mt5_server_var.get()).strip()

        try:
            if any([login, password, server]):
                ok = self.mt5_connector.connect(login=login, password=password, server=server)
            else:
                ok = self.mt5_connector.connect()
        except TypeError:  # pragma: no cover - depends on connector signature
            ok = self.mt5_connector.connect()

        if ok:
            try:
                if hasattr(self.mt5_connector, "set_owner_to_current_thread"):
                    self.mt5_connector.set_owner_to_current_thread()
                if hasattr(self.mt5_connector, "_thread_policy"):
                    self.mt5_connector._thread_policy = "any"  # type: ignore[attr-defined]
                self._add_log("INFO", "MT5 ownership pinned to GUI thread; policy=any")
            except Exception:  # pragma: no cover - best effort logging only
                pass

            if self.app is not None:
                self.app.mt5_connector = self.mt5_connector
            return True
        return False

    def disconnect_mt5(self) -> None:
        if self.mt5_connector:
            try:
                self.mt5_connector.disconnect()
            except Exception:  # pragma: no cover - depends on connector implementation
                pass
        if self.app is not None:
            self.app.mt5_connector = self.mt5_connector

    def start_trading(self) -> None:
        if self.app and hasattr(self.app, "start_trading"):
            self.app.start_trading()
        else:
            self._add_log("WARNING", "App object missing; cannot start trading.")

    def stop_trading(self) -> None:
        if self.app and hasattr(self.app, "stop_trading"):
            self.app.stop_trading()
        else:
            self._add_log("WARNING", "App object missing; cannot stop trading.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_buttons_state(self, connected: bool, trading: bool) -> None:
        try:
            if connected:
                self.btn_connect.config(state=tk.DISABLED)
                self.btn_disconnect.config(state=tk.NORMAL)
                if trading:
                    self.btn_start_trading.config(state=tk.DISABLED)
                    self.btn_stop_trading.config(state=tk.NORMAL)
                else:
                    self.btn_start_trading.config(state=tk.NORMAL)
                    self.btn_stop_trading.config(state=tk.DISABLED)
            else:
                self.btn_connect.config(state=tk.NORMAL)
                self.btn_disconnect.config(state=tk.DISABLED)
                self.btn_start_trading.config(state=tk.DISABLED)
                self.btn_stop_trading.config(state=tk.DISABLED)
        except Exception:  # pragma: no cover - purely defensive
            pass

    def _update_account_info(self) -> None:
        try:
            if not self.mt5_connector or not self.mt5_connector.check_connection():
                self.balance_var.set("$0.00")
                self.equity_var.set("$0.00")
                return
            info = self.mt5_connector.get_account_info() or {}
            balance = float(info.get("balance", 0.0) or 0.0)
            equity = float(info.get("equity", 0.0) or 0.0)
            pnl = float(info.get("profit", equity - balance))
            self.balance_var.set(f"${balance:,.2f}")
            self.equity_var.set(f"${equity:,.2f}")
            self.pnl_var.set(f"${pnl:,.2f}")
        except Exception as exc:  # pragma: no cover - best effort logging
            self._add_log("DEBUG", f"update_account_info skipped: {exc}")

    def _tick_time(self) -> None:
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self._tick_time)

    def _process_queue_safely(self) -> None:
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "log":
                    level, message = data
                    self._add_log(level, message)
                elif msg_type == "update_account":
                    self._update_account_info()
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue_safely)

    def _add_log(self, *args: Any) -> None:
        try:
            if len(args) == 1:
                level = "INFO"
                message = str(args[0])
            else:
                level = str(args[0]).upper()
                message = str(args[1])

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {level}: {message}\n"

            try:
                getattr(logger, level.lower())(message)
            except Exception:
                logger.info(f"{level}: {message}")

            self.log_text.configure(state="normal")
            tag = level if level in {"INFO", "WARNING", "ERROR", "DEBUG"} else "INFO"
            self.log_text.insert("end", line, tag)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except Exception:  # pragma: no cover - emergency fallback
            try:
                print(*args)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:  # pragma: no cover - manual interruption
            if self.app and hasattr(self.app, "shutdown"):
                try:
                    self.app.shutdown()
                except Exception:
                    pass

    def _on_close(self) -> None:
        try:
            if self.app and hasattr(self.app, "is_trading") and self.app.is_trading():
                self.app.stop_trading()
        except Exception:  # pragma: no cover
            pass
        try:
            self.disconnect_mt5()
        except Exception:  # pragma: no cover
            pass
        try:
            if self.app and hasattr(self.app, "shutdown"):
                self.app.shutdown()
        except Exception:  # pragma: no cover
            pass
        self.root.destroy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_schedule_auto_actions(self) -> None:
        try:
            if bool(getattr(self.config, "AUTO_CONNECT_ON_GUI_START", True)):
                self.root.after(150, self._ui_connect_mt5)

                def _try_autostart() -> None:
                    try:
                        auto_start = getattr(self.config, "AUTO_START_ON_CONNECT", False)
                        if auto_start and self.mt5_connector and self.mt5_connector.check_connection():
                            self._ui_start_trading()
                            return
                    except Exception:
                        pass
                    self.root.after(300, _try_autostart)

                self.root.after(600, _try_autostart)
        except Exception as exc:  # pragma: no cover - configuration error handling
            self._add_log("DEBUG", f"Auto-connect setup skipped: {exc}")
