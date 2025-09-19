# -*- coding: utf-8 -*-
"""
MT5 Connector Module
مدیریت اتصال و عملیات معاملاتی در MetaTrader 5
(نسخه‌ی پایدار با مدیریت نخ، فالبک Filling Mode، و ابزارهای preload/history)
"""

from __future__ import annotations

import sys
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

# اجازه‌ی import از ریشه پروژه
sys.path.append("..")
from config import TradingConfig


# ============================ Data Models ============================

@dataclass
class SymbolInfo:
    """اطلاعات نماد معاملاتی"""

    symbol: str
    bid: float
    ask: float
    spread: int
    digits: int
    point: float
    trade_mode: int
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float
    swap_long: float
    swap_short: float
    margin_initial: float
    margin_maintenance: float


# ============================ Connector ============================


class MT5Connector:
    """
    کلاس مدیریت اتصال به MetaTrader 5

    نکتهٔ مهم دربارهٔ نخ‌ها (threads):
    - به‌صورت پیش‌فرض سیاست «rebind» فعال است تا با هر نخی که صدا می‌زنی کار کند (سازگار با GUI فعلی).
      لاگ‌های rebind به‌صورت throttle و DEBUG ثبت می‌شوند.
    - اگر در اپلیکیشن فقط یک نخ باید به MT5 دست بزند، در config بنویس:
        MT5_THREAD_POLICY = "owner_only"
      تا rebind غیرفعال شود و پایداری بالاتر برود.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.connected: bool = False
        self.last_ping: Optional[datetime] = None

        # Locks
        self._connection_lock = threading.Lock()
        self._api_lock = threading.RLock()  # سریال‌سازی همهٔ فراخوانی‌های MT5

        # Caches & state
        self._symbol_info_cache: Dict[str, SymbolInfo] = {}
        self._last_error: Optional[str] = None

        # مالک سشن MT5 (برای جلوگیری از Terminal: Call failed)
        self._owner_thread_id: Optional[int] = None

        # سیاست نخ‌ها
        self._thread_policy: str = str(getattr(self.config, "MT5_THREAD_POLICY", "rebind")).lower()
        # "rebind" | "owner_only"

        # Throttle لاگ‌های rebind
        self._rebind_log_last: float = 0.0
        self._rebind_log_cooldown: float = float(getattr(self.config, "REBIND_LOG_COOLDOWN_SEC", 3.0))
        self._log_rebind_as_debug: bool = bool(getattr(self.config, "MT5_LOG_REBIND_AS_DEBUG", True))

        # Retry تنظیمات عمومی
        self._default_retries: int = int(getattr(self.config, "MT5_DEFAULT_RETRIES", 3))
        self._default_retry_sleep: float = float(getattr(self.config, "MT5_RETRY_SLEEP", 0.35))

    # ------------------------- Utils (thread) -------------------------

    def _is_owner_thread(self) -> bool:
        return self._owner_thread_id is None or self._owner_thread_id == threading.get_ident()

    def set_owner_to_current_thread(self):
        """نخ جاری را به‌عنوان مالک سشن MT5 ثبت می‌کند (برای سیاست owner_only)."""

        self._owner_thread_id = threading.get_ident()
        logger.info(f"MT5 owner thread set to {self._owner_thread_id}")

    def _log_rebind(self, msg: str, level: str = "warning"):
        now = time.time()
        if now - self._rebind_log_last >= self._rebind_log_cooldown:
            self._rebind_log_last = now
            if self._log_rebind_as_debug:
                logger.debug(msg)
            else:
                getattr(logger, level)(msg)

    # ------------------------- Connection ----------------------------

    def connect(self) -> bool:
        """اتصال به MT5 با مدیریت خطا + پشتیبانی از مسیر سفارشی ترمینال"""

        with self._connection_lock:
            try:
                logger.info("Initializing MT5 connection...")
                init_path = getattr(self.config, "MT5_TERMINAL_PATH", None)
                with self._api_lock:
                    ok = mt5.initialize(path=init_path) if init_path else mt5.initialize()
                if not ok:
                    self._last_error = str(mt5.last_error())
                    logger.error(f"MT5 initialization failed: {self._last_error}")
                    return False

                if getattr(self.config, "PAPER_TRADING", True):
                    logger.info("Running in PAPER TRADING mode")
                else:
                    with self._api_lock:
                        authorized = mt5.login(
                            login=int(self.config.MT5_LOGIN),
                            password=str(self.config.MT5_PASSWORD),
                            server=str(self.config.MT5_SERVER),
                            timeout=int(getattr(self.config, "MT5_TIMEOUT", 60000)),
                        )
                    if not authorized:
                        self._last_error = str(mt5.last_error())
                        logger.error(f"Login failed: {self._last_error}")
                        with self._api_lock:
                            try:
                                mt5.shutdown()
                            except Exception:
                                pass
                        return False

                # Verify / resolve symbol
                if not self._verify_and_resolve_symbol(self.config.SYMBOL):
                    logger.error(f"Symbol {self.config.SYMBOL} not available")
                    with self._api_lock:
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                    return False

                self.connected = True
                self.last_ping = datetime.now()
                self._owner_thread_id = threading.get_ident()  # ثبت نخ صاحب

                with self._api_lock:
                    account = mt5.account_info()
                if account:
                    logger.success(f"Connected to MT5: {account.server}")
                    logger.info(f"Account: {account.login} | Balance: ${account.balance:.2f}")

                # --- پری‌لود تاریخچه پس از اتصال (قابل پیکربندی) ---
                if bool(getattr(self.config, "AUTO_PRELOAD_HISTORY", True)):
                    try:
                        tfs = getattr(self.config, "AUTO_PRELOAD_TFS", ["M5", "M15", "H1", "H4"])
                        days = int(getattr(self.config, "AUTO_PRELOAD_DAYS", 30))
                        self.preload_history(self.config.SYMBOL, timeframes=list(tfs), days=days)
                    except Exception as e:
                        logger.debug(f"preload_history skipped: {e}")

                return True

            except Exception as e:
                self._last_error = str(e)
                logger.exception(f"Connection error: {e}")
                return False

    def disconnect(self):
        """قطع اتصال از MT5"""

        with self._api_lock:
            if self.connected:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
        self.connected = False
        self._owner_thread_id = None
        logger.info("Disconnected from MT5")

    def _rebind_to_current_thread(self) -> bool:
        """
        سشن MT5 را به نخ فعلی منتقل می‌کند (حل مشکل Terminal: Call failed).
        اگر MT5_THREAD_POLICY = "owner_only"، rebind انجام نمی‌شود و خطا می‌دهیم.
        """

        if self._thread_policy == "owner_only":
            raise RuntimeError(
                "MT5 called from a non-owner thread while policy=owner_only. "
                "Route all MT5 calls to the owner thread."
            )

        try:
            self._log_rebind("MT5 session called from a different thread; rebinding...")
            with self._api_lock:
                try:
                    mt5.shutdown()
                except Exception:
                    pass

                init_path = getattr(self.config, "MT5_TERMINAL_PATH", None)
                ok = mt5.initialize(path=init_path) if init_path else mt5.initialize()
                if not ok:
                    self._last_error = str(mt5.last_error())
                    logger.error(f"Re-init failed: {self._last_error}")
                    self.connected = False
                    return False

                if not getattr(self.config, "PAPER_TRADING", True):
                    ok = mt5.login(
                        login=int(self.config.MT5_LOGIN),
                        password=str(self.config.MT5_PASSWORD),
                        server=str(self.config.MT5_SERVER),
                        timeout=int(getattr(self.config, "MT5_TIMEOUT", 60000)),
                    )
                    if not ok:
                        self._last_error = str(mt5.last_error())
                        logger.error(f"Re-login failed: {self._last_error}")
                        self.connected = False
                        return False

                if not self._verify_and_resolve_symbol(self.config.SYMBOL):
                    logger.error(f"Symbol resolve failed on rebind: {self.config.SYMBOL}")
                    self.connected = False
                    return False

                self.connected = True
                self.last_ping = datetime.now()
                self._owner_thread_id = threading.get_ident()
                logger.debug("Rebound MT5 session to current thread")
                return True
        except Exception as e:
            logger.exception(f"Rebind error: {e}")
            self.connected = False
            return False

    def check_connection(self) -> bool:
        """بررسی وضعیت اتصال (با سیاست نخ و rebind خودکار در صورت نیاز)"""

        if not bool(self.connected):
            return False
        try:
            if not self._is_owner_thread():
                with self._connection_lock:
                    if not self._is_owner_thread():
                        if not self._rebind_to_current_thread():
                            return False

            with self._api_lock:
                term = mt5.terminal_info()
                if not term or not getattr(term, "connected", False):
                    self.connected = False
                    return False
                # ping سبک (بدون وابستگی به خروجی)
                test_sym = getattr(self.config, "SYMBOL", "XAUUSD")
                try:
                    mt5.symbol_info_tick(test_sym)
                except Exception:
                    pass

            self.last_ping = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self.connected = False
            return False

    def reconnect(self, max_attempts: int = 5) -> bool:
        """تلاش برای اتصال مجدد با backoff نمایی کوتاه"""

        logger.warning("Attempting to reconnect to MT5...")
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{max_attempts}")
            self.disconnect()
            time.sleep(min(2**attempt, 8))
            if self.connect():
                logger.success("Reconnection successful")
                return True
        logger.error("All reconnection attempts failed")
        return False

    @contextmanager
    def ensure_connected(self):
        """Context manager برای اطمینان از اتصال + اعمال سیاست نخ + rebind در صورت نیاز"""

        if not bool(self.connected):
            raise ConnectionError("Not connected to MT5")
        if not self._is_owner_thread():
            with self._connection_lock:
                if not self._is_owner_thread():
                    ok = self._rebind_to_current_thread()
                    if not ok:
                        raise ConnectionError("MT5 rebind to current thread failed")
        try:
            yield
        except Exception as e:
            logger.error(f"Error during MT5 operation: {e}")
            raise

    # -------------------- Symbol / Market Data -----------------------

    def _verify_and_resolve_symbol(self, symbol: str) -> bool:
        """بررسی در دسترس بودن نماد و resolve در صورت نیاز"""

        with self._api_lock:
            info = mt5.symbol_info(symbol)
            if info is None:
                all_syms = mt5.symbols_get() or []
                names = [s.name for s in all_syms]

                def norm(s: str) -> str:
                    return s.replace(".", "_").replace("-", "").upper()

                cand = next((n for n in names if norm(n) == norm(symbol)), None)
                if cand is None:
                    # fallback برای طلا
                    cand = next((n for n in names if "XAU" in n.upper() or "GOLD" in n.upper()), None)

                if cand is None:
                    return False

                if cand != symbol:
                    logger.warning(f"Requested symbol {symbol} resolved to {cand}")
                    self.config.SYMBOL = cand

                info = mt5.symbol_info(cand)
                if info is None:
                    return False
                symbol = cand

            if not info.visible:
                if not mt5.symbol_select(symbol, True):
                    return False
        return True

    # Backward-compat alias
    def _verify_symbol(self, symbol: str) -> bool:
        return self._verify_and_resolve_symbol(symbol)

    def get_symbol_info(self, symbol: str = None) -> Optional[SymbolInfo]:
        """دریافت اطلاعات سمبل (با کش یک دقیقه‌ای)"""

        symbol = symbol or self.config.SYMBOL
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        cached = self._symbol_info_cache.get(cache_key)
        if cached:
            return cached

        with self.ensure_connected():
            with self._api_lock:
                info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None

        symbol_info = SymbolInfo(
            symbol=info.name,
            bid=float(info.bid),
            ask=float(info.ask),
            spread=int(getattr(info, "spread", 0)),
            digits=int(info.digits),
            point=float(info.point),
            trade_mode=int(info.trade_mode),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            trade_contract_size=float(getattr(info, "trade_contract_size", 0.0)),
            swap_long=float(getattr(info, "swap_long", 0.0)),
            swap_short=float(getattr(info, "swap_short", 0.0)),
            margin_initial=float(getattr(info, "margin_initial", 0.0)),
            margin_maintenance=float(getattr(info, "margin_maintenance", 0.0)),
        )
        self._symbol_info_cache[cache_key] = symbol_info
        return symbol_info

    def get_candles(
        self,
        symbol: str = None,
        timeframe: str = "H1",
        count: int = 1000,
        start_pos: int = 0,
    ) -> Optional[pd.DataFrame]:
        """دریافت کندل‌ها (retry سبک + rebind قبل از reconnect + تضمین visible بودن نماد)"""

        symbol = symbol or self.config.SYMBOL

        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        tf = timeframe_map.get(str(timeframe).upper(), mt5.TIMEFRAME_H1)

        with self.ensure_connected():
            with self._api_lock:
                # visible بودن نماد
                info = mt5.symbol_info(symbol)
                if info and not info.visible:
                    mt5.symbol_select(symbol, True)

                rates = None
                last_err = None

                # مرحله 1: چند بار تلاش ساده
                for _ in range(self._default_retries):
                    rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
                    if rates is not None and len(rates) > 0:
                        break
                    last_err = mt5.last_error()
                    if last_err and isinstance(last_err, tuple) and "Call failed" in str(last_err[1]):
                        time.sleep(self._default_retry_sleep)
                        continue
                    else:
                        break

                # مرحله 2: اگر هنوز «Call failed»، یک rebind و یک تلاش دوباره
                if (
                    (rates is None or len(rates) == 0)
                    and last_err
                    and isinstance(last_err, tuple)
                    and "Call failed" in str(last_err[1])
                ):
                    if self._rebind_to_current_thread():
                        info = mt5.symbol_info(symbol)
                        if info and not info.visible:
                            mt5.symbol_select(symbol, True)
                        rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)

                # مرحله 3: اگر باز هم نشد، یک reconnect سبک (فقط یک‌بار)
                if rates is None or len(rates) == 0:
                    logger.error(f"Failed to get rates: {last_err or mt5.last_error()}")
                    if self.reconnect(max_attempts=1):
                        try:
                            time.sleep(0.6)  # ← مکث کوتاه بعد از initialize/login
                            info = mt5.symbol_info(symbol)
                            if info and not info.visible:
                                mt5.symbol_select(symbol, True)
                            rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
                        except Exception:
                            rates = None
                    if rates is None or len(rates) == 0:
                        return None

        try:
            df = pd.DataFrame(rates, copy=False)
        except Exception as e:
            logger.exception(f"Failed to convert rates to DataFrame: {e}")
            return None

        if "time" not in df.columns:
            logger.error(f"'time' column missing in MT5 rates (columns={list(df.columns)})")
            return None

        try:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=False)
            df.set_index("time", inplace=True, drop=True)
        except Exception as e:
            logger.exception(f"Failed to convert time/index: {e}")
            return None

        try:
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["hl2"] = (df["high"] + df["low"]) / 2.0
            df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
            df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            df["range"] = df["high"] - df["low"]
            df["body"] = (df["close"] - df["open"]).abs()
            df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
            df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
            s_info = self.get_symbol_info(symbol)
            if s_info:
                df["point"] = float(s_info.point)
        except Exception as e:
            logger.exception(f"Failed to enrich candle DataFrame: {e}")
            return None

        return df

    def preload_many(self, symbols, timeframes=None, days=30):
        """
        Preload historical data for multiple symbols/timeframes so
        strategies can compute indicators immediately after launch.
        """

        tfs = timeframes or ["M5", "M15", "H1", "H4"]
        for s in symbols:
            try:
                self.preload_history(s, timeframes=tfs, days=days)
                logger.info(f"[preload_many] OK: {s} {tfs} ({days}d)")
            except Exception as e:
                logger.warning(f"[preload_many] failed for {s}: {e}")

    def get_tick_price(self, symbol: str) -> float:
        """قیمت میانگین تیک (mid) اگر bid/ask باشد؛ در غیر این‌صورت last/bid/ask."""

        try:
            t = self.get_tick(symbol)
            if not t:
                return 0.0
            bid = float(t.get("bid", 0.0))
            ask = float(t.get("ask", 0.0))
            last = float(t.get("last", 0.0)) if "last" in t else 0.0
            if bid and ask:
                return (bid + ask) / 2.0
            return last or bid or ask or 0.0
        except Exception:
            return 0.0

    # اختیاری: برای سازگاری با کدهایی که شاید هنوز get_last_tick را صدا بزنند
    def get_last_tick(self, symbol: str = None):
        return self.get_tick(symbol)

    def get_tick_data(
        self,
        symbol: str = None,
        count: int = 1000,
        flags: int = None,
    ) -> Optional[pd.DataFrame]:
        """دریافت داده‌های تیک با لاک و ایندکس‌دهی زمانی"""

        symbol = symbol or self.config.SYMBOL
        flags = flags or mt5.COPY_TICKS_ALL

        with self.ensure_connected():
            with self._api_lock:
                ticks = mt5.copy_ticks_from_pos(symbol, 0, count, flags)
        if ticks is None or len(ticks) == 0:
            return None
        df = pd.DataFrame(ticks)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
        if "time_msc" in df.columns:
            df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms")
        return df

    def get_current_price(self, symbol: str = None) -> Optional[Dict[str, float]]:
        """دریافت قیمت فعلی"""

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            return {
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "spread": symbol_info.spread,
                "time": datetime.now(),
            }
        return None

    # --- Realtime tick for GUI/Dashboard ---

    def get_tick(self, symbol: str = None) -> Optional[Dict[str, Union[float, int, datetime]]]:
        """برگرداندن یک تیک زنده (با لاک + retry سبک)"""

        sym = symbol or self.config.SYMBOL
        last_err = None
        with self.ensure_connected():
            with self._api_lock:
                for _ in range(self._default_retries):
                    try:
                        t = mt5.symbol_info_tick(sym)
                        if t is not None:
                            # تبدیل زمان ثانیه‌ای به datetime
                            t_time = getattr(t, "time", None)
                            if isinstance(t_time, (int, float)):
                                t_time = datetime.fromtimestamp(t_time)
                            return {
                                "bid": float(getattr(t, "bid", 0.0)),
                                "ask": float(getattr(t, "ask", 0.0)),
                                "volume": float(getattr(t, "volume", 0.0)) if hasattr(t, "volume") else 0.0,
                                "time": t_time,
                            }
                    except Exception:
                        pass
                    last_err = mt5.last_error()
                    if last_err and isinstance(last_err, tuple) and "Call failed" in str(last_err[1]):
                        time.sleep(self._default_retry_sleep)
                        continue
                    break
        logger.debug(f"symbol_info_tick failed: {last_err}")
        return None

    def get_symbol_info_tick(self, symbol: str = None) -> Optional[Dict[str, Union[float, int, datetime]]]:
        """alias برای سازگاری با GUI"""

        return self.get_tick(symbol)

    def symbol_info_tick(self, symbol: str = None) -> Optional[Dict[str, Union[float, int, datetime]]]:
        """alias دیگر برای سازگاری"""

        return self.get_tick(symbol)

    # ------------------- Trading (order helpers) ---------------------

    def _success_retcode(self, rc: int) -> bool:
        """آیا retcode موفقیت‌آمیز است؟"""

        ok_codes = {
            getattr(mt5, "TRADE_RETCODE_DONE", None),
            getattr(mt5, "TRADE_RETCODE_PLACED", None),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
        }
        ok_codes.discard(None)
        return rc in ok_codes or rc == 0  # برخی محیط‌ها 0 را هم برمی‌گردانند

    def _unsupported_fill_retcode(self, rc: int) -> bool:
        """آیا خطا مربوط به عدم پشتیبانی Filling Mode است؟"""

        invalid_fill = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
        unsupported = 10030  # «Unsupported filling mode»
        return rc in (invalid_fill, unsupported)

    def _detect_filling_modes(self, symbol: str) -> List[int]:
        """
        تشخیص ترتیب اولویت Filling Mode بر اساس info.filling_mode
        به‌همراه فهرست فالبک کامل.
        """

        modes: List[int] = []
        with self.ensure_connected():
            with self._api_lock:
                info = mt5.symbol_info(symbol)
        try:
            fm = int(getattr(info, "filling_mode", -1)) if info else -1
            if fm in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                modes.append(fm)
        except Exception:
            pass

        # فهرست کامل فالبک
        for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            if m not in modes:
                modes.append(m)
        return modes

    def _order_send_with_filling_fallback(self, request: Dict, symbol: str):
        """
        سفارش را با امتحان کردن چند Filling Mode ارسال می‌کند.
        اگر 'type_filling' از قبل داخل request باشد، همان اول امتحان می‌شود.
        """

        candidates = self._detect_filling_modes(symbol)

        # اگر کاربر خودش type_filling داده، آن را در اولویت اول قرار بده
        preset = request.get("type_filling", None)
        if preset and preset in candidates:
            candidates.remove(preset)
            candidates.insert(0, preset)
        elif preset and preset not in candidates:
            candidates.insert(0, preset)

        # اطمینان از visible بودن
        with self.ensure_connected():
            with self._api_lock:
                info = mt5.symbol_info(symbol)
                if info and not info.visible:
                    mt5.symbol_select(symbol, True)

        last_result = None
        for idx, fill in enumerate(candidates, start=1):
            req = dict(request)
            req["type_filling"] = int(fill)
            with self.ensure_connected():
                with self._api_lock:
                    result = mt5.order_send(req)
            last_result = result

            if result is None:
                logger.error(f"order_send returned None (try {idx}/{len(candidates)} with fill={fill})")
                continue

            if self._success_retcode(result.retcode):
                logger.info(f"Order accepted with filling={fill} (retcode={result.retcode})")
                return result

            # اگر مود پشتیبانی نشد، بعدی را تست کن
            if self._unsupported_fill_retcode(result.retcode):
                logger.warning(
                    f"Unsupported filling mode ({fill}) for {symbol} (retcode={result.retcode}). Trying next..."
                )
                continue

            # خطای دیگری بود: همین را برگردان
            logger.error(f"Order failed (retcode={result.retcode}). comment={getattr(result, 'comment', '')}")
            return result

        # اگر همه رد شدند
        return last_result

    # ------------------------- Trading API --------------------------

    def open_position(
        self,
        symbol: str = None,
        order_type: str = "BUY",
        volume: float = 0.01,
        sl_points: Optional[int] = None,
        tp_points: Optional[int] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = "",
        magic: Optional[int] = None,
        deviation: int = 20,
    ) -> Optional[Dict]:
        """باز کردن پوزیشن جدید (با فالبک خودکار Filling Mode)"""

        symbol = symbol or self.config.SYMBOL
        magic = magic or self.config.MAGIC_NUMBER

        s_info = self.get_symbol_info(symbol)
        if s_info is None:
            return None

        price = s_info.ask if order_type.upper() == "BUY" else s_info.bid
        order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL

        sl = tp = 0.0
        if sl_price is not None:
            sl = float(sl_price)
        elif sl_points is not None:
            sl = price - sl_points * s_info.point if order_type.upper() == "BUY" else price + sl_points * s_info.point

        if tp_price is not None:
            tp = float(tp_price)
        elif tp_points is not None:
            tp = price + tp_points * s_info.point if order_type.upper() == "BUY" else price - tp_points * s_info.point

        volume = self._normalize_volume(volume, s_info)

        base_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_mt5,
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            # "type_filling" عمداً اینجا تنظیم نمی‌شود؛ فالبک خودش مدیریت می‌کند.
        }
        if sl > 0:
            base_request["sl"] = round(sl, s_info.digits)
        if tp > 0:
            base_request["tp"] = round(tp, s_info.digits)

        result = self._order_send_with_filling_fallback(base_request, symbol)
        if result is None:
            logger.error("Order send failed: No result returned")
            return None
        if not self._success_retcode(result.retcode):
            logger.error(f"Order failed: {getattr(result, 'comment', '')} (retcode: {result.retcode})")
            return None

        ticket = getattr(result, "order", None) or getattr(result, "deal", None)
        logger.success(f"Position opened: ticket={ticket}")
        return {
            "ticket": ticket,
            "symbol": symbol,
            "type": order_type.upper(),
            "volume": volume,
            "price": getattr(result, "price", float(price)),
            "sl": base_request.get("sl", 0.0),
            "tp": base_request.get("tp", 0.0),
            "comment": comment,
            "time": datetime.now(),
            "deal": getattr(result, "deal", None),
            "profit": 0.0,
        }

    def close_position(self, ticket: int, deviation: int = 20) -> bool:
        """بستن پوزیشن (با فالبک Filling Mode)"""

        with self.ensure_connected():
            with self._api_lock:
                pos = mt5.positions_get(ticket=ticket)
        if not pos:
            logger.error(f"Position {ticket} not found")
            return False
        pos = pos[0]
        s_info = self.get_symbol_info(pos.symbol)
        if s_info is None:
            return False

        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = s_info.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = s_info.ask

        base_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(pos.volume),
            "type": order_type,
            "position": int(ticket),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(pos.magic),
            "comment": f"Close {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = self._order_send_with_filling_fallback(base_request, pos.symbol)
        if result is None or not self._success_retcode(result.retcode):
            logger.error(
                f"Close failed: {getattr(result, 'comment', 'No result')} (retcode={getattr(result, 'retcode', None)})"
            )
            return False
        logger.info(f"Position {ticket} closed at {price}")
        return True

    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """تغییر SL/TP پوزیشن"""

        with self.ensure_connected():
            with self._api_lock:
                pos = mt5.positions_get(ticket=ticket)
        if not pos:
            logger.error(f"Position {ticket} not found")
            return False
        pos = pos[0]
        s_info = self.get_symbol_info(pos.symbol)
        if s_info is None:
            return False

        new_sl = sl if sl is not None else pos.sl
        new_tp = tp if tp is not None else pos.tp
        new_sl = round(new_sl, s_info.digits) if new_sl and new_sl > 0 else 0.0
        new_tp = round(new_tp, s_info.digits) if new_tp and new_tp > 0 else 0.0

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": int(ticket),
            "sl": new_sl,
            "tp": new_tp,
            "magic": int(pos.magic),
        }
        with self.ensure_connected():
            with self._api_lock:
                result = mt5.order_send(request)

        if result is None or not self._success_retcode(result.retcode):
            logger.error(
                f"Modify failed: {getattr(result, 'comment', 'No result')} (retcode={getattr(result, 'retcode', None)})"
            )
            return False
        logger.info(f"Position {ticket} modified: SL={new_sl}, TP={new_tp}")
        return True

    def close_partial(self, ticket: int, volume: float, deviation: int = 20) -> bool:
        """بستن بخشی از پوزیشن (با فالبک Filling Mode)"""

        with self.ensure_connected():
            with self._api_lock:
                pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]
        s_info = self.get_symbol_info(pos.symbol)
        if s_info is None:
            return False

        if volume >= pos.volume:
            return self.close_position(ticket, deviation)

        volume = self._normalize_volume(volume, s_info)
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = s_info.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = s_info.ask

        base_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(volume),
            "type": order_type,
            "position": int(ticket),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(pos.magic),
            "comment": f"Partial close {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = self._order_send_with_filling_fallback(base_request, pos.symbol)
        if result is None or not self._success_retcode(result.retcode):
            logger.error(
                f"Partial close failed: {getattr(result, 'comment', 'No result')} (retcode={getattr(result, 'retcode', None)})"
            )
            return False
        logger.info(f"Partial close {volume} lots of position {ticket}")
        return True

    # ------------------ Position / Order Management ------------------

    def get_positions(self, symbol: str = None, magic: int = None) -> List[Dict]:
        """دریافت پوزیشن‌های باز"""

        with self.ensure_connected():
            with self._api_lock:
                positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        result: List[Dict] = []
        for p in positions:
            if magic is not None and int(p.magic) != int(magic):
                continue
            s_info = self.get_symbol_info(p.symbol)
            if not s_info:
                continue
            current_price = s_info.bid if p.type == mt5.ORDER_TYPE_BUY else s_info.ask
            try:
                profit_points = int(
                    (current_price - p.price_open)
                    / s_info.point
                    * (1 if p.type == mt5.ORDER_TYPE_BUY else -1)
                )
            except Exception:
                profit_points = 0
            result.append(
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": float(p.volume),
                    "price_open": float(p.price_open),
                    "price_current": float(current_price),
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "profit": float(p.profit),
                    "swap": float(getattr(p, "swap", 0.0)),
                    "commission": float(getattr(p, "commission", 0.0)),
                    "magic": int(p.magic),
                    "comment": getattr(p, "comment", ""),
                    "time": datetime.fromtimestamp(p.time),
                    "profit_points": profit_points,
                }
            )
        return result

    def get_orders(self, symbol: str = None) -> List[Dict]:
        """دریافت سفارشات معلق"""

        with self.ensure_connected():
            with self._api_lock:
                orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if orders is None:
            return []
        result: List[Dict] = []
        for o in orders:
            result.append(
                {
                    "ticket": o.ticket,
                    "symbol": o.symbol,
                    "type": self._get_order_type_string(o.type),
                    "volume": float(o.volume),
                    "price": float(o.price_open),
                    "sl": float(o.sl),
                    "tp": float(o.tp),
                    "magic": int(o.magic),
                    "comment": getattr(o, "comment", ""),
                    "time_setup": datetime.fromtimestamp(o.time_setup),
                }
            )
        return result

    def delete_order(self, ticket: int) -> bool:
        """حذف سفارش معلق"""

        with self.ensure_connected():
            with self._api_lock:
                order = mt5.orders_get(ticket=ticket)
                if not order:
                    return False
                result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)})
        return result is not None and self._success_retcode(result.retcode)

    # -------------------- Account / History --------------------------

    def get_account_info(self) -> Dict:
        """دریافت اطلاعات حساب"""

        with self.ensure_connected():
            with self._api_lock:
                a = mt5.account_info()
        if a is None:
            return {}
        return {
            "login": a.login,
            "server": a.server,
            "currency": a.currency,
            "leverage": a.leverage,
            "balance": float(a.balance),
            "equity": float(a.equity),
            "margin": float(a.margin),
            "margin_free": float(a.margin_free),
            "margin_level": float(a.margin_level),
            "profit": float(a.profit),
            "credit": float(a.credit),
            "name": a.name,
            "company": a.company,
            "trade_allowed": bool(a.trade_allowed),
            "limit_orders": int(a.limit_orders),
            "margin_so_mode": int(a.margin_so_mode),
            "margin_so_call": float(a.margin_so_call),
            "margin_so_so": float(a.margin_so_so),
        }

    def get_daily_profit(self) -> Tuple[float, int, float]:
        """محاسبه سود/زیان روزانه"""

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        with self.ensure_connected():
            with self._api_lock:
                deals = mt5.history_deals_get(today, tomorrow)
        if deals is None:
            return 0.0, 0, 0.0

        total_profit = 0.0
        total_commission = 0.0
        trade_count = 0
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                total_profit += float(d.profit + d.swap)
                total_commission += float(d.commission)
                trade_count += 1
        net_profit = total_profit + total_commission
        return net_profit, trade_count, total_commission

    def get_deals_history(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = None,
    ) -> List[Dict]:
        """دریافت تاریخچه معاملات"""

        with self.ensure_connected():
            with self._api_lock:
                deals = mt5.history_deals_get(start_date, end_date)
        if deals is None:
            return []
        out: List[Dict] = []
        for d in deals:
            if symbol and d.symbol != symbol:
                continue
            out.append(
                {
                    "ticket": d.ticket,
                    "order": d.order,
                    "symbol": d.symbol,
                    "type": "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                    "volume": float(d.volume),
                    "price": float(d.price),
                    "profit": float(d.profit),
                    "swap": float(d.swap),
                    "commission": float(d.commission),
                    "magic": int(d.magic),
                    "comment": getattr(d, "comment", ""),
                    "time": datetime.fromtimestamp(d.time),
                    "entry": "IN" if d.entry == mt5.DEAL_ENTRY_IN else "OUT",
                }
            )
        return out

    # --------------------------- Utility -----------------------------

    def calculate_lot_size(self, risk_amount: float, sl_points: int, symbol: str = None) -> float:
        """محاسبه حجم لات بر اساس ریسک"""

        symbol = symbol or self.config.SYMBOL
        s_info = self.get_symbol_info(symbol)
        if not s_info or not sl_points or sl_points <= 0:
            return float(getattr(self.config, "MIN_POSITION_SIZE", 0.01))

        # Tick value تقریبی با تکیه بر contract size
        tcs = float(getattr(s_info, "trade_contract_size", 0.0))
        tick_value = (tcs * s_info.point) if tcs > 0 else s_info.point
        lot = (
            risk_amount / (sl_points * tick_value)
            if tick_value > 0
            else float(getattr(self.config, "MIN_POSITION_SIZE", 0.01))
        )
        return self._normalize_volume(lot, s_info)

    def _calculate_tick_value(self, symbol: str) -> float:
        """محاسبه ارزش هر تیک (ساده/تقریبی)"""

        s_info = self.get_symbol_info(symbol)
        if not s_info:
            return 0.0
        if "USD" in symbol.upper():
            tick_value = s_info.trade_contract_size * s_info.point
        else:
            # تقریبی برای غیر-USD: بر اساس قیمت ask فعلی
            ask = s_info.ask if s_info.ask > 0 else (s_info.bid if s_info.bid > 0 else 1.0)
            tick_value = s_info.trade_contract_size * s_info.point / max(ask, 1e-8)
        return float(tick_value)

    def _normalize_volume(self, volume: float, symbol_info: SymbolInfo) -> float:
        """نرمال‌سازی حجم بر اساس محدودیت‌های نماد"""

        v = float(volume)
        step = max(float(symbol_info.volume_step), 0.0) or 0.01
        v = round(v / step) * step
        v = max(symbol_info.volume_min, v)
        v = min(symbol_info.volume_max, v)
        v = max(float(getattr(self.config, "MIN_POSITION_SIZE", 0.01)), v)
        v = min(float(getattr(self.config, "MAX_POSITION_SIZE", 1.0)), v)
        # گرد کردن متناسب با step (برای دقت ارسال سفارش)
        decimals = str(step)[::-1].find('.') if '.' in str(step) else 0
        return round(v, max(decimals, 2))

    def _get_filling_mode(self, symbol: str) -> int:
        """تعیین حالت filling مناسب (برای سازگاری قدیمی — اکنون از فالبک استفاده می‌کنیم)"""

        with self.ensure_connected():
            with self._api_lock:
                info = mt5.symbol_info(symbol)
        try:
            if info and info.filling_mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                return int(info.filling_mode)
        except Exception:
            pass
        return mt5.ORDER_FILLING_IOC

    def _get_order_type_string(self, order_type: int) -> str:
        """تبدیل نوع سفارش به رشته"""

        order_types = {
            mt5.ORDER_TYPE_BUY: "BUY",
            mt5.ORDER_TYPE_SELL: "SELL",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY_STOP_LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL_STOP_LIMIT",
        }
        return order_types.get(order_type, "UNKNOWN")

    def get_last_error(self) -> Optional[str]:
        """دریافت آخرین خطا"""

        return self._last_error

    # ------------------ Convenience / Preload (جدید) ------------------

    def preload_history(self, symbol: str = None, timeframes: Optional[List[str]] = None, days: int = 30) -> dict:
        """
        تاریخچه چند تایم‌فریم را پیش‌لود می‌کند تا MT5 کش شود و SignalEngine داده کافی داشته باشد.
        خروجی: { "M5": bars_loaded, "H1": bars_loaded, ... }
        """

        symbol = symbol or self.config.SYMBOL
        tfs = timeframes or ["M5", "M15", "H1", "H4"]
        loaded: Dict[str, int] = {}

        # اطمینان از visible بودن نماد
        with self.ensure_connected():
            with self._api_lock:
                info = mt5.symbol_info(symbol)
                if info and not info.visible:
                    mt5.symbol_select(symbol, True)

        bars_per_day = {
            "M1": 1440,
            "M5": 288,
            "M15": 96,
            "M30": 48,
            "H1": 24,
            "H4": 6,
            "D1": 1,
            "W1": 1,
            "MN1": 1,
        }

        for tf in tfs:
            bpd = bars_per_day.get(str(tf).upper(), 24)
            count = min(int(bpd * days * 1.2) + 50, 20000)  # حاشیه امن + سقف
            df = self.get_candles(symbol=symbol, timeframe=tf, count=count)
            loaded[tf] = 0 if df is None else len(df)

        logger.info(f"History preloaded for {symbol}: " + ", ".join(f"{k}={v}" for k, v in loaded.items()))
        return loaded

    def get_spread(self, symbol: str = None, in_points: bool = True) -> Optional[float]:
        """
        دریافت اسپرد فعلی نماد.
        اگر in_points=True باشد، مقدار به «پوینت» برمی‌گردد (int در MT5).
        اگر False باشد، اسپرد قیمتی (points * point) برمی‌گردد.
        """

        info = self.get_symbol_info(symbol)
        if not info:
            return None
        return float(info.spread) if in_points else float(info.spread * info.point)

    def get_point(self, symbol: str = None) -> Optional[float]:
        """دریافت اندازهٔ point نماد (برای محاسبات SL/TP و حجم)."""

        info = self.get_symbol_info(symbol)
        return float(info.point) if info else None

    def calculate_tick_value(self, symbol: str = None) -> float:
        """نسخهٔ پابلیک متد داخلیِ محاسبه ارزش هر تیک."""

        return self._calculate_tick_value(symbol or self.config.SYMBOL)

    # ------------------ 🔗 GUI Compatibility Aliases ------------------

    def get_rates(self, symbol: str = None, timeframe: str = "H1", count: int = 300) -> List[Dict]:
        """
        سازگار با GUI: برمی‌گرداند لیستی از دیکشنری‌های
        {time, open, high, low, close}
        """

        symbol = symbol or self.config.SYMBOL
        df = self.get_candles(symbol=symbol, timeframe=timeframe, count=count)
        if df is None or df.empty:
            return []
        out: List[Dict] = []
        for ts, row in df.tail(count).iterrows():
            try:
                out.append(
                    {
                        "time": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                )
            except Exception:
                continue
        return out

    def get_ohlcv(self, symbol: str = None, timeframe: str = "H1", count: int = 300) -> List[Dict]:
        """alias برای get_rates جهت سازگاری با GUI"""

        return self.get_rates(symbol=symbol, timeframe=timeframe, count=count)

    def copy_rates_from_pos(
        self, symbol: str = None, timeframe: str = "H1", start_pos: int = 0, count: int = 300
    ) -> List[Dict]:
        """
        alias سازگار با GUI: پشت‌صحنه از MT5 استفاده می‌کند اما
        خروجی را به لیست دیکشنریِ time/open/high/low/close تبدیل می‌کند.
        """

        symbol = symbol or self.config.SYMBOL
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        tf = tf_map.get(str(timeframe).upper(), mt5.TIMEFRAME_H1)
        with self.ensure_connected():
            with self._api_lock:
                rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
        if rates is None or len(rates) == 0:
            return []
        try:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            out = []
            for _, r in df.iterrows():
                out.append(
                    {
                        "time": r["time"].to_pydatetime(),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                    }
                )
            return out
        except Exception:
            # فالبک: خروجی خام را به شکل قابل‌مصرف برگردانیم
            out = []
            for r in rates:
                try:
                    t = getattr(r, "time", None)
                    if isinstance(t, (int, float)):
                        t = datetime.fromtimestamp(t)
                    out.append(
                        {
                            "time": t,
                            "open": float(getattr(r, "open", 0.0)),
                            "high": float(getattr(r, "high", 0.0)),
                            "low": float(getattr(r, "low", 0.0)),
                            "close": float(getattr(r, "close", 0.0)),
                        }
                    )
                except Exception:
                    continue
            return out

    def get_price_history(self, symbol: str = None, timeframe: str = "H1", count: int = 300) -> List[Dict]:
        """alias دیگر برای GUI"""

        return self.get_rates(symbol=symbol, timeframe=timeframe, count=count)

    def get_info(self, symbol: str = None) -> Optional[SymbolInfo]:
        """alias برای get_symbol_info (برای سازگاری با GUI قدیمی)"""

        return self.get_symbol_info(symbol)

