import yfinance as yf
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import config

# ==============================================================
# TELEGRAM CONFIG
# ==============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==============================================================
# PARAMETERS
# ==============================================================
THRESHOLD_PCT = 8.0            # % below ATH
MIN_CANDLES_SINCE_ATH = 4

# Strategy A (balanced version)
SHORT_EMA = 9
MID_EMA = 21
LONG_SMA = 200
RSI_PERIOD = 14
MIN_VOLUME_FACTOR = 1.15
LOCAL_LOOKBACK = 40
RETEST_ZONE_PCT = 8.0
DEFAULT_RR = 2.0  # target2 = 2R


# ==============================================================
# INDICATOR HELPERS
# ==============================================================
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_sma(series, window):
    return series.rolling(window, min_periods=1).mean()

def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


# ==============================================================
# TELEGRAM
# ==============================================================
def send_telegram_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except:
        return False


# ==============================================================
# TA CHECKS
# ==============================================================
def check_breakout_retest_and_indicators(data):
    out = {
        "ma_ok": False, "rsi_ok": False, "volume_ok": False,
        "local_breakout": False, "retest": False,
        "local_resistance": None, "recent_swing_low": None,
        "rsi": None, "retest_distance_pct": None
    }

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    vol = data.get("Volume", None)

    # MAs
    data["ema_short"] = compute_ema(close, SHORT_EMA)
    data["ema_mid"] = compute_ema(close, MID_EMA)
    data["sma_long"] = compute_sma(close, LONG_SMA)

    last = -1
    last_close = float(close.iloc[last])

    # MA alignment
    out["ma_ok"] = (
        data["ema_short"].iloc[last] > data["ema_mid"].iloc[last]
        and data["ema_mid"].iloc[last] > data["sma_long"].iloc[last]
    )

    # RSI
    rsi = compute_rsi(close)
    out["rsi"] = float(rsi.iloc[last])
    out["rsi_ok"] = 50 <= out["rsi"] <= 80

    # Local resistance (highest high of last N days)
    lb = min(len(data), LOCAL_LOOKBACK)
    out["local_resistance"] = float(high.iloc[-lb:-1].max())

    out["local_breakout"] = last_close > out["local_resistance"]

    # Recent swing low (SL)
    out["recent_swing_low"] = float(low.iloc[-lb:-1].min())

    # Retest detection
    min_low = float(low.iloc[-lb:].min())
    out["retest_distance_pct"] = (
        (out["local_resistance"] - min_low) / out["local_resistance"] * 100
        if out["local_resistance"] else None
    )
    out["retest"] = out["retest_distance_pct"] is not None and out["retest_distance_pct"] <= RETEST_ZONE_PCT

    # Volume check
    if vol is not None and len(vol) >= 21:
        avg_vol = float(vol.iloc[-21:-1].mean())
        last_vol = float(vol.iloc[last])
        out["volume_ok"] = last_vol >= MIN_VOLUME_FACTOR * avg_vol

    return out


# ==============================================================
# ENTRY/SL/TARGET CALC
# ==============================================================
def compute_entry_sl_targets(ta):
    entry = float(ta["local_resistance"])
    sl = float(ta["recent_swing_low"])

    if sl >= entry:
        sl = entry * 0.99

    risk = entry - sl
    tgt1 = entry + risk
    tgt2 = entry + DEFAULT_RR * risk
    return entry, sl, tgt1, tgt2


# ==============================================================
# MAIN SYMBOL CHECK
# ==============================================================
def check_all_time_high_once(symbol):
    try:
        data = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)

        if data is None or data.empty:
            print(f"⚠️ No data for {symbol}")
            return symbol, False

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        current = float(data["Close"].iloc[-1])
        ath = float(data["High"].max())
        diff = 0 if current >= ath else ((ath - current) / ath * 100)

        # ATH age
        ath_pos = data.index[data["High"] == ath]
        if len(ath_pos) == 0:
            return symbol, False

        last_ath = ath_pos[-1]
        candles_since_ath = len(data) - 1 - data.index.get_loc(last_ath)

        ath_ok = (
            current < ath
            and diff <= THRESHOLD_PCT
            and candles_since_ath > MIN_CANDLES_SINCE_ATH
        )

        # TA Checks
        ta = check_breakout_retest_and_indicators(data)

        # FIXED STRATEGY LOGIC
        breakout_condition = (
            ta["local_breakout"]
            and ta["ma_ok"]
            and ta["rsi_ok"]
            and ta["volume_ok"]
        )

        retest_condition = (
            ta["retest"]
            and ta["ma_ok"]
            and ta["rsi_ok"]
            and ta["volume_ok"]
        )

        strategy_ok = breakout_condition or retest_condition

        # Print debug
        print(
            f"{symbol} | ATH diff:{diff:.2f}% | MA:{ta['ma_ok']} | RSI:{ta['rsi_ok']} "
            f"| Vol:{ta['volume_ok']} | B/O:{ta['local_breakout']} | Retest:{ta['retest']}"
        )

        should_alert = ath_ok and strategy_ok

        if should_alert:
            entry, sl, tgt1, tgt2 = compute_entry_sl_targets(ta)
            msg = (
                f"🚨 A-GRADE SETUP FOUND — {symbol}\n\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"T1: {tgt1:.2f}\n"
                f"T2 (2R): {tgt2:.2f}\n\n"
                f"MA Trend: {ta['ma_ok']}\nRSI: {ta['rsi']:.2f}\nVolume OK: {ta['volume_ok']}\n"
                f"Breakout: {ta['local_breakout']} | Retest: {ta['retest']}\n"
            )
            send_telegram_alert(msg)
            return symbol, True

        return symbol, False

    except Exception as e:
        print(f"⚠️ Error {symbol}: {e}")
        return symbol, False


# ==============================================================
# PARALLEL SCAN
# ==============================================================
def run_parallel_scan(stock_list):
    workers = max(1, cpu_count() - 1)
    print(f"Starting parallel scan on {workers} workers...")
    with Pool(workers) as pool:
        return pool.map(check_all_time_high_once, stock_list)


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    stock_list = config.NIFTY50_STOCKS

    print(f"\n📈 Starting scan — {datetime.now()}\n")

    results = run_parallel_scan(stock_list)
    alerted = [s for s, ok in results if ok]

    summary = (
        f"Summary ({datetime.now()}):\n"
        f"Scanned: {len(stock_list)}\n"
        f"Alerts: {len(alerted)}\n"
        f"Symbols: {', '.join(alerted) if alerted else 'None'}"
    )

    send_telegram_alert(summary)
    print(summary)
