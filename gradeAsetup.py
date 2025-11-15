import yfinance as yf
import requests
from datetime import datetime
import pandas as pd
import config
import os
import time
import numpy as np

# ==============================================================  
# TELEGRAM CONFIG  
# ==============================================================  
TELEGRAM_BOT_TOKEN = config.BOT_TOKEN
TELEGRAM_CHAT_ID = config.CHAT_ID

LOG_FILE = "ath_alert_log.csv"

# User-defined parameters
THRESHOLD_PCT = 2.0               # Percent below ATH
MIN_CANDLES_SINCE_ATH = 10        # Minimum candles since ATH

# ==============================================================
# STRATEGY A (Breakout + Retest + MA + RSI + Volume)
# ==============================================================

SHORT_EMA = 9
MID_EMA = 21
LONG_SMA = 200
RSI_PERIOD = 14
MIN_VOLUME_FACTOR = 1.5
LOCAL_LOOKBACK = 60
RSI_MIN = 50
RSI_MAX = 80
RETEST_ZONE_PCT = 3.0


# ------------------- Helper Functions -------------------------

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


def check_breakout_retest_and_indicators(data):
    out = {}
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data.get("Volume", None)

    # ---- MA Calculation ----
    data["ema_short"] = compute_ema(close, SHORT_EMA)
    data["ema_mid"] = compute_ema(close, MID_EMA)
    data["sma_long"] = compute_sma(close, LONG_SMA)

    last_idx = -1
    last_close = float(close.iloc[last_idx])

    # Trend confirmation
    ma_ok = (
        data["ema_short"].iloc[last_idx] > data["ema_mid"].iloc[last_idx] and
        data["ema_mid"].iloc[last_idx] > data["sma_long"].iloc[last_idx]
    )

    # ---- RSI ----
    rsi = compute_rsi(close)
    rsi_last = float(rsi.iloc[last_idx])
    rsi_ok = RSI_MIN <= rsi_last <= RSI_MAX

    # ---- Local Breakout ----
    lookback = min(len(data), LOCAL_LOOKBACK)
    local_resistance = float(high.iloc[-lookback:-1].max())
    local_breakout = last_close > local_resistance

    # ---- Retest Detection ----
    retest = False
    retest_distance_pct = None

    if local_breakout:
        lows = low.iloc[-lookback:]
        min_low_after_breakout = float(lows.min())
        retest_distance_pct = ((local_resistance - min_low_after_breakout) / local_resistance) * 100
        retest = retest_distance_pct <= RETEST_ZONE_PCT

    # ---- Volume Confirmation ----
    volume_ok = True
    avg_vol = None
    if volume is not None and len(volume) >= 20:
        avg_vol = float(volume.iloc[-21:-1].mean())
        last_vol = float(volume.iloc[last_idx])
        volume_ok = last_vol >= MIN_VOLUME_FACTOR * avg_vol

    out.update({
        "ma_ok": ma_ok,
        "rsi_ok": rsi_ok,
        "volume_ok": volume_ok,
        "local_breakout": local_breakout,
        "retest": retest,
        "local_resistance": local_resistance,
        "rsi": rsi_last,
        "retest_distance_pct": retest_distance_pct,
    })
    return out


# ==============================================================  
# TELEGRAM  
# ==============================================================  
def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=15)
        return response.status_code == 200
    except:
        return False


def append_to_csv(file_path, data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)


# ==============================================================  
# MAIN LOGIC  
# ==============================================================  

def check_all_time_high_once(symbol: str, threshold_pct=THRESHOLD_PCT, min_candles_since_ath=MIN_CANDLES_SINCE_ATH):
    try:
        data = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            return symbol, False, None

        current_price = float(data["Close"].iloc[-1])
        all_time_high = float(data["High"].max())

        # ATH %
        diff_percent = 0 if current_price >= all_time_high else ((all_time_high - current_price) / all_time_high) * 100

        # Candles since last ATH
        ath_positions = data.index[data["High"] == all_time_high]
        if len(ath_positions) == 0:
            candles_since_ath = None
        else:
            last_ath = ath_positions[-1]
            candles_since_ath = len(data) - 1 - data.index.get_loc(last_ath)

        # ATH conditions
        cond_below_ath = current_price < all_time_high
        cond_within_pct = diff_percent <= threshold_pct
        cond_old_ath = candles_since_ath is not None and candles_since_ath > min_candles_since_ath

        ath_ok = cond_below_ath and cond_within_pct and cond_old_ath

        # Strategy A conditions
        ta = check_breakout_retest_and_indicators(data)

        strategy_ok = (
            (ta["local_breakout"] and ta["ma_ok"] and ta["rsi_ok"] and ta["volume_ok"])
            or
            (ta["retest"] and ta["ma_ok"] and ta["rsi_ok"])
        )

        # Final Decision
        should_alert = ath_ok and strategy_ok

        alert_sent = False
        if should_alert:
            msg = (
                f"🚨 {symbol} — Breakout Setup Near ATH\n"
                f"Price: {current_price}\nATH: {all_time_high}\n"
                f"Diff: {diff_percent:.2f}%\nCandles Since ATH: {candles_since_ath}\n\n"
                f"📊 Strategy Checks:\n"
                f"• MA Alignment: {ta['ma_ok']}\n"
                f"• RSI: {ta['rsi']:.2f}\n"
                f"• Local Breakout: {ta['local_breakout']}\n"
                f"• Retest: {ta['retest']}\n"
                f"• Volume OK: {ta['volume_ok']}\n"
            )
            alert_sent = send_telegram_alert(msg)

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "current_price": current_price,
            "ath": all_time_high,
            "diff_percent": diff_percent,
            "candles_since_ath": candles_since_ath,
            "alert_sent": alert_sent
        }
        append_to_csv(LOG_FILE, log_data)

        return symbol, alert_sent, log_data

    except Exception as e:
        print(f"Error: {e}")
        return symbol, False, None


# ==============================================================  
# MAIN  
# ==============================================================  
if __name__ == "__main__":
    stock_list = config.NIFTY50_STOCKS
    alerted_symbols = []

    print(f"\nChecking All-Time Highs — {datetime.now()}\n")

    for i, stock in enumerate(stock_list, 1):
        print(f"[{i}/{len(stock_list)}] Scanning {stock}...")
        symbol, alert_sent, _ = check_all_time_high_once(stock)
        if alert_sent:
            alerted_symbols.append(stock)
        time.sleep(1)

    summary = (
        f"Summary at {datetime.now()}\n"
        f"Checked: {len(stock_list)}\nAlerts: {len(alerted_symbols)}\n"
        f"{', '.join(alerted_symbols)}"
    )
    send_telegram_alert(summary)

