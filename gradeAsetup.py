import yfinance as yf
import requests
from datetime import datetime
import pandas as pd
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count
import config

# ==============================================================  
# TELEGRAM CONFIG  
# ==============================================================  
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LOG_FILE = "ath_alert_log.csv"

# User-defined parameters
THRESHOLD_PCT = 5.0               # Percent below ATH
MIN_CANDLES_SINCE_ATH = 10        # Minimum candles since ATH

# ==============================================================
# STRATEGY A (Breakout + Retest + MA + RSI + Volume)
# ==============================================================
SHORT_EMA = 9
MID_EMA = 21
LONG_SMA = 200
RSI_PERIOD = 14
MIN_VOLUME_FACTOR = 1.5
LOCAL_LOOKBACK = 60               # days to find local resistance / swing low
RSI_MIN = 50
RSI_MAX = 80
RETEST_ZONE_PCT = 3.0             # distance from local resistance to qualify as retest zone
DEFAULT_RR = 2.0                  # target multiplier (2 = target2 is 2R)

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

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=15)
        if response.status_code == 200:
            print("✅ Telegram alert sent.")
            return True
        else:
            print(f"❌ Telegram failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"⚠️ Telegram send error: {e}")
        return False

def append_to_csv(file_path, data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)

# ------------------- Core TA checks --------------------------
def check_breakout_retest_and_indicators(data):
    """
    Returns dictionary:
     - ma_ok, rsi_ok, volume_ok, local_breakout, retest,
     - local_resistance, recent_swing_low, rsi, avg_vol, last_vol
    """
    out = {}
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    vol = data.get("Volume", None)

    # MAs
    data["ema_short"] = compute_ema(close, SHORT_EMA)
    data["ema_mid"] = compute_ema(close, MID_EMA)
    data["sma_long"] = compute_sma(close, LONG_SMA)

    last_idx = -1
    last_close = float(close.iloc[last_idx])

    # MA alignment: short > mid > long
    ma_ok = (
        data["ema_short"].iloc[last_idx] > data["ema_mid"].iloc[last_idx]
        and data["ema_mid"].iloc[last_idx] > data["sma_long"].iloc[last_idx]
    )

    # RSI
    rsi = compute_rsi(close)
    rsi_last = float(rsi.iloc[last_idx])
    rsi_ok = RSI_MIN <= rsi_last <= RSI_MAX

    # Local resistance (highest high in lookback excluding last candle)
    lookback = min(len(data), LOCAL_LOOKBACK)
    if lookback > 1:
        local_resistance = float(high.iloc[-lookback:-1].max())
    else:
        local_resistance = float(high.max())

    local_breakout = last_close > local_resistance

    # Recent swing low (used as SL): min low in the lookback window BEFORE the last candle
    # If not enough history, fallback to min of available lows
    if lookback > 1:
        recent_swing_low = float(low.iloc[-lookback:-1].min())
    else:
        recent_swing_low = float(low.min())

    # Retest detection:
    # Find whether price has pulled back into the retest zone (within RETEST_ZONE_PCT of local_resistance)
    # We'll compute the min low in the period since the highest local_resistance was formed
    retest = False
    retest_distance_pct = None
    try:
        # min low in the recent lookback (including last candle) relative to local_resistance
        min_low = float(low.iloc[-lookback:].min())
        retest_distance_pct = ((local_resistance - min_low) / local_resistance) * 100 if local_resistance != 0 else None
        if retest_distance_pct is not None and retest_distance_pct <= RETEST_ZONE_PCT:
            retest = True
    except Exception:
        retest = False

    # Volume check: last candle volume >= MIN_VOLUME_FACTOR * avg prev volume
    volume_ok = True
    avg_vol = None
    last_vol = None
    if vol is not None and len(vol) >= 21:
        avg_vol = float(vol.iloc[-21:-1].mean())
        last_vol = float(vol.iloc[last_idx])
        volume_ok = last_vol >= MIN_VOLUME_FACTOR * avg_vol

    out.update({
        "ma_ok": ma_ok,
        "rsi_ok": rsi_ok,
        "volume_ok": volume_ok,
        "local_breakout": local_breakout,
        "retest": retest,
        "local_resistance": local_resistance,
        "recent_swing_low": recent_swing_low,
        "rsi": rsi_last,
        "avg_vol": avg_vol,
        "last_vol": last_vol,
        "retest_distance_pct": retest_distance_pct
    })
    return out

# ------------------- Entry/SL/Targets logic -------------------
def compute_entry_sl_targets(ta_checks):
    """
    Using user's choices:
     - Entry: Local resistance (retest entry)
     - SL: Recent swing low
     - Targets: 1R and DEFAULT_RR (2R)
    Returns (entry, sl, tgt1, tgt2, rr_calc)
    """
    entry = float(ta_checks["local_resistance"])
    sl = float(ta_checks["recent_swing_low"])

    # ensure sensible SL < entry; if not, fallback to small buffer below entry
    if not (sl < entry):
        sl = entry * 0.99  # 1% buffer below entry as fallback

    risk = entry - sl
    if risk <= 0:
        # degenerate case: set targets based on percentage if risk invalid
        risk = entry * 0.01
        sl = entry - risk

    tgt1 = entry + 1.0 * risk
    tgt2 = entry + DEFAULT_RR * risk
    rr_calc = (tgt2 - entry) / (entry - sl) if (entry - sl) != 0 else None

    return entry, sl, tgt1, tgt2, rr_calc

# ==============================================================  
# MAIN SYMBOL CHECK
# ==============================================================  
def check_all_time_high_once(symbol: str):
    """Fetch historical data once, compute ATH proximity and Strategy-A checks,
       and send an 'A-GRADE' alert if both ATH proximity and strategy_ok are satisfied.
    """
    try:
        data = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            print(f"⚠️ No data for {symbol}")
            return symbol, False, None

        # normalize columns if multiindex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        current_price = float(data["Close"].iloc[-1])
        all_time_high = float(data["High"].max())

        # percent below ATH (0 if at/above)
        diff_percent = 0.0 if current_price >= all_time_high else ((all_time_high - current_price) / all_time_high) * 100

        # candles since last ATH
        ath_positions = data.index[data["High"] == all_time_high]
        if len(ath_positions) == 0:
            candles_since_ath = None
        else:
            last_ath_index = ath_positions[-1]
            last_ath_pos = data.index.get_loc(last_ath_index)
            candles_since_ath = (len(data) - 1) - last_ath_pos

        print(f"{symbol} | Current: {current_price:.2f} | ATH: {all_time_high:.2f} | Diff: {diff_percent:.2f}% | Candles since ATH: {candles_since_ath}")

        # ATH conditions
        cond_price_below_ath = current_price < all_time_high
        cond_within_pct = diff_percent <= THRESHOLD_PCT
        cond_candles = (candles_since_ath is not None) and (candles_since_ath > MIN_CANDLES_SINCE_ATH)
        ath_ok = cond_price_below_ath and cond_within_pct and cond_candles

        # Strategy A TA checks (local breakout/retest, MA, RSI, volume)
        ta = check_breakout_retest_and_indicators(data)

        # Strategy OK: either a local_breakout with confirmations OR a retest with confirmations
        strategy_ok = (
            (ta["local_breakout"] and ta["ma_ok"] and ta["rsi_ok"] and ta["volume_ok"])
            or
            (ta["retest"] and ta["ma_ok"] and ta["rsi_ok"])
        )

        should_alert = ath_ok and strategy_ok
        alert_sent = False
        extra_info = {}

        if should_alert:
            # compute entry/sl/targets per your chosen defaults
            entry, sl, tgt1, tgt2, rr_calc = compute_entry_sl_targets(ta)
            rr_display = f"{rr_calc:.2f}R" if rr_calc is not None else "N/A"

            message = (
                f"🚨 *A-GRADE SETUP FOUND* — {symbol}\n\n"
                f"Setup: Breakout/Retest near ATH\n"
                f"Entry (retest/local resistance): {entry:.2f}\n"
                f"SL (recent swing low): {sl:.2f}\n"
                f"Target1 (1R): {tgt1:.2f}\n"
                f"Target2 (2R): {tgt2:.2f}\n"
                f"Calculated RR (T2): {rr_display}\n\n"
                f"Technical checks:\n"
                f"• MA Alignment (9>21>200): {ta['ma_ok']}\n"
                f"• RSI: {ta['rsi']:.2f}\n"
                f"• Volume OK: {ta['volume_ok']}\n"
                f"• Local Breakout: {ta['local_breakout']}\n"
                f"• Retest: {ta['retest']} (dist%: {ta['retest_distance_pct']})\n\n"
                f"Price: {current_price:.2f} | ATH: {all_time_high:.2f} | Diff: {diff_percent:.2f}%\n"
                f"Candles since ATH: {candles_since_ath}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            # Telegram accepts plain text; if your bot supports Markdown, you can set parse_mode
            alert_sent = send_telegram_alert(message)
            extra_info.update({
                "entry": entry, "sl": sl, "tgt1": tgt1, "tgt2": tgt2, "rr_calc": rr_calc
            })

        # log everything
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "current_price": current_price,
            "ath": all_time_high,
            "diff_percent": diff_percent,
            "candles_since_ath": candles_since_ath,
            "ath_ok": ath_ok,
            "strategy_ok": strategy_ok,
            "ma_ok": ta["ma_ok"],
            "rsi": ta["rsi"],
            "volume_ok": ta["volume_ok"],
            "local_resistance": ta["local_resistance"],
            "recent_swing_low": ta["recent_swing_low"],
            "retest": ta["retest"],
            "retest_distance_pct": ta["retest_distance_pct"],
            "alert_sent": alert_sent
        }
        # include entry/sl/targets if present
        if extra_info:
            log_data.update(extra_info)

        append_to_csv(LOG_FILE, log_data)
        return symbol, alert_sent, log_data

    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")
        return symbol, False, None

# ==============================================================  
# DRIVER (parallel scan)
# ==============================================================  
def run_parallel_scan(stock_list):
    cpu = max(1, cpu_count() - 1)  # leave one core free
    print(f"Starting parallel scan on {cpu} workers...")
    with Pool(cpu) as pool:
        results = pool.map(check_all_time_high_once, stock_list)
    return results

if __name__ == "__main__":
    stock_list = config.NIFTY50_STOCKS  # e.g., ["RELIANCE.NS", ...]
    start = datetime.now()
    print(f"\n📈 Starting scan — {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = run_parallel_scan(stock_list)

    alerted = [r[0] for r in results if r and r[1]]
    total_checked = len([r for r in results if r])
    summary_msg = (
        f"✅ ATH Alert Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        f"Total Stocks Checked: {total_checked}\n"
        f"Alerts Sent: {len(alerted)}\n"
        f"Stocks Alerted: {', '.join(alerted) if alerted else 'None'}"
    )
    send_telegram_alert(summary_msg)
    end = datetime.now()
    print(f"\nDone. Time elapsed: {end - start}\nLogs saved to: {LOG_FILE}")



