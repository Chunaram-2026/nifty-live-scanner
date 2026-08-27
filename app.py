import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal 9-15 EMA Scalping Scanner",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1420;
    color: #d8dee9;
}

.main {
    background-color: #0e1420;
}

h1, h2, h3 {
    color: #d8dee9;
}

.metric-box {
    background: #1b2b3e;
    border: 1px solid #263b52;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 13px;
    color: #aeb8c5;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
    margin-top: 8px;
}

.signal-buy {
    color: #38d996;
    font-weight: bold;
}

.signal-sell {
    color: #ff6464;
    font-weight: bold;
}

.signal-wait {
    color: #ffd166;
    font-weight: bold;
}

.info-box {
    background: #1d344b;
    border-radius: 6px;
    padding: 14px;
    color: #b7d8f5;
    margin-bottom: 15px;
}

.rule-box {
    background: #151d29;
    border: 1px solid #293545;
    border-radius: 8px;
    padding: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SETTINGS
# ============================================================

INDEX_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS"
}

TIMEFRAME_MAP = {
    "1m": ("1m", "5d"),
    "2m": ("2m", "5d"),
    "5m": ("5m", "60d"),
    "15m": ("15m", "60d")
}


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def load_data(symbol, interval, period):

    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.copy()

        required = ["Open", "High", "Low", "Close"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 1

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA 9
    df["EMA9"] = (
        df["Close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # EMA 15
    df["EMA15"] = (
        df["Close"]
        .ewm(span=15, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # VWAP
    # Typical price requested by strategy:
    # (Open + High + Low + Close) / 4
    # --------------------------------------------------------

    df["TP"] = (
        df["Open"] +
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 4

    # Make sure volume is valid
    df["Volume"] = (
        pd.to_numeric(df["Volume"], errors="coerce")
        .fillna(0)
    )

    # If index volume is unavailable/zero,
    # use 1 so OHLC4-based cumulative VWAP still works.
    df.loc[df["Volume"] <= 0, "Volume"] = 1

    # Session date
    if isinstance(df.index, pd.DatetimeIndex):
        df["Session"] = df.index.date
    else:
        df["Session"] = 0

    df["PV"] = df["TP"] * df["Volume"]

    df["CumPV"] = (
        df.groupby("Session")["PV"]
        .cumsum()
    )

    df["CumVolume"] = (
        df.groupby("Session")["Volume"]
        .cumsum()
    )

    df["VWAP"] = (
        df["CumPV"] /
        df["CumVolume"]
    )

    # EMA slopes
    df["EMA9_Slope"] = (
        df["EMA9"] -
        df["EMA9"].shift(3)
    )

    df["EMA15_Slope"] = (
        df["EMA15"] -
        df["EMA15"].shift(3)
    )

    # Trend conditions
    df["BullTrend"] = (
        (df["Close"] > df["VWAP"]) &
        (df["EMA9"] > df["EMA15"]) &
        (df["EMA9_Slope"] > 0)
    )

    df["BearTrend"] = (
        (df["Close"] < df["VWAP"]) &
        (df["EMA9"] < df["EMA15"]) &
        (df["EMA9_Slope"] < 0)
    )

    return df


# ============================================================
# VWAP DISTANCE
# ============================================================

def vwap_distance_percent(price, vwap):

    if vwap == 0 or pd.isna(vwap):
        return 999

    return abs(price - vwap) / vwap * 100


# ============================================================
# STRATEGY SIGNAL
# ============================================================

def get_signal(df):

    if len(df) < 25:
        return {
            "signal": "WAIT",
            "status": "NO DATA",
            "reason": "Not enough candles",
            "entry": None,
            "sl": None,
            "target": None
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    close = float(last["Close"])
    high = float(last["High"])
    low = float(last["Low"])

    ema9 = float(last["EMA9"])
    ema15 = float(last["EMA15"])
    vwap = float(last["VWAP"])

    # --------------------------------------------------------
    # BUY TREND
    # --------------------------------------------------------

    bull_trend = (
        close > vwap and
        ema9 > ema15 and
        last["EMA9_Slope"] > 0
    )

    # Previous candle touches EMA zone
    buy_pullback = (
        prev["Low"] <= max(prev["EMA9"], prev["EMA15"]) and
        prev["Close"] >= prev["EMA15"]
    )

    # Bullish confirmation
    bullish_confirmation = (
        last["Close"] > last["Open"] and
        last["Close"] > last["EMA9"]
    )

    # Break previous candle high
    buy_breakout = (
        last["Close"] > prev["High"]
    )

    buy_vwap_distance = vwap_distance_percent(
        close,
        vwap
    )

    # VWAP near setup gets priority
    vwap_near_buy = buy_vwap_distance <= 0.80

    if (
        bull_trend and
        buy_pullback and
        bullish_confirmation and
        buy_breakout
    ):

        entry = close

        sl = min(
            float(prev["Low"]),
            float(last["Low"])
        )

        risk = entry - sl

        if risk > 0:

            target = entry + (risk * 2)

            return {
                "signal": "BUY",
                "status": "SETUP READY",
                "reason": (
                    "Price above VWAP + "
                    "EMA9 above EMA15 + "
                    "EMA pullback + "
                    "bullish confirmation + "
                    "previous candle high breakout"
                ),
                "entry": entry,
                "sl": sl,
                "target": target,
                "vwap_near": vwap_near_buy
            }

    # --------------------------------------------------------
    # SELL TREND
    # --------------------------------------------------------

    bear_trend = (
        close < vwap and
        ema9 < ema15 and
        last["EMA9_Slope"] < 0
    )

    # Previous candle touches EMA zone
    sell_pullback = (
        prev["High"] >= min(prev["EMA9"], prev["EMA15"]) and
        prev["Close"] <= prev["EMA15"]
    )

    # Bearish confirmation
    bearish_confirmation = (
        last["Close"] < last["Open"] and
        last["Close"] < last["EMA9"]
    )

    # Break previous candle low
    sell_breakout = (
        last["Close"] < prev["Low"]
    )

    sell_vwap_distance = vwap_distance_percent(
        close,
        vwap
    )

    vwap_near_sell = sell_vwap_distance <= 0.80

    if (
        bear_trend and
        sell_pullback and
        bearish_confirmation and
        sell_breakout
    ):

        entry = close

        sl = max(
            float(prev["High"]),
            float(last["High"])
        )

        risk = sl - entry

        if risk > 0:

            target = entry - (risk * 2)

            return {
                "signal": "SELL",
                "status": "SETUP READY",
                "reason": (
                    "Price below VWAP + "
                    "EMA9 below EMA15 + "
                    "EMA pullback + "
                    "bearish confirmation + "
                    "previous candle low breakout"
                ),
                "entry": entry,
                "sl": sl,
                "target": target,
                "vwap_near": vwap_near_sell
            }

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    if close > vwap:
        reason = "Bullish side: waiting for clean EMA pullback + confirmation"
    elif close < vwap:
        reason = "Bearish side: waiting for clean EMA pullback + confirmation"
    else:
        reason = "Waiting for clear direction around VWAP"

    return {
        "signal": "WAIT",
        "status": "WAIT",
        "reason": reason,
        "entry": None,
        "sl": None,
        "target": None
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    if len(df) < 30:
        return trades

    # --------------------------------------------------------
    # ONE MOVE = ONE TRADE
    #
    # After a trade in one direction,
    # same-direction entries remain locked.
    #
    # New trade allowed only after
    # opposite trend/reset appears.
    # --------------------------------------------------------

    direction_lock = None

    position = None

    for i in range(20, len(df) - 1):

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # ====================================================
        # OPEN POSITION MANAGEMENT
        # ====================================================

        if position is not None:

            side = position["side"]

            entry = position["entry"]
            sl = position["sl"]
            target = position["target"]

            exit_price = None
            exit_reason = None

            # ----------------------------------------------
            # BUY MANAGEMENT
            # ----------------------------------------------

            if side == "BUY":

                # Conservative order:
                # SL checked first if both occur in same candle
                if row["Low"] <= sl:
                    exit_price = sl
                    exit_reason = "STOP LOSS HIT"

                elif row["High"] >= target:
                    exit_price = target
                    exit_reason = "TARGET 1:2 HIT"

            # ----------------------------------------------
            # SELL MANAGEMENT
            # ----------------------------------------------

            elif side == "SELL":

                if row["High"] >= sl:
                    exit_price = sl
                    exit_reason = "STOP LOSS HIT"

                elif row["Low"] <= target:
                    exit_price = target
                    exit_reason = "TARGET 1:2 HIT"

            # ----------------------------------------------
            # CLOSE TRADE
            # ----------------------------------------------

            if exit_price is not None:

                if side == "BUY":
                    points = exit_price - entry
                else:
                    points = entry - exit_price

                risk = abs(entry - sl)

                r_multiple = (
                    points / risk
                    if risk > 0
                    else 0
                )

                result = (
                    "WIN"
                    if points > 0
                    else "LOSS"
                )

                trades.append({
                    "Signal": side,
                    "Entry Time": position["entry_time"],
                    "Exit Time": row.name,
                    "Entry": round(entry, 2),
                    "SL": round(sl, 2),
                    "Target": round(target, 2),
                    "Exit": round(exit_price, 2),
                    "Points": round(points, 2),
                    "R Multiple": round(r_multiple, 2),
                    "Exit Reason": exit_reason,
                    "Result": result
                })

                position = None

            continue

        # ====================================================
        # RESET LOCK
        # ====================================================

        if direction_lock == "BUY":

            if (
                row["Close"] < row["VWAP"] or
                row["EMA9"] < row["EMA15"]
            ):
                direction_lock = None

        elif direction_lock == "SELL":

            if (
                row["Close"] > row["VWAP"] or
                row["EMA9"] > row["EMA15"]
            ):
                direction_lock = None

        # ====================================================
        # BUY CONDITIONS
        # ====================================================

        prev = df.iloc[i - 1]

        bull_trend = (
            row["Close"] > row["VWAP"] and
            row["EMA9"] > row["EMA15"] and
            row["EMA9_Slope"] > 0
        )

        buy_pullback = (
            prev["Low"] <= max(prev["EMA9"], prev["EMA15"]) and
            prev["Close"] >= prev["EMA15"]
        )

        buy_confirmation = (
            row["Close"] > row["Open"] and
            row["Close"] > row["EMA9"]
        )

        buy_breakout = (
            row["Close"] > prev["High"]
        )

        if (
            direction_lock != "BUY" and
            bull_trend and
            buy_pullback and
            buy_confirmation and
            buy_breakout
        ):

            # Entry on next candle open
            entry = float(next_row["Open"])

            sl = min(
                float(prev["Low"]),
                float(row["Low"])
            )

            risk = entry - sl

            # Avoid zero/tiny risk
            if risk > 0.01:

                target = entry + (risk * 2)

                position = {
                    "side": "BUY",
                    "entry": entry,
                    "sl": sl,
                    "target": target,
                    "entry_time": next_row.name
                }

                direction_lock = "BUY"

                continue

        # ====================================================
        # SELL CONDITIONS
        # ====================================================

        bear_trend = (
            row["Close"] < row["VWAP"] and
            row["EMA9"] < row["EMA15"] and
            row["EMA9_Slope"] < 0
        )

        sell_pullback = (
            prev["High"] >= min(prev["EMA9"], prev["EMA15"]) and
            prev["Close"] <= prev["EMA15"]
        )

        sell_confirmation = (
            row["Close"] < row["Open"] and
            row["Close"] < row["EMA9"]
        )

        sell_breakout = (
            row["Close"] < prev["Low"]
        )

        if (
            direction_lock != "SELL" and
            bear_trend and
            sell_pullback and
            sell_confirmation and
            sell_breakout
        ):

            entry = float(next_row["Open"])

            sl = max(
                float(prev["High"]),
                float(row["High"])
            )

            risk = sl - entry

            if risk > 0.01:

                target = entry - (risk * 2)

                position = {
                    "side": "SELL",
                    "entry": entry,
                    "sl": sl,
                    "target": target,
                    "entry_time": next_row.name
                }

                direction_lock = "SELL"

    return trades


# ============================================================
# CHART
# ============================================================

def create_chart(df, symbol_name):

    chart_df = df.tail(375).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["EMA9"],
            mode="lines",
            name="EMA 9",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["EMA15"],
            mode="lines",
            name="EMA 15",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["VWAP"],
            mode="lines",
            name="VWAP",
            line=dict(width=1.5)
        )
    )

    fig.update_layout(
        title=f"{symbol_name} Index Chart",
        height=620,
        template="plotly_dark",
        paper_bgcolor="#111722",
        plot_bgcolor="#111722",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#27313f"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#27313f"
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("📈 Personal 9-15 EMA Scalping Scanner")

st.caption(
    "Index Only Strategy | 9 EMA + 15 EMA | "
    "VWAP = (O+H+L+C)/4 | "
    "VWAP Direction Filter | EMA Pullback | "
    "Confirmation Candle | Breakout Entry | "
    "One Move = One Trade | Fixed 1:2 Risk:Reward"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns([1, 1, 0.55])

with col1:
    selected_index = st.selectbox(
        "Select Index",
        list(INDEX_MAP.keys()),
        index=0
    )

with col2:
    selected_tf = st.selectbox(
        "Select Timeframe",
        list(TIMEFRAME_MAP.keys()),
        index=2
    )

with col3:
    st.write("")
    st.write("")
    refresh = st.button("🔄 Refresh")


# ============================================================
# LOAD DATA
# ============================================================

symbol = INDEX_MAP[selected_index]

interval, period = TIMEFRAME_MAP[selected_tf]

df = load_data(
    symbol,
    interval,
    period
)


# ============================================================
# NO DATA
# ============================================================

if df.empty:

    st.error(
        "Market data is currently unavailable. "
        "Please refresh and try again."
    )

    st.stop()


# ============================================================
# INDICATORS
# ============================================================

df = add_indicators(df)

signal = get_signal(df)

last = df.iloc[-1]


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.header("🎯 Current Signal")

signal_col, status_col = st.columns(2)

with signal_
