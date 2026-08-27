import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal 9-15 EMA Scalping Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background-color: #1f2d3d;
        border: 1px solid #2f4257;
        border-radius: 10px;
        padding: 16px;
        min-height: 100px;
    }

    .signal-box {
        background-color: #1f2d3d;
        border: 1px solid #2f4257;
        border-radius: 10px;
        padding: 18px;
    }

    .small-text {
        font-size: 12px;
        opacity: 0.75;
    }

    .footer-text {
        font-size: 12px;
        opacity: 0.65;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INDEX SYMBOLS
# ============================================================

INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
    "SENSEX": "^BSESN"
}


# ============================================================
# STRATEGY SETTINGS
# ============================================================

EMA_FAST = 9
EMA_SLOW = 15

RISK_REWARD = 2.0

# EMA pullback zone tolerance
EMA_PULLBACK_ATR_MULTIPLIER = 0.35

# Minimum candle body compared with ATR
MIN_CONFIRMATION_BODY_ATR = 0.10

# Maximum bars allowed after pullback
MAX_PULLBACK_BARS = 5

# Avoid extremely flat EMA
MIN_EMA_GAP_ATR = 0.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_columns(df):
    """
    Handles yfinance MultiIndex columns if returned.
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    return df


def safe_number(value):
    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

@st.cache_data(ttl=60)
def load_market_data(symbol, period, interval):

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = normalize_columns(df)

        required = ["Open", "High", "Low", "Close"]

        for column in required:
            if column not in df.columns:
                return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 1.0

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        ).copy()

        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        ).fillna(0)

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# TIMEFRAME CONFIGURATION
# ============================================================

def get_period_for_interval(interval):

    mapping = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "1d": "2y"
    }

    return mapping.get(interval, "60d")


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] -
        df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] -
        df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


# ============================================================
# VWAP
# IMPORTANT:
# USER REQUESTED PRICE = (O + H + L + C) / 4
# ============================================================

def calculate_session_vwap(df):

    df = df.copy()

    # Requested VWAP source price
    df["VWAP_PRICE"] = (
        df["Open"] +
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 4.0

    # Volume safety
    volume = df["Volume"].copy()

    # Some index feeds have zero volume.
    # In that case use 1 so the VWAP calculation remains usable.
    volume = volume.where(volume > 0, 1.0)

    df["_VWAP_VOLUME"] = volume

    # Session/day grouping
    if isinstance(df.index, pd.DatetimeIndex):

        try:
            session_key = pd.Series(
                df.index.tz_localize(None).date,
                index=df.index
            )
        except Exception:
            session_key = pd.Series(
                df.index.date,
                index=df.index
            )

    else:
        session_key = pd.Series(
            np.zeros(len(df)),
            index=df.index
        )

    price_volume = (
        df["VWAP_PRICE"] *
        df["_VWAP_VOLUME"]
    )

    cumulative_pv = (
        price_volume
        .groupby(session_key)
        .cumsum()
    )

    cumulative_volume = (
        df["_VWAP_VOLUME"]
        .groupby(session_key)
        .cumsum()
    )

    df["VWAP"] = (
        cumulative_pv /
        cumulative_volume.replace(0, np.nan)
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["EMA9"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["EMA15"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    df["ATR"] = calculate_atr(df, 14)

    df = calculate_session_vwap(df)

    return df


# ============================================================
# TREND
# ============================================================

def get_market_bias(row):

    if (
        row["Close"] > row["VWAP"] and
        row["EMA9"] > row["EMA15"]
    ):
        return "BULLISH"

    if (
        row["Close"] < row["VWAP"] and
        row["EMA9"] < row["EMA15"]
    ):
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# EMA SEPARATION
# ============================================================

def ema_has_separation(row):

    atr = row["ATR"]

    if pd.isna(atr) or atr <= 0:
        return False

    ema_gap = abs(
        row["EMA9"] -
        row["EMA15"]
    )

    return ema_gap >= (
        atr * MIN_EMA_GAP_ATR
    )


# ============================================================
# CHECK EMA PULLBACK
# ============================================================

def bullish_pullback(row):

    tolerance = (
        row["ATR"] *
        EMA_PULLBACK_ATR_MULTIPLIER
    )

    # Candle comes near EMA area
    near_ema = (
        row["Low"] <= row["EMA9"] + tolerance
    )

    # Pullback should not completely break bearish
    above_slow_ema = (
        row["Close"] >= row["EMA15"] - tolerance
    )

    return near_ema and above_slow_ema


def bearish_pullback(row):

    tolerance = (
        row["ATR"] *
        EMA_PULLBACK_ATR_MULTIPLIER
    )

    near_ema = (
        row["High"] >= row["EMA9"] - tolerance
    )

    below_slow_ema = (
        row["Close"] <= row["EMA15"] + tolerance
    )

    return near_ema and below_slow_ema


# ============================================================
# CONFIRMATION CANDLE
# ============================================================

def bullish_confirmation(row):

    body = abs(
        row["Close"] -
        row["Open"]
    )

    minimum_body = (
        row["ATR"] *
        MIN_CONFIRMATION_BODY_ATR
    )

    return (
        row["Close"] > row["Open"] and
        row["Close"] > row["EMA9"] and
        body >= minimum_body
    )


def bearish_confirmation(row):

    body = abs(
        row["Close"] -
        row["Open"]
    )

    minimum_body = (
        row["ATR"] *
        MIN_CONFIRMATION_BODY_ATR
    )

    return (
        row["Close"] < row["Open"] and
        row["Close"] < row["EMA9"] and
        body >= minimum_body
    )


# ============================================================
# STRATEGY SIGNAL
#
# BULLISH:
# 1. Price above VWAP
# 2. EMA9 above EMA15
# 3. Pullback near EMA
# 4. Confirmation candle
# 5. Break confirmation high
#
# BEARISH:
# 1. Price below VWAP
# 2. EMA9 below EMA15
# 3. Pullback near EMA
# 4. Confirmation candle
# 5. Break confirmation low
# ============================================================

def detect_setup(df, i):

    if i < 20:
        return None

    row = df.iloc[i]

    if (
        pd.isna(row["EMA9"]) or
        pd.isna(row["EMA15"]) or
        pd.isna(row["VWAP"]) or
        pd.isna(row["ATR"])
    ):
        return None

    if not ema_has_separation(row):
        return None

    # ========================================================
    # BULLISH SETUP
    # ========================================================

    if (
        row["Close"] > row["VWAP"] and
        row["EMA9"] > row["EMA15"]
    ):

        start = max(
            0,
            i - MAX_PULLBACK_BARS
        )

        pullback_found = False

        for j in range(start, i):

            previous = df.iloc[j]

            if (
                previous["Close"] > previous["VWAP"] and
                previous["EMA9"] > previous["EMA15"] and
                bullish_pullback(previous)
            ):
                pullback_found = True
                break

        if (
            pullback_found and
            bullish_confirmation(row)
        ):

            return {
                "signal": "BUY",
                "confirmation_high": row["High"],
                "confirmation_low": row["Low"],
                "signal_index": i
            }

    # ========================================================
    # BEARISH SETUP
    # ========================================================

    if (
        row["Close"] < row["VWAP"] and
        row["EMA9"] < row["EMA15"]
    ):

        start = max(
            0,
            i - MAX_PULLBACK_BARS
        )

        pullback_found = False

        for j in range(start, i):

            previous = df.iloc[j]

            if (
                previous["Close"] < previous["VWAP"] and
                previous["EMA9"] < previous["EMA15"] and
                bearish_pullback(previous)
            ):
                pullback_found = True
                break

        if (
            pullback_found and
            bearish_confirmation(row)
        ):

            return {
                "signal": "SELL",
                "confirmation_high": row["High"],
                "confirmation_low": row["Low"],
                "signal_index": i
            }

    return None


# ============================================================
# CURRENT SIGNAL
# ============================================================

def get_current_signal(df):

    if len(df) < 30:

        return {
            "signal": "WAIT",
            "status": "NO TRADE",
            "reason": "Not enough candle data"
        }

    i = len(df) - 1

    row = df.iloc[i]

    setup = detect_setup(df, i)

    bias = get_market_bias(row)

    if setup is None:

        if bias == "BULLISH":

            reason = (
                "Bullish bias: Price above VWAP, "
                "waiting for EMA pullback + confirmation"
            )

        elif bias == "BEARISH":

            reason = (
                "Bearish bias: Price below VWAP, "
                "waiting for EMA pullback + confirmation"
            )

        else:

            reason = (
                "No clear alignment between "
                "Price, VWAP and 9/15 EMA"
            )

        return {
            "signal": "WAIT",
            "status": "NO TRADE",
            "reason": reason
        }

    signal = setup["signal"]

    if signal == "BUY":

        entry = setup["confirmation_high"]

        stop_loss = setup["confirmation_low"]

        risk = entry - stop_loss

        if risk <= 0:
            return {
                "signal": "WAIT",
                "status": "NO TRADE",
                "reason": "Invalid BUY risk structure"
            }

        target = (
            entry +
            risk * RISK_REWARD
        )

        return {
            "signal": "BUY",
            "status": "SETUP READY",
            "reason": (
                "Price above VWAP + "
                "EMA9 above EMA15 + "
                "EMA pullback + bullish confirmation. "
                "Entry only above confirmation candle high."
            ),
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target
        }

    else:

        entry = setup["confirmation_low"]

        stop_loss = setup["confirmation_high"]

        risk = stop_loss - entry

        if risk <= 0:
            return {
                "signal": "WAIT",
                "status": "NO TRADE",
                "reason": "Invalid SELL risk structure"
            }

        target = (
            entry -
            risk * RISK_REWARD
        )

        return {
            "signal": "SELL",
            "status": "SETUP READY",
            "reason": (
                "Price below VWAP + "
                "EMA9 below EMA15 + "
                "EMA pullback + bearish confirmation. "
                "Entry only below confirmation candle low."
            ),
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target
        }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    if len(df) < 40:
        return trades

    i = 25

    while i < len(df) - 2:

        setup = detect_setup(df, i)

        if setup is None:
            i += 1
            continue

        signal = setup["signal"]

        signal_time = df.index[i]

        # Entry starts from next candle
        entry_candle_index = i + 1

        next_row = df.iloc[entry_candle_index]

        if signal == "BUY":

            entry_price = setup["confirmation_high"]
            stop_loss = setup["confirmation_low"]

            risk = (
                entry_price -
                stop_loss
            )

            if risk <= 0:
                i += 1
                continue

            target = (
                entry_price +
                risk * RISK_REWARD
            )

            # Next candle must break entry
            if next_row["High"] < entry_price:
                i += 1
                continue

        else:

            entry_price = setup["confirmation_low"]
            stop_loss = setup["confirmation_high"]

            risk = (
                stop_loss -
                entry_price
            )

            if risk <= 0:
                i += 1
                continue

            target = (
                entry_price -
                risk * RISK_REWARD
            )

            if next_row["Low"] > entry_price:
                i += 1
                continue

        entry_time = df.index[entry_candle_index]

        exit_time = None
        exit_price = None
        exit_reason = None

        # Start after entry candle
        j = entry_candle_index

        while j < len(df):

            candle = df.iloc[j]

            if signal == "BUY":

                stop_hit = (
                    candle["Low"] <= stop_loss
                )

                target_hit = (
                    candle["High"] >= target
                )

                # Conservative:
                # if both hit in same candle,
                # count stop first
                if stop_hit:

                    exit_price = stop_loss
                    exit_time = df.index[j]
                    exit_reason = "STOP LOSS"
                    break

                if target_hit:

                    exit_price = target
                    exit_time = df.index[j]
                    exit_reason = "TARGET"
                    break

            else:

                stop_hit = (
                    candle["High"] >= stop_loss
                )

                target_hit = (
                    candle["Low"] <= target
                )

                if stop_hit:

                    exit_price = stop_loss
                    exit_time = df.index[j]
                    exit_reason = "STOP LOSS"
                    break

                if target_hit:

                    exit_price = target
                    exit_time = df.index[j]
                    exit_reason = "TARGET"
                    break

            j += 1

        # If trade not completed,
        # close at final available candle
        if exit_price is None:

            exit_price = df.iloc[-1]["Close"]
            exit_time = df.index[-1]
            exit_reason = "DATA END"

        if signal == "BUY":

            points = (
                exit_price -
                entry_price
            )

        else:

            points = (
                entry_price -
                exit_price
            )

        result = (
            "WIN"
            if points > 0
            else "LOSS"
        )

        trades.append({
            "Signal": signal,
            "Strength": "CONFIRMED",
            "Signal Time": signal_time,
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Entry": round(entry_price, 2),
            "SL": round(stop_loss, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Points": round(points, 2),
            "Exit Reason": exit_reason,
            "Result": result
        })

        # Move after exit
        i = max(
            j + 1,
            i + 1
        )

    return trades


# ============================================================
# CHART
# ============================================================

def create_chart(df, index_name):

    chart_df = df.tail(250).copy()

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
            name="EMA 9",
            mode="lines",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["EMA15"],
            name="EMA 15",
            mode="lines",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["VWAP"],
            name="VWAP",
            mode="lines",
            line=dict(width=1.5)
        )
    )

    fig.update_layout(
        title=f"{index_name} Index Chart",
        height=600,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_yaxes(
        side="right",
        fixedrange=False
    )

    fig.update_xaxes(
        fixedrange=False
    )

    return fig


# ============================================================
# APP HEADER
# ============================================================

st.title("📈 Personal 9-15 EMA Scalping Scanner")

st.caption(
    "Index Only Strategy | "
    "9 EMA + 15 EMA | "
    "VWAP = (O+H+L+C)/4 | "
    "VWAP Trend Filter | "
    "EMA Pullback + Confirmation | "
    "Fixed 1:2 Risk:Reward"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns(
    [1.2, 1.2, 0.6]
)

with col1:

    selected_index = st.selectbox(
        "Select Index",
        list(INDEX_SYMBOLS.keys()),
        index=0
    )


with col2:

    selected_timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "2m", "5m", "15m", "30m", "1h"],
        index=2
    )


with col3:

    st.write("")

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):
        load_market_data.clear()
        st.rerun()


symbol = INDEX_SYMBOLS[selected_index]

period = get_period_for_interval(
    selected_timeframe
)


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading market data..."):

    data = load_market_data(
        symbol,
        period,
        selected_timeframe
    )


if data.empty:

    st.error(
        "Market data could not be loaded. "
        "Please try Refresh."
    )

    st.stop()


data = add_indicators(data)

data = data.dropna().copy()


if data.empty:

    st.error(
        "Not enough valid candle data."
    )

    st.stop()


# ============================================================
# CURRENT SIGNAL
# ============================================================

current = get_current_signal(data)

latest = data.iloc[-1]


st.markdown("## 🎯 Current Signal")


signal_col, status_col = st.columns(2)


with signal_col:

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="small-text">Signal</div>
            <h3>{current["signal"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


with status_col:

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="small-text">Trade Status</div>
            <h3>{current["status"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MARKET VALUES
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Price",
    f"{latest['Close']:.2f}"
)

m2.metric(
    "EMA 9",
    f"{latest['EMA9']:.2f}"
)

m3.metric(
    "EMA 15",
    f"{latest['EMA15']:.2f}"
)

m4.metric(
    "VWAP",
    f"{latest['VWAP']:.2f}"
)


# ============================================================
# SIGNAL CONDITION
# ============================================================

st.markdown("### Signal Condition")

st.write(
    current["reason"]
)


# ============================================================
# LIVE TRADE LEVELS
# ============================================================

if current["signal"] in ["BUY", "SELL"]:

    st.markdown("### 📍 Trade Levels")

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Entry",
        f"{current['entry']:.2f}"
    )

    e2.metric(
        "Stop Loss",
        f"{current['stop_loss']:.2f}"
    )

    e3.metric(
        "Target (1:2)",
        f"{current['target']:.2f}"
    )


# ============================================================
# STRATEGY EXPLANATION
# ============================================================

with st.expander(
    "📘 Strategy Rules"
):

    st.markdown("""
### Bullish Trade

1. Price VWAP के ऊपर होना चाहिए.
2. 9 EMA, 15 EMA के ऊपर होना चाहिए.
3. Price EMA area की तरफ pullback करे.
4. Pullback के बाद bullish confirmation candle बने.
5. Confirmation candle के High के ऊपर entry.
6. Stop Loss confirmation candle के Low पर.
7. Target = Risk × 2.

### Bearish Trade

1. Price VWAP के नीचे होना चाहिए.
2. 9 EMA, 15 EMA के नीचे होना चाहिए.
3. Price EMA area की तरफ pullback करे.
4. Pullback के बाद bearish confirmation candle बने.
5. Confirmation candle के Low के नीचे entry.
6. Stop Loss confirmation candle के High पर.
7. Target = Risk × 2.

### VWAP का काम

- Price VWAP के ऊपर → केवल Bullish trade खोजें.
- Price VWAP के नीचे → केवल Bearish trade खोजें.
- VWAP cross को entry signal नहीं माना गया है.
- Actual entry EMA pullback + confirmation से होती है.
""")


# ============================================================
# LIVE STATUS
# ============================================================

st.markdown("## 🔵 Running Trade")

if current["signal"] in ["BUY", "SELL"]:

    st.info(
        f"Active setup: {current['signal']} "
        f"| Entry: {current['entry']:.2f} "
        f"| SL: {current['stop_loss']:.2f} "
        f"| Target: {current['target']:.2f}"
    )

else:

    st.info(
        "No running trade. Waiting for a valid setup."
    )


# ============================================================
# INDEX CHART
# ============================================================

st.markdown("## 📊 Index Chart")

chart = create_chart(
    data,
    selected_index
)

st.plotly_chart(
    chart,
    use_container_width=True
)


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

st.markdown("## 📉 Historical Backtest")


with st.spinner("Running historical backtest..."):

    backtest_trades = run_backtest(data)


total_trades = len(backtest_trades)

wins = sum(
    1
    for trade in backtest_trades
    if trade["Result"] == "WIN"
)

losses = sum(
    1
    for trade in backtest_trades
    if trade["Result"] == "LOSS"
)

breakeven = sum(
    1
    for trade in backtest_trades
    if trade["Points"] == 0
)

net_points = sum(
    trade["Points"]
    for trade in backtest_trades
)

win_rate = (
    (wins / total_trades) * 100
    if total_trades > 0
    else 0
)


# ============================================================
# BACKTEST METRICS
# ============================================================

b1, b2, b3, b4 = st.columns(4)

b1.metric(
    "Total Trades",
    total_trades
)

b2.metric(
    "Wins",
    wins
)

b3.metric(
    "Losses",
    losses
)

b4.metric(
    "Win Rate",
    f"{win_rate:.1f}%"
)


b5, b6, b7 = st.columns(3)

b5.metric(
    "Breakeven",
    breakeven
)

b6.metric(
    "Net Points",
    f"{net_points:.2f}"
)

b7.metric(
    "Candles Tested",
    len(data)
)


# ============================================================
# RECENT BACKTEST TRADES
# ============================================================

st.markdown("### Recent Backtest Trades")

if backtest_trades:

    trades_df = pd.DataFrame(
        backtest_trades
    )

    trades_df = trades_df.iloc[::-1].copy()

    display_columns = [
        "Signal",
        "Strength",
        "Entry Time",
        "Exit Time",
        "Entry",
        "SL",
        "Target",
        "Exit",
        "Points",
        "Exit Reason",
        "Result"
    ]

    for column in [
        "Entry Time",
        "Exit Time"
    ]:

        if column in trades_df.columns:

            trades_df[column] = pd.to_datetime(
                trades_df[column]
            ).astype(str)

    st.dataframe(
        trades_df[display_columns],
        use_container_width=True,
        hide_index=True,
        height=450
    )

else:

    st.info(
        "No valid historical trades found for this strategy and available data."
    )


# ============================================================
# RECENT LIVE CLOSED TRADES
# ============================================================

st.markdown("## Recent Live Closed Trades")

l1, l2, l3, l4 = st.columns(4)

l1.metric(
    "Closed Trades",
    0
)

l2.metric(
    "Wins",
    0
)

l3.metric(
    "Win Rate",
    "0.0%"
)

l4.metric(
    "Net Points",
    "0.00"
)


st.info(
    "Live trade tracking starts when a separate persistent live-trade engine is added."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
    9 EMA + 15 EMA Index Scalping Strategy |
    VWAP = (O + H + L + C) / 4 |
    VWAP Trend Filter |
    EMA Pullback |
    Confirmation Candle |
    Breakout Entry |
    Fixed 1:2 Risk:Reward
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LAST UPDATED
# ============================================================

st.caption(
    f"Last data check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
