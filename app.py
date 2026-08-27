import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal 9-15 EMA Scalping Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
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
        background-color: #1b2b3e;
        border: 1px solid #263b52;
        border-radius: 10px;
        padding: 18px;
        min-height: 115px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 14px;
        color: #aeb8c5;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .signal-buy {
        color: #38d996;
    }

    .signal-sell {
        color: #ff6464;
    }

    .signal-wait {
        color: #ffd166;
    }

    .info-box {
        background-color: #1d344b;
        border-radius: 8px;
        padding: 14px;
        color: #b7d8f5;
        margin-bottom: 15px;
    }

    .success-box {
        background-color: #173b32;
        border-radius: 8px;
        padding: 14px;
        color: #a7e6c5;
        margin-bottom: 15px;
    }

    .warning-box {
        background-color: #40371d;
        border-radius: 8px;
        padding: 14px;
        color: #ffe5a0;
        margin-bottom: 15px;
    }

    .rule-box {
        background-color: #151d29;
        border: 1px solid #293545;
        border-radius: 8px;
        padding: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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


MIN_STOP_LOSS = 10.0
MAX_STOP_LOSS = 15.0
RISK_REWARD = 2.0


# ============================================================
# DATA DOWNLOAD
# ============================================================

@st.cache_data(ttl=60)
def load_data(symbol, interval, period):

    try:

        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.copy()

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for column in required_columns:

            if column not in df.columns:
                return pd.DataFrame()

        # Volume handling
        if "Volume" not in df.columns:
            df["Volume"] = 1

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )

        df["Volume"] = df["Volume"].fillna(0)

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA 9
    # --------------------------------------------------------

    df["EMA9"] = (
        df["Close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # EMA 15
    # --------------------------------------------------------

    df["EMA15"] = (
        df["Close"]
        .ewm(
            span=15,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # VWAP
    #
    # Strategy formula:
    # (Open + High + Low + Close) / 4
    # --------------------------------------------------------

    df["TP"] = (
        df["Open"]
        + df["High"]
        + df["Low"]
        + df["Close"]
    ) / 4

    # --------------------------------------------------------
    # Ensure valid volume
    # --------------------------------------------------------

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    ).fillna(0)

    # For index data where volume may be zero
    df.loc[
        df["Volume"] <= 0,
        "Volume"
    ] = 1

    # --------------------------------------------------------
    # SESSION VWAP
    # --------------------------------------------------------

    if isinstance(df.index, pd.DatetimeIndex):

        session_values = pd.Series(
            df.index.date,
            index=df.index
        )

        df["Session"] = session_values

    else:

        df["Session"] = 0

    df["PV"] = (
        df["TP"]
        * df["Volume"]
    )

    df["CumPV"] = (
        df.groupby("Session")["PV"]
        .cumsum()
    )

    df["CumVolume"] = (
        df.groupby("Session")["Volume"]
        .cumsum()
    )

    df["VWAP"] = (
        df["CumPV"]
        / df["CumVolume"]
    )

    # --------------------------------------------------------
    # EMA SLOPE
    # --------------------------------------------------------

    df["EMA9_Slope"] = (
        df["EMA9"]
        - df["EMA9"].shift(3)
    )

    df["EMA15_Slope"] = (
        df["EMA15"]
        - df["EMA15"].shift(3)
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["BullTrend"] = (
        (df["Close"] > df["VWAP"])
        &
        (df["EMA9"] > df["EMA15"])
        &
        (df["EMA9_Slope"] > 0)
    )

    df["BearTrend"] = (
        (df["Close"] < df["VWAP"])
        &
        (df["EMA9"] < df["EMA15"])
        &
        (df["EMA9_Slope"] < 0)
    )

    return df


# ============================================================
# VWAP DISTANCE
# ============================================================

def vwap_distance_percent(price, vwap):

    if pd.isna(vwap):
        return 999

    if vwap == 0:
        return 999

    return (
        abs(price - vwap)
        / vwap
        * 100
    )


# ============================================================
# STOP LOSS VALIDATION
# ============================================================

def valid_stop_loss(risk):

    return (
        risk >= MIN_STOP_LOSS
        and risk <= MAX_STOP_LOSS
    )


# ============================================================
# LIVE SIGNAL
# ============================================================

def get_signal(df):

    if len(df) < 25:

        return {
            "signal": "WAIT",
            "status": "NO DATA",
            "reason": "Not enough candles available.",
            "entry": None,
            "sl": None,
            "target": None,
            "risk": None,
            "vwap_near": False
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])

    ema9 = float(last["EMA9"])
    ema15 = float(last["EMA15"])
    vwap = float(last["VWAP"])

    # ========================================================
    # BUY CONDITIONS
    # ========================================================

    bull_trend = (
        close > vwap
        and ema9 > ema15
        and last["EMA9_Slope"] > 0
    )

    # Previous candle pulled back to EMA zone
    buy_pullback = (

        prev["Low"]
        <= max(
            prev["EMA9"],
            prev["EMA15"]
        )

        and

        prev["Close"]
        >= prev["EMA15"]
    )

    # Bullish confirmation
    bullish_confirmation = (

        last["Close"]
        > last["Open"]

        and

        last["Close"]
        > last["EMA9"]
    )

    # Previous candle high breakout
    buy_breakout = (

        last["Close"]
        > prev["High"]
    )

    buy_vwap_distance = (
        vwap_distance_percent(
            close,
            vwap
        )
    )

    vwap_near_buy = (
        buy_vwap_distance
        <= 0.80
    )

    if (

        bull_trend
        and buy_pullback
        and bullish_confirmation
        and buy_breakout

    ):

        entry = close

        raw_sl = min(
            float(prev["Low"]),
            float(last["Low"])
        )

        risk = entry - raw_sl

        # ----------------------------------------------------
        # ONLY 10 TO 15 POINT STOP LOSS
        # ----------------------------------------------------

        if valid_stop_loss(risk):

            target = (
                entry
                + risk * RISK_REWARD
            )

            return {

                "signal": "BUY",

                "status": "SETUP READY",

                "reason": (
                    "Price above VWAP + "
                    "EMA9 above EMA15 + "
                    "EMA slope upward + "
                    "clean EMA pullback + "
                    "bullish confirmation + "
                    "previous candle high breakout"
                ),

                "entry": entry,

                "sl": raw_sl,

                "target": target,

                "risk": risk,

                "vwap_near": vwap_near_buy
            }

    # ========================================================
    # SELL CONDITIONS
    # ========================================================

    bear_trend = (
        close < vwap
        and ema9 < ema15
        and last["EMA9_Slope"] < 0
    )

    # Previous candle pulled back to EMA zone
    sell_pullback = (

        prev["High"]
        >= min(
            prev["EMA9"],
            prev["EMA15"]
        )

        and

        prev["Close"]
        <= prev["EMA15"]
    )

    # Bearish confirmation
    bearish_confirmation = (

        last["Close"]
        < last["Open"]

        and

        last["Close"]
        < last["EMA9"]
    )

    # Previous candle low breakout
    sell_breakout = (

        last["Close"]
        < prev["Low"]
    )

    sell_vwap_distance = (
        vwap_distance_percent(
            close,
            vwap
        )
    )

    vwap_near_sell = (
        sell_vwap_distance
        <= 0.80
    )

    if (

        bear_trend
        and sell_pullback
        and bearish_confirmation
        and sell_breakout

    ):

        entry = close

        raw_sl = max(
            float(prev["High"]),
            float(last["High"])
        )

        risk = raw_sl - entry

        # ----------------------------------------------------
        # ONLY 10 TO 15 POINT STOP LOSS
        # ----------------------------------------------------

        if valid_stop_loss(risk):

            target = (
                entry
                - risk * RISK_REWARD
            )

            return {

                "signal": "SELL",

                "status": "SETUP READY",

                "reason": (
                    "Price below VWAP + "
                    "EMA9 below EMA15 + "
                    "EMA slope downward + "
                    "clean EMA pullback + "
                    "bearish confirmation + "
                    "previous candle low breakout"
                ),

                "entry": entry,

                "sl": raw_sl,

                "target": target,

                "risk": risk,

                "vwap_near": vwap_near_sell
            }

    # ========================================================
    # WAIT
    # ========================================================

    if close > vwap:

        reason = (
            "Bullish side: waiting for clean "
            "EMA pullback + confirmation + breakout."
        )

    elif close < vwap:

        reason = (
            "Bearish side: waiting for clean "
            "EMA pullback + confirmation + breakout."
        )

    else:

        reason = (
            "Waiting for clear direction around VWAP."
        )

    return {

        "signal": "WAIT",

        "status": "WAIT",

        "reason": reason,

        "entry": None,

        "sl": None,

        "target": None,

        "risk": None,

        "vwap_near": (
            vwap_distance_percent(
                close,
                vwap
            ) <= 0.80
        )
    }


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    if len(df) < 30:
        return trades

    # One direction = one trade lock
    direction_lock = None

    position = None

    for i in range(20, len(df) - 1):

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # ====================================================
        # MANAGE OPEN TRADE
        # ====================================================

        if position is not None:

            side = position["side"]

            entry = position["entry"]
            sl = position["sl"]
            target = position["target"]
            risk = position["risk"]

            exit_price = None
            exit_reason = None

            # ------------------------------------------------
            # BUY
            # ------------------------------------------------

            if side == "BUY":

                # Conservative:
                # Stop loss checked first

                if row["Low"] <= sl:

                    exit_price = sl
                    exit_reason = "STOP LOSS HIT"

                elif row["High"] >= target:

                    exit_price = target
                    exit_reason = "TARGET 1:2 HIT"

            # ------------------------------------------------
            # SELL
            # ------------------------------------------------

            elif side == "SELL":

                if row["High"] >= sl:

                    exit_price = sl
                    exit_reason = "STOP LOSS HIT"

                elif row["Low"] <= target:

                    exit_price = target
                    exit_reason = "TARGET 1:2 HIT"

            # ------------------------------------------------
            # CLOSE POSITION
            # ------------------------------------------------

            if exit_price is not None:

                if side == "BUY":

                    points = (
                        exit_price
                        - entry
                    )

                else:

                    points = (
                        entry
                        - exit_price
                    )

                r_multiple = (

                    points
                    / risk

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

                    "Entry Time":
                        position["entry_time"],

                    "Exit Time":
                        row.name,

                    "Entry":
                        round(entry, 2),

                    "SL":
                        round(sl, 2),

                    "Risk (Pts)":
                        round(risk, 2),

                    "Target":
                        round(target, 2),

                    "Exit":
                        round(exit_price, 2),

                    "Points":
                        round(points, 2),

                    "R Multiple":
                        round(r_multiple, 2),

                    "Exit Reason":
                        exit_reason,

                    "Result":
                        result
                })

                position = None

            continue

        # ====================================================
        # RESET ONE MOVE LOCK
        # ====================================================

        if direction_lock == "BUY":

            if (

                row["Close"]
                < row["VWAP"]

                or

                row["EMA9"]
                < row["EMA15"]

            ):

                direction_lock = None

        elif direction_lock == "SELL":

            if (

                row["Close"]
                > row["VWAP"]

                or

                row["EMA9"]
                > row["EMA15"]

            ):

                direction_lock = None

        prev = df.iloc[i - 1]

        # ====================================================
        # BUY BACKTEST CONDITIONS
        # ====================================================

        bull_trend = (

            row["Close"]
            > row["VWAP"]

            and

            row["EMA9"]
            > row["EMA15"]

            and

            row["EMA9_Slope"]
            > 0
        )

        buy_pullback = (

            prev["Low"]
            <= max(
                prev["EMA9"],
                prev["EMA15"]
            )

            and

            prev["Close"]
            >= prev["EMA15"]
        )

        buy_confirmation = (

            row["Close"]
            > row["Open"]

            and

            row["Close"]
            > row["EMA9"]
        )

        buy_breakout = (

            row["Close"]
            > prev["High"]
        )

        if (

            direction_lock != "BUY"

            and bull_trend

            and buy_pullback

            and buy_confirmation

            and buy_breakout

        ):

            # Entry on next candle open
            entry = float(
                next_row["Open"]
            )

            sl = min(

                float(prev["Low"]),

                float(row["Low"])
            )

            risk = entry - sl

            # ------------------------------------------------
            # VALID ONLY 10 TO 15 POINT SL
            # ------------------------------------------------

            if valid_stop_loss(risk):

                target = (

                    entry
                    + risk * RISK_REWARD
                )

                position = {

                    "side": "BUY",

                    "entry": entry,

                    "sl": sl,

                    "target": target,

                    "risk": risk,

                    "entry_time":
                        next_row.name
                }

                # One move one trade
                direction_lock = "BUY"

                continue

        # ====================================================
        # SELL BACKTEST CONDITIONS
        # ====================================================

        bear_trend = (

            row["Close"]
            < row["VWAP"]

            and

            row["EMA9"]
            < row["EMA15"]

            and

            row["EMA9_Slope"]
            < 0
        )

        sell_pullback = (

            prev["High"]
            >= min(
                prev["EMA9"],
                prev["EMA15"]
            )

            and

            prev["Close"]
            <= prev["EMA15"]
        )

        sell_confirmation = (

            row["Close"]
            < row["Open"]

            and

            row["Close"]
            < row["EMA9"]
        )

        sell_breakout = (

            row["Close"]
            < prev["Low"]
        )

        if (

            direction_lock != "SELL"

            and bear_trend

            and sell_pullback

            and sell_confirmation

            and sell_breakout

        ):

            entry = float(
                next_row["Open"]
            )

            sl = max(

                float(prev["High"]),

                float(row["High"])
            )

            risk = sl - entry

            # ------------------------------------------------
            # VALID ONLY 10 TO 15 POINT SL
            # ------------------------------------------------

            if valid_stop_loss(risk):

                target = (

                    entry
                    - risk * RISK_REWARD
                )

                position = {

                    "side": "SELL",

                    "entry": entry,

                    "sl": sl,

                    "target": target,

                    "risk": risk,

                    "entry_time":
                        next_row.name
                }

                # One move one trade
                direction_lock = "SELL"

    return trades


# ============================================================
# CREATE CHART
# ============================================================

def create_chart(df, symbol_name):

    chart_df = df.tail(375).copy()

    fig = go.Figure()

    # --------------------------------------------------------
    # CANDLESTICK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMA 9
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=chart_df.index,

            y=chart_df["EMA9"],

            mode="lines",

            name="EMA 9",

            line=dict(width=1.5)
        )
    )

    # --------------------------------------------------------
    # EMA 15
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=chart_df.index,

            y=chart_df["EMA15"],

            mode="lines",

            name="EMA 15",

            line=dict(width=1.5)
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=chart_df.index,

            y=chart_df["VWAP"],

            mode="lines",

            name="VWAP",

            line=dict(width=1.5)
        )
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.update_layout(

        title=f"{symbol_name} Index Chart",

        height=620,

        template="plotly_dark",

        paper_bgcolor="#111722",

        plot_bgcolor="#111722",

        xaxis_rangeslider_visible=False,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),

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
    "Index Only Strategy | "
    "9 EMA + 15 EMA | "
    "VWAP = (O+H+L+C)/4 | "
    "VWAP Direction Filter | "
    "EMA Pullback | "
    "Confirmation Candle | "
    "Breakout Entry | "
    "One Move = One Trade | "
    "Stop Loss 10-15 Points | "
    "Fixed 1:2 Risk:Reward"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns(
    [1, 1, 0.55]
)


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

    if st.button("🔄 Refresh"):

        load_data.clear()

        st.rerun()


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
        "Please press Refresh and try again."
    )

    st.stop()


# ============================================================
# ADD INDICATORS
# ============================================================

df = add_indicators(df)

signal = get_signal(df)

last = df.iloc[-1]


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.header("🎯 Current Signal")

signal_col, status_col = st.columns(2)


# ------------------------------------------------------------
# SIGNAL BOX
# ------------------------------------------------------------

with signal_col:

    signal_name = signal["signal"]

    if signal_name == "BUY":

        signal_class = "signal-buy"

    elif signal_name == "SELL":

        signal_class = "signal-sell"

    else:

        signal_class = "signal-wait"

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">
                Signal
            </div>

            <div class="metric-value {signal_class}">
                {signal_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# STATUS BOX
# ------------------------------------------------------------

with status_col:

    status_value = signal["status"]

    if status_value == "SETUP READY":

        status_class = "signal-buy"

    else:

        status_class = "signal-wait"

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">
                Trade Status
            </div>

            <div class="metric-value {status_class}">
                {status_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MARKET VALUES
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Price",
    f"{float(last['Close']):.2f}"
)

metric2.metric(
    "EMA 9",
    f"{float(last['EMA9']):.2f}"
)

metric3.metric(
    "EMA 15",
    f"{float(last['EMA15']):.2f}"
)

metric4.metric(
    "VWAP",
    f"{float(last['VWAP']):.2f}"
)


# ============================================================
# SIGNAL CONDITION
# ============================================================

st.subheader("Signal Condition")

st.write(signal["reason"])

if signal["vwap_near"]:

    st.markdown(
        """
        <div class="success-box">
        ✓ VWAP is relatively near the current price.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="info-box">
        VWAP direction filter is active.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRADE LEVELS
# ============================================================

if signal["signal"] in ["BUY", "SELL"]:

    st.header("📍 Trade Levels")

    entry_col, sl_col, target_col = st.columns(3)

    entry_col.metric(
        "Entry",
        f"{signal['entry']:.2f}"
    )

    sl_col.metric(
        "Stop Loss",
        f"{signal['sl']:.2f}"
    )

    target_col.metric(
        "Target (1:2)",
        f"{signal['target']:.2f}"
    )

    st.caption(
        f"Risk: {signal['risk']:.2f} points | "
        f"Stop Loss rule: "
        f"{MIN_STOP_LOSS:.0f}-{MAX_STOP_LOSS:.0f} points | "
        f"Target = 2R"
    )

else:

    st.markdown(
        """
        <div class="warning-box">
        No active trade setup. Wait for all strategy
        conditions to match.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# STRATEGY RULES
# ============================================================

with st.expander("📘 Strategy Rules"):

    rule_col1, rule_col2 = st.columns(2)

    with rule_col1:

        st.markdown("### 🟢 BUY")

        st.markdown(
            """
            1. Price VWAP के ऊपर  
            2. EMA 9 EMA 15 के ऊपर  
            3. EMA 9 slope upward  
            4. Price EMA 9/15 zone तक pullback करे  
            5. Bullish confirmation candle बने  
            6. Previous candle high breakout  
            7. एक move में केवल एक trade  
            8. Stop Loss minimum 10 points  
            9. Stop Loss maximum 15 points  
            10. Target fixed 1:2
            """
        )

    with rule_col2:

        st.markdown("### 🔴 SELL")

        st.markdown(
            """
            1. Price VWAP के नीचे  
            2. EMA 9 EMA 15 के नीचे  
            3. EMA 9 slope downward  
            4. Price EMA 9/15 zone तक pullback करे  
            5. Bearish confirmation candle बने  
            6. Previous candle low breakout  
            7. एक move में केवल एक trade  
            8. Stop Loss minimum 10 points  
            9. Stop Loss maximum 15 points  
            10. Target fixed 1:2
            """
        )


# ============================================================
# INDEX CHART
# ============================================================

st.header("📊 Index Chart")

fig = create_chart(
    df,
    selected_index
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

st.header("📉 Historical Backtest")

trades = run_backtest(df)


# ============================================================
# BACKTEST RESULTS
# ============================================================

if len(trades) == 0:

    st.info(
        "No completed historical trades found "
        "with the current strategy rules."
    )

else:

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)

    wins = len(
        trades_df[
            trades_df["Result"] == "WIN"
        ]
    )

    losses = len(
        trades_df[
            trades_df["Result"] == "LOSS"
        ]
    )

    win_rate = (

        wins
        / total_trades
        * 100

        if total_trades > 0

        else 0
    )

    net_points = (
        trades_df["Points"]
        .sum()
    )

    average_r = (
        trades_df["R Multiple"]
        .mean()
    )

    best_trade = (
        trades_df["R Multiple"]
        .max()
    )

    # --------------------------------------------------------
    # BACKTEST METRICS
    # --------------------------------------------------------

    back_cols = st.columns(7)

    back_cols[0].metric(
        "Total Trades",
        total_trades
    )

    back_cols[1].metric(
        "Wins",
        wins
    )

    back_cols[2].metric(
        "Losses",
        losses
    )

    back_cols[3].metric(
        "Win Rate",
        f"{win_rate:.1f}%"
    )

    back_cols[4].metric(
        "Net Points",
        f"{net_points:.2f}"
    )

    back_cols[5].metric(
        "Average R",
        f"{average_r:.2f}R"
    )

    back_cols[6].metric(
        "Best Trade",
        f"{best_trade:.2f}R"
    )

    # --------------------------------------------------------
    # RECENT BACKTEST TRADES
    # --------------------------------------------------------

    st.subheader("Recent Backtest Trades")

    display_df = (
        trades_df
        .tail(20)
        .iloc[::-1]
        .copy()
    )

    # Format datetime columns
    for time_col in [
        "Entry Time",
        "Exit Time"
    ]:

        if time_col in display_df.columns:

            display_df[time_col] = pd.to_datetime(
                display_df[time_col]
            ).astype(str)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Strategy: "
    "VWAP = Direction | "
    "EMA 9/15 = Trend | "
    "Pullback + Confirmation + Breakout = Entry | "
    "One Move = One Trade | "
    "Stop Loss = 10-15 Points | "
    "Target = Fixed 1:2"
)

st.caption(
    f"Last data candle: {df.index[-1]}"
)
