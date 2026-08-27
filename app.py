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

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in required_columns:
            if col not in df.columns:
                return pd.DataFrame()

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
    # Strategy Formula:
    # (Open + High + Low + Close) / 4
    # --------------------------------------------------------

    df["TP"] = (
        df["Open"] +
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 4

    # --------------------------------------------------------
    # VOLUME CLEANUP
    # --------------------------------------------------------

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    ).fillna(0)

    df.loc[
        df["Volume"] <= 0,
        "Volume"
    ] = 1

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    if isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        df["Session"] = df.index.date
    else:
        df["Session"] = 0

    # --------------------------------------------------------
    # VWAP CALCULATION
    # --------------------------------------------------------

    df["PV"] = (
        df["TP"] *
        df["Volume"]
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
        df["CumPV"] /
        df["CumVolume"]
    )

    # --------------------------------------------------------
    # EMA SLOPES
    # --------------------------------------------------------

    df["EMA9_Slope"] = (
        df["EMA9"] -
        df["EMA9"].shift(3)
    )

    df["EMA15_Slope"] = (
        df["EMA15"] -
        df["EMA15"].shift(3)
    )

    return df


# ============================================================
# VWAP DISTANCE
# ============================================================

def vwap_distance_percent(price, vwap):

    if pd.isna(vwap) or vwap == 0:
        return 999.0

    return abs(price - vwap) / vwap * 100


# ============================================================
# SIGNAL FUNCTION
# ============================================================

def get_signal(df):

    if len(df) < 25:

        return {
            "signal": "WAIT",
            "status": "NO DATA",
            "reason": "Not enough candles",
            "entry": None,
            "sl": None,
            "target": None,
            "vwap_near": False
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema15 = float(last["EMA15"])
    vwap = float(last["VWAP"])

    # --------------------------------------------------------
    # VWAP DISTANCE
    # --------------------------------------------------------

    distance = vwap_distance_percent(
        close,
        vwap
    )

    vwap_near = distance <= 0.80

    # ========================================================
    # BUY CONDITIONS
    # ========================================================

    bull_trend = (
        close > vwap and
        ema9 > ema15 and
        last["EMA9_Slope"] > 0
    )

    buy_pullback = (
        prev["Low"] <= max(
            prev["EMA9"],
            prev["EMA15"]
        )
        and
        prev["Close"] >= prev["EMA15"]
    )

    bullish_confirmation = (
        last["Close"] > last["Open"]
        and
        last["Close"] > last["EMA9"]
    )

    buy_breakout = (
        last["Close"] > prev["High"]
    )

    if (
        bull_trend
        and buy_pullback
        and bullish_confirmation
        and buy_breakout
    ):

        entry = close

        sl = min(
            float(prev["Low"]),
            float(last["Low"])
        )

        risk = entry - sl

        if risk > 0:

            target = entry + (
                risk * 2
            )

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
                "vwap_near": vwap_near
            }

    # ========================================================
    # SELL CONDITIONS
    # ========================================================

    bear_trend = (
        close < vwap and
        ema9 < ema15 and
        last["EMA9_Slope"] < 0
    )

    sell_pullback = (
        prev["High"] >= min(
            prev["EMA9"],
            prev["EMA15"]
        )
        and
        prev["Close"] <= prev["EMA15"]
    )

    bearish_confirmation = (
        last["Close"] < last["Open"]
        and
        last["Close"] < last["EMA9"]
    )

    sell_breakout = (
        last["Close"] < prev["Low"]
    )

    if (
        bear_trend
        and sell_pullback
        and bearish_confirmation
        and sell_breakout
    ):

        entry = close

        sl = max(
            float(prev["High"]),
            float(last["High"])
        )

        risk = sl - entry

        if risk > 0:

            target = entry - (
                risk * 2
            )

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
                "vwap_near": vwap_near
            }

    # ========================================================
    # WAIT
    # ========================================================

    if close > vwap:

        reason = (
            "Bullish side: waiting for clean "
            "EMA pullback + confirmation + breakout"
        )

    elif close < vwap:

        reason = (
            "Bearish side: waiting for clean "
            "EMA pullback + confirmation + breakout"
        )

    else:

        reason = (
            "Waiting for clear direction around VWAP"
        )

    return {
        "signal": "WAIT",
        "status": "WAIT",
        "reason": reason,
        "entry": None,
        "sl": None,
        "target": None,
        "vwap_near": vwap_near
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
    # --------------------------------------------------------

    direction_lock = None

    position = None

    for i in range(
        20,
        len(df) - 1
    ):

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

            # ------------------------------------------------
            # BUY
            # ------------------------------------------------

            if side == "BUY":

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
                        exit_price -
                        entry
                    )

                else:

                    points = (
                        entry -
                        exit_price
                    )

                risk = abs(
                    entry - sl
                )

                if risk > 0:

                    r_multiple = (
                        points /
                        risk
                    )

                else:

                    r_multiple = 0

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
                row["Close"] < row["VWAP"]
                or
                row["EMA9"] < row["EMA15"]
            ):

                direction_lock = None

        elif direction_lock == "SELL":

            if (
                row["Close"] > row["VWAP"]
                or
                row["EMA9"] > row["EMA15"]
            ):

                direction_lock = None

        # ====================================================
        # PREVIOUS CANDLE
        # ====================================================

        prev = df.iloc[i - 1]

        # ====================================================
        # BUY SETUP
        # ====================================================

        bull_trend = (

            row["Close"] >
            row["VWAP"]

            and

            row["EMA9"] >
            row["EMA15"]

            and

            row["EMA9_Slope"] > 0
        )

        buy_pullback = (

            prev["Low"] <= max(
                prev["EMA9"],
                prev["EMA15"]
            )

            and

            prev["Close"] >=
            prev["EMA15"]
        )

        buy_confirmation = (

            row["Close"] >
            row["Open"]

            and

            row["Close"] >
            row["EMA9"]
        )

        buy_breakout = (

            row["Close"] >
            prev["High"]
        )

        if (

            direction_lock != "BUY"

            and

            bull_trend

            and

            buy_pullback

            and

            buy_confirmation

            and

            buy_breakout
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

            if risk > 0.01:

                target = (
                    entry +
                    risk * 2
                )

                position = {

                    "side":
                    "BUY",

                    "entry":
                    entry,

                    "sl":
                    sl,

                    "target":
                    target,

                    "entry_time":
                    next_row.name
                }

                direction_lock = "BUY"

                continue

        # ====================================================
        # SELL SETUP
        # ====================================================

        bear_trend = (

            row["Close"] <
            row["VWAP"]

            and

            row["EMA9"] <
            row["EMA15"]

            and

            row["EMA9_Slope"] < 0
        )

        sell_pullback = (

            prev["High"] >= min(
                prev["EMA9"],
                prev["EMA15"]
            )

            and

            prev["Close"] <=
            prev["EMA15"]
        )

        sell_confirmation = (

            row["Close"] <
            row["Open"]

            and

            row["Close"] <
            row["EMA9"]
        )

        sell_breakout = (

            row["Close"] <
            prev["Low"]
        )

        if (

            direction_lock != "SELL"

            and

            bear_trend

            and

            sell_pullback

            and

            sell_confirmation

            and

            sell_breakout
        ):

            # Entry on next candle open

            entry = float(
                next_row["Open"]
            )

            sl = max(
                float(prev["High"]),
                float(row["High"])
            )

            risk = sl - entry

            if risk > 0.01:

                target = (
                    entry -
                    risk * 2
                )

                position = {

                    "side":
                    "SELL",

                    "entry":
                    entry,

                    "sl":
                    sl,

                    "target":
                    target,

                    "entry_time":
                    next_row.name
                }

                direction_lock = "SELL"

    return trades


# ============================================================
# CHART FUNCTION
# ============================================================

def create_chart(
    df,
    symbol_name
):

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

        title=(
            f"{symbol_name} Index Chart"
        ),

        height=620,

        template="plotly_dark",

        paper_bgcolor="#111722",

        plot_bgcolor="#111722",

        xaxis_rangeslider_visible=False,

        margin=dict(
            l=20,
            r=20,
            t=50,
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

st.title(
    "📈 Personal 9-15 EMA Scalping Scanner"
)

st.caption(
    "Index Only Strategy | "
    "9 EMA + 15 EMA | "
    "VWAP = (O+H+L+C)/4 | "
    "VWAP Direction Filter | "
    "EMA Pullback | "
    "Confirmation Candle | "
    "Breakout Entry | "
    "One Move = One Trade | "
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

        st.cache_data.clear()

        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

symbol = INDEX_MAP[
    selected_index
]

interval, period = TIMEFRAME_MAP[
    selected_tf
]

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
# ADD INDICATORS
# ============================================================

df = add_indicators(df)

signal = get_signal(df)

last = df.iloc[-1]


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.header(
    "🎯 Current Signal"
)

signal_col, status_col = st.columns(2)


with signal_col:

    signal_name = signal["signal"]

    if signal_name == "BUY":

        signal_color = "#38d996"

    elif signal_name == "SELL":

        signal_color = "#ff6464"

    else:

        signal_color = "#ffd166"

    st.markdown(

        f"""
        <div class="metric-box">

            <div class="metric-label">
                Signal
            </div>

            <div class="metric-value"
                 style="color:{signal_color};">

                {signal_name}

            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


with status_col:

    st.markdown(

        f"""
        <div class="metric-box">

            <div class="metric-label">
                Trade Status
            </div>

            <div class="metric-value">

                {signal["status"]}

            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# CURRENT INDICATOR VALUES
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Price",
    f"{last['Close']:.2f}"
)

metric2.metric(
    "EMA 9",
    f"{last['EMA9']:.2f}"
)

metric3.metric(
    "EMA 15",
    f"{last['EMA15']:.2f}"
)

metric4.metric(
    "VWAP",
    f"{last['VWAP']:.2f}"
)


# ============================================================
# SIGNAL CONDITION
# ============================================================

st.subheader(
    "Signal Condition"
)

st.write(
    signal["reason"]
)

if signal["vwap_near"]:

    st.success(
        "VWAP is relatively near the current price."
    )

else:

    st.info(
        "VWAP is farther from current price. "
        "Setup quality should be judged carefully."
    )


# ============================================================
# TRADE LEVELS
# ============================================================

if signal["signal"] in [
    "BUY",
    "SELL"
]:

    st.header(
        "📍 Trade Levels"
    )

    level1, level2, level3 = st.columns(3)

    level1.metric(
        "Entry",
        f"{signal['entry']:.2f}"
    )

    level2.metric(
        "Stop Loss",
        f"{signal['sl']:.2f}"
    )

    level3.metric(
        "Target (1:2)",
        f"{signal['target']:.2f}"
    )

    st.markdown(

        f"""
        <div class="info-box">

        Active setup:
        <b>{signal["signal"]}</b>

        | Entry:
        {signal["entry"]:.2f}

        | SL:
        {signal["sl"]:.2f}

        | Target:
        {signal["target"]:.2f}

        </div>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# STRATEGY RULES
# ============================================================

with st.expander(
    "📘 Strategy Rules"
):

    left, right = st.columns(2)

    with left:

        st.subheader("BUY")

        st.markdown("""

1. Price **VWAP के ऊपर**
2. **EMA 9 > EMA 15**
3. EMA 9 का slope ऊपर
4. Market EMA zone तक pullback करे
5. Bullish confirmation candle बने
6. Previous candle high के ऊपर breakout मिले
7. VWAP पास में हो तो setup बेहतर
8. एक move में केवल एक trade
9. Target = **1:2 Risk : Reward**

        """)

    with right:

        st.subheader("SELL")

        st.markdown("""

1. Price **VWAP के नीचे**
2. **EMA 9 < EMA 15**
3. EMA 9 का slope नीचे
4. Market EMA zone तक pullback करे
5. Bearish confirmation candle बने
6. Previous candle low के नीचे breakout मिले
7. VWAP पास में हो तो setup बेहतर
8. एक move में केवल एक trade
9. Target = **1:2 Risk : Reward**

        """)


# ============================================================
# INDEX CHART
# ============================================================

st.header(
    "📊 Index Chart"
)

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

st.header(
    "📉 Historical Backtest"
)

trades = run_backtest(df)


if len(trades) == 0:

    st.warning(
        "No completed trades found in the available data."
    )

else:

    trades_df = pd.DataFrame(trades)

    total_trades = len(
        trades_df
    )

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
        wins /
        total_trades *
        100
        if total_trades > 0
        else 0
    )

    net_points = (
        trades_df["Points"]
        .sum()
    )

    avg_r = (
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

    row1 = st.columns(4)

    row1[0].metric(
        "Total Trades",
        total_trades
    )

    row1[1].metric(
        "Wins",
        wins
    )

    row1[2].metric(
        "Losses",
        losses
    )

    row1[3].metric(
        "Win Rate",
        f"{win_rate:.1f}%"
    )

    row2 = st.columns(3)

    row2[0].metric(
        "Net Points",
        f"{net_points:.2f}"
    )

    row2[1].metric(
        "Average R",
        f"{avg_r:.2f}R"
    )

    row2[2].metric(
        "Best Trade",
        f"{best_trade:.2f}R"
    )

    # --------------------------------------------------------
    # RECENT TRADES
    # --------------------------------------------------------

    st.subheader(
        "Recent Backtest Trades"
    )

    display_df = trades_df.copy()

    display_df = display_df.iloc[
        ::-1
    ]

    display_df = display_df.head(
        30
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Strategy: "
    "VWAP = Direction | "
    "EMA 9/15 = Trend | "
    "Pullback + Confirmation = Entry | "
    "One Move = One Trade | "
    "Fixed 1:2 Risk:Reward"
)

st.caption(
    f"Last data candle: {df.index[-1]}"
)
