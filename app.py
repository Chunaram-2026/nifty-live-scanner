import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EMA 9/15 VWAP Scalping Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #080d16;
    color: #d9e2ef;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.metric-box {
    background: #172231;
    border: 1px solid #26384b;
    padding: 12px;
    border-radius: 9px;
    min-height: 82px;
    margin-bottom: 8px;
}

.metric-label {
    color: #9aa7b7;
    font-size: 11px;
}

.metric-value {
    color: #dce5ef;
    font-size: 18px;
    font-weight: 700;
    margin-top: 6px;
}

.signal-buy {
    color: #48d597;
    font-weight: 800;
}

.signal-sell {
    color: #ff6b81;
    font-weight: 800;
}

.signal-wait {
    color: #f3c969;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_CONFIG = {

    "NIFTY 50": {
        "symbol": "^NSEI"
    },

    "BANK NIFTY": {
        "symbol": "^NSEBANK"
    },

    "SENSEX": {
        "symbol": "^BSESN"
    }

}


TIMEFRAME_CONFIG = {

    "1m": {
        "interval": "1m",
        "period": "5d",
        "resample": None
    },

    "2m": {
        "interval": "2m",
        "period": "5d",
        "resample": None
    },

    "3m": {
        "interval": "1m",
        "period": "5d",
        "resample": "3min"
    },

    "5m": {
        "interval": "5m",
        "period": "5d",
        "resample": None
    },

    "15m": {
        "interval": "15m",
        "period": "5d",
        "resample": None
    },

    "1h": {
        "interval": "60m",
        "period": "1mo",
        "resample": None
    }

}


# ============================================================
# STRATEGY SETTINGS
# ============================================================

EMA_FAST = 9
EMA_SLOW = 15

# VWAP के पास कितना acceptable है
VWAP_NEAR_PERCENT = 0.20

# Pullback candle EMA9 के पास
EMA_PULLBACK_PERCENT = 0.12

# पिछले कितने candles में pullback valid
PULLBACK_LOOKBACK = 5

# Minimum risk
MIN_RISK_POINTS = 8

# Maximum risk
MAX_RISK_POINTS = 25

# Minimum RR
MIN_RR = 2

# Backtest
BACKTEST_MAX_HOLD_CANDLES = 80

# Same move में तुरंत re-entry रोकने के लिए
COOLDOWN_CANDLES = 5

# EMA structure exit confirmation
EXIT_CONFIRM_CANDLES = 2


# ============================================================
# SESSION STATE
# ============================================================

if "running_trade" not in st.session_state:
    st.session_state.running_trade = None

if "closed_trades" not in st.session_state:
    st.session_state.closed_trades = []

if "last_trade_signal_time" not in st.session_state:
    st.session_state.last_trade_signal_time = None


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_number(value, digits=2):

    if value is None:
        return "-"

    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "-"


def metric_box(label, value, value_class=""):

    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value {value_class}">
            {value}
        </div>
    </div>
    """


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=30)
def get_market_data(symbol, timeframe):

    config = TIMEFRAME_CONFIG[timeframe]

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period=config["period"],
            interval=config["interval"],
            auto_adjust=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):

            if df.index.tz is not None:

                df.index = df.index.tz_localize(None)

        if config["resample"] is not None:

            df = (

                df.resample(
                    config["resample"]
                )

                .agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                })

                .dropna()

            )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    data = df.copy()

    # --------------------------------------------------------
    # EMA 9
    # --------------------------------------------------------

    data["EMA9"] = (

        data["Close"]

        .ewm(
            span=EMA_FAST,
            adjust=False
        )

        .mean()

    )

    # --------------------------------------------------------
    # EMA 15
    # --------------------------------------------------------

    data["EMA15"] = (

        data["Close"]

        .ewm(
            span=EMA_SLOW,
            adjust=False
        )

        .mean()

    )

    # --------------------------------------------------------
    # PRICE FOR VWAP
    #
    # (OPEN + HIGH + LOW + CLOSE) / 4
    # --------------------------------------------------------

    data["VWAPPrice"] = (

        data["Open"]

        + data["High"]

        + data["Low"]

        + data["Close"]

    ) / 4

    # --------------------------------------------------------
    # DAILY VWAP
    # --------------------------------------------------------

    data["TradeDate"] = pd.to_datetime(
        data.index
    ).date

    volume = (

        data["Volume"]

        .replace(0, np.nan)

        .fillna(1)

    )

    cumulative_price_volume = (

        data["VWAPPrice"]

        * volume

    ).groupby(
        data["TradeDate"]
    ).cumsum()

    cumulative_volume = (

        volume

        .groupby(
            data["TradeDate"]
        )

        .cumsum()

    )

    data["VWAP"] = (

        cumulative_price_volume

        / cumulative_volume

    )

    data["VWAP"] = (

        data["VWAP"]

        .fillna(
            data["VWAPPrice"]
        )

    )

    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    data["Body"] = (

        data["Close"]

        - data["Open"]

    ).abs()

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    data["Range"] = (

        data["High"]

        - data["Low"]

    )

    # --------------------------------------------------------
    # UPPER WICK
    # --------------------------------------------------------

    data["UpperWick"] = (

        data["High"]

        - data[
            ["Open", "Close"]
        ].max(axis=1)

    )

    # --------------------------------------------------------
    # LOWER WICK
    # --------------------------------------------------------

    data["LowerWick"] = (

        data[
            ["Open", "Close"]
        ].min(axis=1)

        - data["Low"]

    )

    return data


# ============================================================
# MARKET DIRECTION
# ============================================================

def get_market_direction(last):

    price = safe_float(
        last["Close"]
    )

    vwap = safe_float(
        last["VWAP"]
    )

    if price is None or vwap is None:

        return "NEUTRAL"

    if price > vwap:

        return "BULLISH"

    if price < vwap:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# VWAP DISTANCE
# ============================================================

def is_near_vwap(price, vwap):

    if price is None or vwap is None:
        return False

    if vwap == 0:
        return False

    distance_percent = (

        abs(price - vwap)

        / vwap

        * 100

    )

    return distance_percent <= VWAP_NEAR_PERCENT


# ============================================================
# EMA PULLBACK
# ============================================================

def is_ema9_pullback(data, position, direction):

    start = max(
        0,
        position - PULLBACK_LOOKBACK
    )

    recent = data.iloc[
        start:position
    ]

    if recent.empty:
        return False

    for _, row in recent.iterrows():

        low = safe_float(
            row["Low"]
        )

        high = safe_float(
            row["High"]
        )

        ema9 = safe_float(
            row["EMA9"]
        )

        if ema9 is None:
            continue

        tolerance = (

            ema9

            * EMA_PULLBACK_PERCENT

            / 100

        )

        if direction == "BUY":

            if low <= ema9 + tolerance:

                return True

        elif direction == "SELL":

            if high >= ema9 - tolerance:

                return True

    return False


# ============================================================
# EMA TREND
# ============================================================

def get_ema_direction(last):

    price = safe_float(
        last["Close"]
    )

    ema9 = safe_float(
        last["EMA9"]
    )

    ema15 = safe_float(
        last["EMA15"]
    )

    if (

        price > ema9

        and

        ema9 > ema15

    ):

        return "BUY"

    if (

        price < ema9

        and

        ema9 < ema15

    ):

        return "SELL"

    return "NEUTRAL"


# ============================================================
# CONFIRMATION CANDLE
# ============================================================

def is_confirmation_candle(last, direction):

    close = safe_float(
        last["Close"]
    )

    open_price = safe_float(
        last["Open"]
    )

    high = safe_float(
        last["High"]
    )

    low = safe_float(
        last["Low"]
    )

    body = abs(
        close - open_price
    )

    candle_range = max(
        high - low,
        0.0001
    )

    body_percent = (

        body

        / candle_range

        * 100

    )

    # Very weak candle avoid
    if body_percent < 30:

        return False

    if direction == "BUY":

        return close > open_price

    if direction == "SELL":

        return close < open_price

    return False


# ============================================================
# BREAK OF PREVIOUS CANDLE
# ============================================================

def is_momentum_confirmation(
    last,
    previous,
    direction
):

    close = safe_float(
        last["Close"]
    )

    previous_high = safe_float(
        previous["High"]
    )

    previous_low = safe_float(
        previous["Low"]
    )

    if direction == "BUY":

        return close > previous_high

    if direction == "SELL":

        return close < previous_low

    return False


# ============================================================
# SIGNAL CALCULATION
# ============================================================

def calculate_signal_at(data, position):

    if position < 25:

        return None

    last = data.iloc[position]

    previous = data.iloc[
        position - 1
    ]

    price = safe_float(
        last["Close"]
    )

    ema9 = safe_float(
        last["EMA9"]
    )

    ema15 = safe_float(
        last["EMA15"]
    )

    vwap = safe_float(
        last["VWAP"]
    )

    market_direction = (
        get_market_direction(last)
    )

    ema_direction = (
        get_ema_direction(last)
    )

    direction = "WAIT"

    reason = []

    # --------------------------------------------------------
    # BUY CONDITIONS
    # --------------------------------------------------------

    if (

        market_direction == "BULLISH"

        and

        ema_direction == "BUY"

    ):

        pullback = is_ema9_pullback(
            data,
            position,
            "BUY"
        )

        confirmation = (
            is_confirmation_candle(
                last,
                "BUY"
            )
        )

        momentum = (
            is_momentum_confirmation(
                last,
                previous,
                "BUY"
            )
        )

        near_vwap = is_near_vwap(
            price,
            vwap
        )

        if (

            pullback

            and

            confirmation

            and

            momentum

        ):

            direction = "BUY"

            reason.append(
                "Price above VWAP"
            )

            reason.append(
                "EMA 9 > EMA 15"
            )

            reason.append(
                "EMA pullback"
            )

            reason.append(
                "Bullish confirmation"
            )

            if near_vwap:

                reason.append(
                    "Near VWAP"
                )

    # --------------------------------------------------------
    # SELL CONDITIONS
    # --------------------------------------------------------

    elif (

        market_direction == "BEARISH"

        and

        ema_direction == "SELL"

    ):

        pullback = is_ema9_pullback(
            data,
            position,
            "SELL"
        )

        confirmation = (
            is_confirmation_candle(
                last,
                "SELL"
            )
        )

        momentum = (
            is_momentum_confirmation(
                last,
                previous,
                "SELL"
            )
        )

        near_vwap = is_near_vwap(
            price,
            vwap
        )

        if (

            pullback

            and

            confirmation

            and

            momentum

        ):

            direction = "SELL"

            reason.append(
                "Price below VWAP"
            )

            reason.append(
                "EMA 9 < EMA 15"
            )

            reason.append(
                "EMA pullback"
            )

            reason.append(
                "Bearish confirmation"
            )

            if near_vwap:

                reason.append(
                    "Near VWAP"
                )

    if direction == "WAIT":

        reason.append(
            "Waiting for EMA pullback + confirmation"
        )

    return {

        "signal": direction,

        "price": price,

        "ema9": ema9,

        "ema15": ema15,

        "vwap": vwap,

        "market_direction": (
            market_direction
        ),

        "reason": (
            " | ".join(reason)
        ),

        "time": data.index[position]

    }


# ============================================================
# CURRENT SIGNAL
# ============================================================

def get_signal(data):

    if len(data) < 26:
        return None

    return calculate_signal_at(
        data,
        len(data) - 1
    )


# ============================================================
# CALCULATE RISK
# ============================================================

def calculate_risk(candle):

    candle_range = (

        safe_float(candle["High"])

        - safe_float(candle["Low"])

    )

    risk = candle_range * 0.75

    risk = max(
        MIN_RISK_POINTS,
        risk
    )

    risk = min(
        MAX_RISK_POINTS,
        risk
    )

    return risk


# ============================================================
# CREATE TRADE LEVELS
# ============================================================

def create_trade_levels(
    signal,
    entry,
    candle
):

    risk = calculate_risk(
        candle
    )

    if signal == "BUY":

        sl = entry - risk

        minimum_target = (

            entry

            + risk * MIN_RR

        )

    else:

        sl = entry + risk

        minimum_target = (

            entry

            - risk * MIN_RR

        )

    return {

        "entry": entry,

        "sl": sl,

        "minimum_target":
            minimum_target,

        "risk": risk

    }


# ============================================================
# RUNNER EXIT CHECK
# ============================================================

def runner_exit(
    data,
    position,
    signal
):

    if position < EXIT_CONFIRM_CANDLES:

        return False

    recent = data.iloc[
        position - EXIT_CONFIRM_CANDLES + 1:
        position + 1
    ]

    if len(recent) < EXIT_CONFIRM_CANDLES:

        return False

    if signal == "BUY":

        conditions = (

            recent["Close"]

            < recent["EMA9"]

        )

        return conditions.all()

    if signal == "SELL":

        conditions = (

            recent["Close"]

            > recent["EMA9"]

        )

        return conditions.all()

    return False


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

@st.cache_data(ttl=30)
def run_historical_backtest(data):

    trades = []

    if len(data) < 50:

        return pd.DataFrame()

    position = 25

    last_exit_position = -999

    while position < len(data) - 2:

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if (

            position

            <=

            last_exit_position
            + COOLDOWN_CANDLES

        ):

            position += 1

            continue

        signal_data = (
            calculate_signal_at(
                data,
                position
            )
        )

        if signal_data is None:

            position += 1

            continue

        signal = signal_data[
            "signal"
        ]

        if signal not in [
            "BUY",
            "SELL"
        ]:

            position += 1

            continue

        # ----------------------------------------------------
        # ENTRY NEXT CANDLE
        # ----------------------------------------------------

        entry_position = (
            position + 1
        )

        entry_candle = data.iloc[
            entry_position
        ]

        entry = safe_float(
            entry_candle["Open"]
        )

        levels = create_trade_levels(

            signal,

            entry,

            data.iloc[position]

        )

        sl = levels["sl"]

        minimum_target = (
            levels["minimum_target"]
        )

        risk = levels["risk"]

        target_2r_hit = False

        exit_found = False

        highest_price = entry

        lowest_price = entry

        end_position = min(

            entry_position
            + BACKTEST_MAX_HOLD_CANDLES,

            len(data) - 1

        )

        future_position = entry_position

        for future_position in range(

            entry_position,

            end_position + 1

        ):

            future = data.iloc[
                future_position
            ]

            high = safe_float(
                future["High"]
            )

            low = safe_float(
                future["Low"]
            )

            close = safe_float(
                future["Close"]
            )

            # ------------------------------------------------
            # BUY TRADE
            # ------------------------------------------------

            if signal == "BUY":

                highest_price = max(
                    highest_price,
                    high
                )

                # Hard SL

                if low <= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_found = True

                    break

                # Minimum 1:2 reached

                if high >= minimum_target:

                    target_2r_hit = True

                # After 1:2 runner logic

                if target_2r_hit:

                    if runner_exit(
                        data,
                        future_position,
                        "BUY"
                    ):

                        exit_price = close

                        exit_reason = (
                            "RUNNER EXIT"
                        )

                        exit_found = True

                        break

            # ------------------------------------------------
            # SELL TRADE
            # ------------------------------------------------

            else:

                lowest_price = min(
                    lowest_price,
                    low
                )

                # Hard SL

                if high >= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_found = True

                    break

                # Minimum 1:2 reached

                if low <= minimum_target:

                    target_2r_hit = True

                # After 1:2 runner logic

                if target_2r_hit:

                    if runner_exit(
                        data,
                        future_position,
                        "SELL"
                    ):

                        exit_price = close

                        exit_reason = (
                            "RUNNER EXIT"
                        )

                        exit_found = True

                        break

        # ----------------------------------------------------
        # TIME EXIT
        # ----------------------------------------------------

        if not exit_found:

            final_candle = data.iloc[
                end_position
            ]

            exit_price = safe_float(
                final_candle["Close"]
            )

            exit_reason = (
                "TIME EXIT"
            )

            future_position = end_position

        # ----------------------------------------------------
        # PROFIT POINTS
        # ----------------------------------------------------

        if signal == "BUY":

            points = (
                exit_price - entry
            )

        else:

            points = (
                entry - exit_price
            )

        # ----------------------------------------------------
        # R MULTIPLE
        # ----------------------------------------------------

        r_multiple = (

            points / risk

            if risk > 0

            else 0

        )

        trades.append({

            "Signal":
                signal,

            "Entry Time":
                str(
                    data.index[
                        entry_position
                    ]
                ),

            "Exit Time":
                str(
                    data.index[
                        future_position
                    ]
                ),

            "Entry":
                round(entry, 2),

            "SL":
                round(sl, 2),

            "Min Target 1:2":
                round(
                    minimum_target,
                    2
                ),

            "Exit":
                round(
                    exit_price,
                    2
                ),

            "Points":
                round(
                    points,
                    2
                ),

            "R Multiple":
                round(
                    r_multiple,
                    2
                ),

            "2R Reached":
                "YES"

                if target_2r_hit

                else "NO",

            "Exit Reason":
                exit_reason

        })

        # ----------------------------------------------------
        # ONE MOVE = ONE TRADE
        # ----------------------------------------------------

        last_exit_position = (
            future_position
        )

        position = (
            future_position + 1
        )

    if not trades:

        return pd.DataFrame()

    return pd.DataFrame(
        trades
    )


# ============================================================
# BACKTEST STATS
# ============================================================

def get_backtest_stats(backtest_df):

    if (
        backtest_df is None
        or backtest_df.empty
    ):

        return {

            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0,
            "avg_r": 0,
            "best_r": 0

        }

    total = len(
        backtest_df
    )

    wins = len(

        backtest_df[
            backtest_df[
                "Points"
            ] > 0
        ]

    )

    losses = len(

        backtest_df[
            backtest_df[
                "Points"
            ] < 0
        ]

    )

    win_rate = (

        wins / total * 100

        if total > 0

        else 0

    )

    net_points = (

        backtest_df[
            "Points"
        ].sum()

    )

    avg_r = (

        backtest_df[
            "R Multiple"
        ].mean()

    )

    best_r = (

        backtest_df[
            "R Multiple"
        ].max()

    )

    return {

        "total": total,

        "wins": wins,

        "losses": losses,

        "win_rate": win_rate,

        "net_points": net_points,

        "avg_r": avg_r,

        "best_r": best_r

    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 EMA 9/15 + VWAP Scalping Scanner"
)

st.caption(
    "VWAP Direction → EMA Trend → Pullback → "
    "Confirmation → One Move One Trade → "
    "Minimum 1:2 + Runner"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns(
    [1, 1, 0.6]
)

with col1:

    index_name = st.selectbox(

        "Select Index",

        list(
            INDEX_CONFIG.keys()
        ),

        index=0

    )

with col2:

    timeframe = st.selectbox(

        "Select Timeframe",

        list(
            TIMEFRAME_CONFIG.keys()
        ),

        index=2

    )

with col3:

    st.write("")

    if st.button(
        "🔄 Refresh"
    ):

        get_market_data.clear()

        run_historical_backtest.clear()

        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

symbol = (
    INDEX_CONFIG[
        index_name
    ]["symbol"]
)

df = get_market_data(
    symbol,
    timeframe
)


if df.empty:

    st.error(
        "Market data load नहीं हो पाया।"
    )

    st.stop()


# ============================================================
# CALCULATE INDICATORS
# ============================================================

data = calculate_indicators(
    df
)

signal_data = get_signal(
    data
)


if signal_data is None:

    st.warning(
        "Signal के लिए पर्याप्त data नहीं है।"
    )

    st.stop()


signal = (
    signal_data["signal"]
)

price = (
    signal_data["price"]
)

ema9 = (
    signal_data["ema9"]
)

ema15 = (
    signal_data["ema15"]
)

vwap = (
    signal_data["vwap"]
)

market_direction = (
    signal_data[
        "market_direction"
    ]
)


# ============================================================
# SIGNAL COLOR
# ============================================================

if signal == "BUY":

    signal_class = (
        "signal-buy"
    )

elif signal == "SELL":

    signal_class = (
        "signal-sell"
    )

else:

    signal_class = (
        "signal-wait"
    )


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.markdown(
    "## 🎯 Current Market Signal"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(

        metric_box(

            "Signal",

            signal,

            signal_class

        ),

        unsafe_allow_html=True

    )

with c2:

    direction_class = (

        "signal-buy"

        if market_direction == "BULLISH"

        else "signal-sell"

        if market_direction == "BEARISH"

        else "signal-wait"

    )

    st.markdown(

        metric_box(

            "VWAP Direction",

            market_direction,

            direction_class

        ),

        unsafe_allow_html=True

    )

with c3:

    st.markdown(

        metric_box(

            "Current Price",

            format_number(price)

        ),

        unsafe_allow_html=True

    )


c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(

        metric_box(

            "EMA 9",

            format_number(ema9)

        ),

        unsafe_allow_html=True

    )

with c2:

    st.markdown(

        metric_box(

            "EMA 15",

            format_number(ema15)

        ),

        unsafe_allow_html=True

    )

with c3:

    st.markdown(

        metric_box(

            "VWAP",

            format_number(vwap)

        ),

        unsafe_allow_html=True

    )


# ============================================================
# TRADE LEVELS
# ============================================================

levels = None

if signal in [
    "BUY",
    "SELL"
]:

    levels = create_trade_levels(

        signal,

        price,

        data.iloc[-1]

    )

    st.markdown(
        "## 📌 Trade Plan"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Entry",
            format_number(
                levels["entry"]
            )
        )

    with c2:

        st.metric(
            "Stop Loss",
            format_number(
                levels["sl"]
            )
        )

    with c3:

        st.metric(
            "Minimum Target",
            format_number(
                levels[
                    "minimum_target"
                ]
            )
        )

    with c4:

        st.metric(
            "Risk : Reward",
            "1 : 2 Minimum"
        )

    st.info(
        "1:2 केवल minimum target है। "
        "Trend strong रहने पर EMA 9 trailing logic से "
        "trade 1:3, 1:4, 1:6, 1:7 या अधिक चल सकता है।"
    )


# ============================================================
# SIGNAL CONDITION
# ============================================================

st.markdown(
    "### Signal Condition"
)

st.write(
    signal_data["reason"]
)


# ============================================================
# CHART
# ============================================================

st.markdown(
    "## 📊 Index Chart"
)

chart_data = data.tail(
    200
).copy()

fig = go.Figure()


# ============================================================
# CANDLESTICK
# ============================================================

fig.add_trace(

    go.Candlestick(

        x=chart_data.index,

        open=chart_data["Open"],

        high=chart_data["High"],

        low=chart_data["Low"],

        close=chart_data["Close"],

        name="Price"

    )

)


# ============================================================
# EMA 9
# ============================================================

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA9"],

        mode="lines",

        name="EMA 9",

        line=dict(
            width=2
        )

    )

)


# ============================================================
# EMA 15
# ============================================================

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA15"],

        mode="lines",

        name="EMA 15",

        line=dict(
            width=2
        )

    )

)


# ============================================================
# VWAP
# ============================================================

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["VWAP"],

        mode="lines",

        name="VWAP",

        line=dict(
            width=2
        )

    )

)


# ============================================================
# CURRENT TRADE LEVELS
# ============================================================

if levels is not None:

    fig.add_hline(

        y=levels["entry"],

        line_dash="dot",

        annotation_text=(
            f"{signal} ENTRY"
        )

    )

    fig.add_hline(

        y=levels["sl"],

        line_dash="dash",

        annotation_text="SL"

    )

    fig.add_hline(

        y=levels[
            "minimum_target"
        ],

        line_dash="dash",

        annotation_text="MINIMUM 1:2"

    )

    marker_symbol = (

        "triangle-up"

        if signal == "BUY"

        else "triangle-down"

    )

    fig.add_trace(

        go.Scatter(

            x=[
                chart_data.index[-1]
            ],

            y=[
                price
            ],

            mode="markers+text",

            marker=dict(

                size=14,

                symbol=marker_symbol

            ),

            text=[
                signal
            ],

            textposition="top center",

            name=signal

        )

    )


fig.update_layout(

    height=650,

    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10
    ),

    template="plotly_dark",

    xaxis_rangeslider_visible=False,

    legend=dict(
        orientation="h"
    )

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# STRATEGY RULES
# ============================================================

st.markdown(
    "## 📋 Strategy Rules"
)

st.markdown("""

### BUY

1. Price **VWAP के ऊपर**
2. EMA 9 **EMA 15 के ऊपर**
3. Market EMA 9 / EMA zone तक pullback करे
4. Bullish confirmation candle बने
5. Previous candle high के ऊपर confirmation मिले
6. VWAP के पास setup को प्राथमिकता
7. एक move में केवल एक trade
8. Minimum target 1:2
9. उसके बाद EMA 9 के नीचे लगातार 2 candles मिलने पर runner exit

### SELL

1. Price **VWAP के नीचे**
2. EMA 9 **EMA 15 के नीचे**
3. Market EMA 9 / EMA zone तक pullback करे
4. Bearish confirmation candle बने
5. Previous candle low के नीचे confirmation मिले
6. VWAP के पास setup को प्राथमिकता
7. एक move में केवल एक trade
8. Minimum target 1:2
9. उसके बाद EMA 9 के ऊपर लगातार 2 candles मिलने पर runner exit

""")


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

st.markdown(
    "## 📉 Historical Backtest"
)

backtest_df = (
    run_historical_backtest(
        data
    )
)

stats = get_backtest_stats(
    backtest_df
)


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Total Trades",
        stats["total"]
    )

with c2:

    st.metric(
        "Wins",
        stats["wins"]
    )

with c3:

    st.metric(
        "Losses",
        stats["losses"]
    )

with c4:

    st.metric(
        "Win Rate",
        f"{stats['win_rate']:.1f}%"
    )


c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Net Points",
        f"{stats['net_points']:.2f}"
    )

with c2:

    st.metric(
        "Average R",
        f"{stats['avg_r']:.2f}R"
    )

with c3:

    st.metric(
        "Best Trade",
        f"{stats['best_r']:.2f}R"
    )


# ============================================================
# BACKTEST TABLE
# ============================================================

st.markdown(
    "### Recent Backtest Trades"
)

if backtest_df.empty:

    st.info(
        "Historical trades नहीं मिले।"
    )

else:

    st.dataframe(

        backtest_df.iloc[
            ::-1
        ].head(20),

        use_container_width=True

    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "VWAP = Direction | EMA 9/15 = Trend | "
    "Pullback + Confirmation = Entry | "
    "One Move = One Trade | "
    "Minimum 1:2 + Runner"
)
