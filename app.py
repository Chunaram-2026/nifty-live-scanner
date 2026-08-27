import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
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
    background-color: #080d16;
    color: #d9e2ef;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.signal-box {
    background: #172231;
    border: 1px solid #26384b;
    border-radius: 10px;
    padding: 16px;
    min-height: 100px;
}

.signal-label {
    color: #9aa7b7;
    font-size: 12px;
    margin-bottom: 8px;
}

.signal-value {
    color: #dce5ef;
    font-size: 18px;
    font-weight: 800;
}

.call-text {
    color: #48d597;
}

.put-text {
    color: #ff6b81;
}

.wait-text {
    color: #f3c969;
}

</style>
""", unsafe_allow_html=True)


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
# INDEX CONFIG
# ============================================================

INDEX_CONFIG = {

    "NIFTY 50": {
        "yahoo": "^NSEI",
        "nse": "NIFTY",
        "strike_step": 50
    },

    "BANK NIFTY": {
        "yahoo": "^NSEBANK",
        "nse": "BANKNIFTY",
        "strike_step": 100
    },

    "SENSEX": {
        "yahoo": "^BSESN",
        "nse": None,
        "strike_step": 100
    }

}


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

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
    }

}


# ============================================================
# STRATEGY SETTINGS
# ============================================================

EMA_FAST = 9
EMA_SLOW = 15

RISK_REWARD = 2.0

BACKTEST_MAX_HOLD_CANDLES = 30

SETUP_LOOKBACK = 5

# EMA separation filter
MIN_EMA_SEPARATION_ATR = 0.08

# EMA slope filter
MIN_EMA_SLOPE_ATR = 0.04

# Candle body strength
MIN_STRONG_BODY_RATIO = 0.55

# Maximum confirmation wick ratio
MAX_WICK_BODY_RATIO = 1.5

# Pullback should touch/enter EMA zone
EMA_TOUCH_TOLERANCE_ATR = 0.10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "")

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_number(value, digits=2):

    if value is None:
        return "-"

    try:
        return f"{float(value):.{digits}f}"

    except Exception:
        return "-"


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

        # Convert to India timezone
        if isinstance(df.index, pd.DatetimeIndex):

            if df.index.tz is not None:

                try:
                    df.index = df.index.tz_convert(
                        "Asia/Kolkata"
                    )

                except Exception:
                    pass

            try:
                df.index = df.index.tz_localize(None)

            except Exception:
                pass

        # Resample
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
    # CANDLE RANGE
    # --------------------------------------------------------

    data["Range"] = (
        data["High"] -
        data["Low"]
    )

    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    data["Body"] = (
        data["Close"] -
        data["Open"]
    ).abs()

    # --------------------------------------------------------
    # BODY RATIO
    # --------------------------------------------------------

    data["BodyRatio"] = (
        data["Body"] /
        data["Range"].replace(0, np.nan)
    ).fillna(0)

    # --------------------------------------------------------
    # UPPER WICK
    # --------------------------------------------------------

    data["UpperWick"] = (
        data["High"] -
        data[
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
        -
        data["Low"]
    )

    # --------------------------------------------------------
    # ATR-LIKE VOLATILITY
    # --------------------------------------------------------

    previous_close = (
        data["Close"]
        .shift(1)
    )

    tr1 = (
        data["High"] -
        data["Low"]
    )

    tr2 = (
        data["High"] -
        previous_close
    ).abs()

    tr3 = (
        data["Low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    data["ATR"] = (
        data["ATR"]
        .fillna(
            data["Range"].rolling(5).mean()
        )
        .fillna(
            data["Range"]
        )
    )

    # --------------------------------------------------------
    # EMA SLOPE
    #
    # Literal chart angle cannot be reliably calculated
    # because chart zoom changes visual angle.
    #
    # ATR-normalized slope is used instead.
    # --------------------------------------------------------

    data["EMA9Slope"] = (
        data["EMA9"] -
        data["EMA9"].shift(3)
    )

    data["EMA15Slope"] = (
        data["EMA15"] -
        data["EMA15"].shift(3)
    )

    data["EMASeparation"] = (
        data["EMA9"] -
        data["EMA15"]
    ).abs()

    return data


# ============================================================
# CANDLE HELPERS
# ============================================================

def is_bullish(candle):

    return (
        safe_float(candle["Close"], 0)
        >
        safe_float(candle["Open"], 0)
    )


def is_bearish(candle):

    return (
        safe_float(candle["Close"], 0)
        <
        safe_float(candle["Open"], 0)
    )


def is_bullish_engulfing(previous, current):

    previous_bearish = is_bearish(previous)

    current_bullish = is_bullish(current)

    return (

        previous_bearish

        and

        current_bullish

        and

        safe_float(current["Open"])
        <=
        safe_float(previous["Close"])

        and

        safe_float(current["Close"])
        >=
        safe_float(previous["Open"])

    )


def is_bearish_engulfing(previous, current):

    previous_bullish = is_bullish(previous)

    current_bearish = is_bearish(current)

    return (

        previous_bullish

        and

        current_bearish

        and

        safe_float(current["Open"])
        >=
        safe_float(previous["Close"])

        and

        safe_float(current["Close"])
        <=
        safe_float(previous["Open"])

    )


# ============================================================
# EMA TREND FILTER
# ============================================================

def get_trend_state(data, position):

    if position < 20:
        return None

    candle = data.iloc[position]

    ema9 = safe_float(candle["EMA9"])
    ema15 = safe_float(candle["EMA15"])

    slope9 = safe_float(
        candle["EMA9Slope"],
        0
    )

    slope15 = safe_float(
        candle["EMA15Slope"],
        0
    )

    atr = safe_float(
        candle["ATR"],
        1
    )

    separation = safe_float(
        candle["EMASeparation"],
        0
    )

    if atr <= 0:
        return None

    # --------------------------------------------------------
    # NORMALIZED FILTERS
    # --------------------------------------------------------

    separation_ok = (

        separation >=
        atr * MIN_EMA_SEPARATION_ATR

    )

    bullish_slope = (

        slope9 >
        atr * MIN_EMA_SLOPE_ATR

        and

        slope15 >
        atr * MIN_EMA_SLOPE_ATR

    )

    bearish_slope = (

        slope9 <
        -atr * MIN_EMA_SLOPE_ATR

        and

        slope15 <
        -atr * MIN_EMA_SLOPE_ATR

    )

    bullish = (

        ema9 > ema15

        and

        separation_ok

        and

        bullish_slope

    )

    bearish = (

        ema9 < ema15

        and

        separation_ok

        and

        bearish_slope

    )

    if bullish:

        return "BULLISH"

    if bearish:

        return "BEARISH"

    return None


# ============================================================
# PULLBACK DETECTION
# ============================================================

def bullish_pullback(data, position):

    start = max(
        0,
        position - SETUP_LOOKBACK
    )

    recent = data.iloc[
        start:
        position + 1
    ]

    for _, candle in recent.iterrows():

        low = safe_float(candle["Low"])

        ema9 = safe_float(candle["EMA9"])

        ema15 = safe_float(candle["EMA15"])

        atr = safe_float(
            candle["ATR"],
            0
        )

        tolerance = (
            atr *
            EMA_TOUCH_TOLERANCE_ATR
        )

        zone_high = max(
            ema9,
            ema15
        ) + tolerance

        zone_low = min(
            ema9,
            ema15
        ) - tolerance

        if (
            low <= zone_high

            and

            safe_float(candle["High"])
            >= zone_low
        ):

            return True

    return False


def bearish_pullback(data, position):

    start = max(
        0,
        position - SETUP_LOOKBACK
    )

    recent = data.iloc[
        start:
        position + 1
    ]

    for _, candle in recent.iterrows():

        high = safe_float(candle["High"])

        ema9 = safe_float(candle["EMA9"])

        ema15 = safe_float(candle["EMA15"])

        atr = safe_float(
            candle["ATR"],
            0
        )

        tolerance = (
            atr *
            EMA_TOUCH_TOLERANCE_ATR
        )

        zone_high = max(
            ema9,
            ema15
        ) + tolerance

        zone_low = min(
            ema9,
            ema15
        ) - tolerance

        if (
            high >= zone_low

            and

            safe_float(candle["Low"])
            <= zone_high
        ):

            return True

    return False


# ============================================================
# CONFIRMATION CANDLE
# ============================================================

def bullish_confirmation(data, position):

    if position < 1:
        return (
            False,
            ""
        )

    candle = data.iloc[position]

    previous = data.iloc[
        position - 1
    ]

    body = safe_float(
        candle["Body"],
        0
    )

    body_ratio = safe_float(
        candle["BodyRatio"],
        0
    )

    lower_wick = safe_float(
        candle["LowerWick"],
        0
    )

    upper_wick = safe_float(
        candle["UpperWick"],
        0
    )

    # --------------------------------------------------------
    # 1. BULLISH REJECTION / PIN BAR
    # --------------------------------------------------------

    pin_bar = (

        is_bullish(candle)

        and

        lower_wick >=
        body * 1.20

        and

        upper_wick <=
        max(body * MAX_WICK_BODY_RATIO, 0.01)

    )

    # --------------------------------------------------------
    # 2. STRONG BULLISH BODY
    # --------------------------------------------------------

    strong_body = (

        is_bullish(candle)

        and

        body_ratio >=
        MIN_STRONG_BODY_RATIO

    )

    # --------------------------------------------------------
    # 3. BULLISH ENGULFING
    # --------------------------------------------------------

    engulfing = (
        is_bullish_engulfing(
            previous,
            candle
        )
    )

    if pin_bar:

        return (
            True,
            "Bullish rejection"
        )

    if engulfing:

        return (
            True,
            "Bullish engulfing"
        )

    if strong_body:

        return (
            True,
            "Strong bullish candle"
        )

    return (
        False,
        ""
    )


def bearish_confirmation(data, position):

    if position < 1:
        return (
            False,
            ""
        )

    candle = data.iloc[position]

    previous = data.iloc[
        position - 1
    ]

    body = safe_float(
        candle["Body"],
        0
    )

    body_ratio = safe_float(
        candle["BodyRatio"],
        0
    )

    upper_wick = safe_float(
        candle["UpperWick"],
        0
    )

    lower_wick = safe_float(
        candle["LowerWick"],
        0
    )

    # --------------------------------------------------------
    # 1. BEARISH REJECTION / PIN BAR
    # --------------------------------------------------------

    pin_bar = (

        is_bearish(candle)

        and

        upper_wick >=
        body * 1.20

        and

        lower_wick <=
        max(body * MAX_WICK_BODY_RATIO, 0.01)

    )

    # --------------------------------------------------------
    # 2. STRONG BEARISH BODY
    # --------------------------------------------------------

    strong_body = (

        is_bearish(candle)

        and

        body_ratio >=
        MIN_STRONG_BODY_RATIO

    )

    # --------------------------------------------------------
    # 3. BEARISH ENGULFING
    # --------------------------------------------------------

    engulfing = (
        is_bearish_engulfing(
            previous,
            candle
        )
    )

    if pin_bar:

        return (
            True,
            "Bearish rejection"
        )

    if engulfing:

        return (
            True,
            "Bearish engulfing"
        )

    if strong_body:

        return (
            True,
            "Strong bearish candle"
        )

    return (
        False,
        ""
    )


# ============================================================
# SIGNAL CALCULATION
#
# position = BREAK / ENTRY TRIGGER CANDLE
# confirmation candle = previous candle
# ============================================================

def calculate_signal_at(data, position):

    if position < 25:
        return None

    trigger = data.iloc[position]

    confirmation_position = (
        position - 1
    )

    confirmation = data.iloc[
        confirmation_position
    ]

    trend = get_trend_state(
        data,
        confirmation_position
    )

    trigger_high = safe_float(
        trigger["High"]
    )

    trigger_low = safe_float(
        trigger["Low"]
    )

    confirmation_high = safe_float(
        confirmation["High"]
    )

    confirmation_low = safe_float(
        confirmation["Low"]
    )

    trigger_close = safe_float(
        trigger["Close"]
    )

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if trend == "BULLISH":

        pullback_ok = bullish_pullback(
            data,
            confirmation_position
        )

        confirmation_ok, reason = (
            bullish_confirmation(
                data,
                confirmation_position
            )
        )

        break_ok = (

            trigger_high >
            confirmation_high

            and

            trigger_close >
            confirmation_high

        )

        if (
            pullback_ok

            and

            confirmation_ok

            and

            break_ok
        ):

            return {

                "signal":
                    "CALL",

                "strength":
                    "STRONG",

                "reason":
                    (
                        "Uptrend + EMA separation + "
                        "Pullback + "
                        f"{reason} + "
                        "Confirmation High Break"
                    ),

                "time":
                    data.index[position],

                "confirmation_time":
                    data.index[
                        confirmation_position
                    ],

                "entry_trigger":
                    confirmation_high,

                "stop_loss":
                    confirmation_low

            }

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    if trend == "BEARISH":

        pullback_ok = bearish_pullback(
            data,
            confirmation_position
        )

        confirmation_ok, reason = (
            bearish_confirmation(
                data,
                confirmation_position
            )
        )

        break_ok = (

            trigger_low <
            confirmation_low

            and

            trigger_close <
            confirmation_low

        )

        if (
            pullback_ok

            and

            confirmation_ok

            and

            break_ok
        ):

            return {

                "signal":
                    "PUT",

                "strength":
                    "STRONG",

                "reason":
                    (
                        "Downtrend + EMA separation + "
                        "Pullback + "
                        f"{reason} + "
                        "Confirmation Low Break"
                    ),

                "time":
                    data.index[position],

                "confirmation_time":
                    data.index[
                        confirmation_position
                    ],

                "entry_trigger":
                    confirmation_low,

                "stop_loss":
                    confirmation_high

            }

    return {

        "signal":
            "WAIT",

        "strength":
            "LOW",

        "reason":
            "Setup confirmation pending",

        "time":
            data.index[position],

        "confirmation_time":
            None,

        "entry_trigger":
            None,

        "stop_loss":
            None

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
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    signal_data
):

    if signal_data["signal"] == "WAIT":

        return None

    entry = safe_float(
        signal_data["entry_trigger"]
    )

    stop_loss = safe_float(
        signal_data["stop_loss"]
    )

    risk = abs(
        entry -
        stop_loss
    )

    if risk <= 0:

        return None

    if signal_data["signal"] == "CALL":

        target = (
            entry +
            risk * RISK_REWARD
        )

    else:

        target = (
            entry -
            risk * RISK_REWARD
        )

    return {

        "entry":
            entry,

        "stop_loss":
            stop_loss,

        "target":
            target,

        "risk":
            risk

    }


# ============================================================
# NSE SESSION
# ============================================================

def get_nse_session():

    session = requests.Session()

    session.headers.update({

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            ),

        "Accept":
            (
                "application/json, "
                "text/plain, */*"
            ),

        "Referer":
            "https://www.nseindia.com/"

    })

    try:

        session.get(
            "https://www.nseindia.com",
            timeout=10
        )

    except Exception:
        pass

    return session


# ============================================================
# OPTION CHAIN
# ============================================================

@st.cache_data(ttl=20)
def get_option_chain(index_name):

    nse_symbol = (
        INDEX_CONFIG[index_name]["nse"]
    )

    if not nse_symbol:

        return None

    try:

        session = get_nse_session()

        url = (
            "https://www.nseindia.com/"
            "api/option-chain-indices"
            f"?symbol={nse_symbol}"
        )

        response = session.get(
            url,
            timeout=15
        )

        if response.status_code != 200:

            return None

        return response.json()

    except Exception:

        return None


# ============================================================
# NEAREST EXPIRY
# ============================================================

def get_nearest_expiry(option_chain):

    try:

        expiry_dates = (
            option_chain
            .get("records", {})
            .get("expiryDates", [])
        )

        if not expiry_dates:

            return None

        today = datetime.now().date()

        valid = []

        for expiry in expiry_dates:

            try:

                expiry_date = (
                    datetime.strptime(
                        expiry,
                        "%d-%b-%Y"
                    )
                    .date()
                )

                if expiry_date >= today:

                    valid.append(
                        (
                            expiry_date,
                            expiry
                        )
                    )

            except Exception:
                pass

        if valid:

            valid.sort(
                key=lambda x: x[0]
            )

            return valid[0][1]

        return expiry_dates[0]

    except Exception:

        return None


# ============================================================
# OPTION SELECTION
#
# ATM FIRST
# THEN 1 ITM
# THEN 2 ITM
# ============================================================

def find_option_contract(
    option_chain,
    spot_price,
    option_type,
    index_name
):

    if option_chain is None:

        return None

    try:

        records = (
            option_chain
            .get("records", {})
            .get("data", [])
        )

        expiry = get_nearest_expiry(
            option_chain
        )

        if expiry is None:

            return None

        step = (
            INDEX_CONFIG[index_name]
            ["strike_step"]
        )

        atm = (
            round(
                spot_price / step
            )
            * step
        )

        if option_type == "CE":

            preferred = [
                atm,
                atm - step,
                atm - (2 * step)
            ]

        else:

            preferred = [
                atm,
                atm + step,
                atm + (2 * step)
            ]

        available = []

        for row in records:

            if (
                row.get("expiryDate")
                != expiry
            ):
                continue

            if option_type not in row:
                continue

            strike = safe_float(
                row.get("strikePrice")
            )

            option_data = row[
                option_type
            ]

            premium = safe_float(
                option_data.get(
                    "lastPrice"
                )
            )

            if (
                strike is None
                or premium is None
                or premium <= 0
            ):
                continue

            available.append({

                "strike":
                    strike,

                "premium":
                    premium

            })

        for preferred_strike in preferred:

            for item in available:

                if (
                    item["strike"]
                    == preferred_strike
                ):

                    return {

                        "symbol":
                            (
                                f"{index_name} "
                                f"{int(item['strike'])} "
                                f"{option_type}"
                            ),

                        "strike":
                            item["strike"],

                        "premium":
                            item["premium"],

                        "expiry":
                            expiry,

                        "option_type":
                            option_type

                    }

        return None

    except Exception:

        return None


# ============================================================
# FIND EXACT OPTION
# ============================================================

def find_exact_option_contract(
    option_chain,
    strike,
    option_type,
    expiry
):

    if option_chain is None:

        return None

    try:

        records = (
            option_chain
            .get("records", {})
            .get("data", [])
        )

        for row in records:

            if (
                row.get("expiryDate")
                != expiry
            ):
                continue

            row_strike = safe_float(
                row.get("strikePrice")
            )

            if row_strike != strike:
                continue

            if option_type not in row:
                continue

            premium = safe_float(
                row[option_type]
                .get("lastPrice")
            )

            if premium is None:

                return None

            return {

                "premium":
                    premium

            }

        return None

    except Exception:

        return None


# ============================================================
# OPTION LEVELS
#
# INDEX LEVELS CONTROL THE TRADE.
# OPTION PREMIUM IS DISPLAYED.
# ============================================================

def calculate_option_levels(option_premium):

    risk_percent = 0.20

    risk = (
        option_premium *
        risk_percent
    )

    return {

        "entry":
            option_premium,

        "sl":
            option_premium - risk,

        "target":
            option_premium +
            risk * RISK_REWARD

    }


# ============================================================
# CREATE TRADE
# ============================================================

def create_trade(
    signal_data,
    levels,
    option_contract,
    option_levels
):

    return {

        "signal":
            signal_data["signal"],

        "strength":
            signal_data["strength"],

        "status":
            "RUNNING",

        "entry_time":
            datetime.now(),

        "index_entry":
            levels["entry"],

        "index_sl":
            levels["stop_loss"],

        "index_target":
            levels["target"],

        "option_symbol":
            option_contract["symbol"],

        "option_strike":
            option_contract["strike"],

        "option_type":
            option_contract["option_type"],

        "option_expiry":
            option_contract["expiry"],

        "option_entry":
            option_levels["entry"],

        "option_sl":
            option_levels["sl"],

        "option_target":
            option_levels["target"],

        "last_premium":
            option_levels["entry"],

        "points":
            0.0,

        "exit_reason":
            None,

        "exit_premium":
            None

    }


# ============================================================
# CLOSE TRADE
# ============================================================

def close_running_trade():

    trade = (
        st.session_state.running_trade
    )

    if trade is None:

        return

    exit_premium = (
        trade.get("exit_premium")
    )

    if exit_premium is None:

        exit_premium = (
            trade["last_premium"]
        )

    trade["points"] = (

        exit_premium
        -
        trade["option_entry"]

    )

    trade["exit_premium"] = (
        exit_premium
    )

    trade["status"] = (
        "CLOSED"
    )

    trade["exit_time"] = (
        datetime.now()
    )

    st.session_state.closed_trades.append(
        trade.copy()
    )

    st.session_state.running_trade = None


# ============================================================
# UPDATE RUNNING TRADE
# ============================================================

def update_running_trade(
    current_index_price,
    option_data
):

    trade = (
        st.session_state.running_trade
    )

    if trade is None:

        return

    if option_data is None:

        return

    premium = safe_float(
        option_data["premium"]
    )

    trade["last_premium"] = (
        premium
    )

    trade["points"] = (

        premium
        -
        trade["option_entry"]

    )

    # --------------------------------------------------------
    # OPTION TARGET
    # --------------------------------------------------------

    if premium >= trade["option_target"]:

        trade["exit_reason"] = (
            "OPTION TARGET HIT"
        )

        trade["exit_premium"] = (
            premium
        )

        close_running_trade()

        return

    # --------------------------------------------------------
    # OPTION STOP
    # --------------------------------------------------------

    if premium <= trade["option_sl"]:

        trade["exit_reason"] = (
            "OPTION STOP LOSS HIT"
        )

        trade["exit_premium"] = (
            premium
        )

        close_running_trade()

        return

    # --------------------------------------------------------
    # INDEX TARGET / STOP
    # --------------------------------------------------------

    if trade["signal"] == "CALL":

        if (
            current_index_price
            >= trade["index_target"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET HIT"
            )

            trade["exit_premium"] = (
                premium
            )

            close_running_trade()

            return

        if (
            current_index_price
            <= trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = (
                premium
            )

            close_running_trade()

            return

    else:

        if (
            current_index_price
            <= trade["index_target"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET HIT"
            )

            trade["exit_premium"] = (
                premium
            )

            close_running_trade()

            return

        if (
            current_index_price
            >= trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = (
                premium
            )

            close_running_trade()

            return


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

@st.cache_data(ttl=30)
def run_historical_backtest(data):

    trades = []

    if len(data) < 40:

        return pd.DataFrame()

    position = 25

    while position < len(data) - 1:

        signal_data = (
            calculate_signal_at(
                data,
                position
            )
        )

        if signal_data is None:

            position += 1
            continue

        if (
            signal_data["signal"]
            not in ["CALL", "PUT"]
        ):

            position += 1
            continue

        levels = (
            calculate_trade_levels(
                signal_data
            )
        )

        if levels is None:

            position += 1
            continue

        # ----------------------------------------------------
        # ENTRY IS CONFIRMATION BREAK LEVEL
        #
        # Trigger candle already confirmed the break.
        # ----------------------------------------------------

        entry = levels["entry"]

        sl = levels["stop_loss"]

        target = levels["target"]

        risk = levels["risk"]

        if risk <= 0:

            position += 1
            continue

        # ----------------------------------------------------
        # ENTRY AT BREAK
        #
        # Conservative gap handling
        # ----------------------------------------------------

        trigger = data.iloc[position]

        trigger_open = safe_float(
            trigger["Open"]
        )

        if signal_data["signal"] == "CALL":

            actual_entry = max(
                entry,
                trigger_open
            )

        else:

            actual_entry = min(
                entry,
                trigger_open
            )

        actual_risk = abs(
            actual_entry -
            sl
        )

        if actual_risk <= 0:

            position += 1
            continue

        if signal_data["signal"] == "CALL":

            actual_target = (
                actual_entry +
                actual_risk *
                RISK_REWARD
            )

        else:

            actual_target = (
                actual_entry -
                actual_risk *
                RISK_REWARD
            )

        exit_found = False

        end_position = min(

            position +
            BACKTEST_MAX_HOLD_CANDLES,

            len(data) - 1

        )

        # ----------------------------------------------------
        # CHECK FROM NEXT CANDLE
        #
        # Signal candle itself is used only
        # for break confirmation.
        # ----------------------------------------------------

        exit_position = (
            end_position
        )

        for future_position in range(

            position + 1,

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

            if signal_data["signal"] == "CALL":

                # Conservative: SL first
                if low <= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_position = (
                        future_position
                    )

                    exit_found = True

                    break

                if high >= actual_target:

                    exit_price = (
                        actual_target
                    )

                    exit_reason = (
                        "TARGET HIT"
                    )

                    exit_position = (
                        future_position
                    )

                    exit_found = True

                    break

            else:

                # Conservative: SL first
                if high >= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_position = (
                        future_position
                    )

                    exit_found = True

                    break

                if low <= actual_target:

                    exit_price = (
                        actual_target
                    )

                    exit_reason = (
                        "TARGET HIT"
                    )

                    exit_position = (
                        future_position
                    )

                    exit_found = True

                    break

        # ----------------------------------------------------
        # TIME EXIT
        # ----------------------------------------------------

        if not exit_found:

            final = data.iloc[
                end_position
            ]

            exit_price = safe_float(
                final["Close"]
            )

            exit_reason = (
                "TIME EXIT"
            )

            exit_position = (
                end_position
            )

        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------

        if signal_data["signal"] == "CALL":

            points = (

                exit_price
                -
                actual_entry

            )

        else:

            points = (

                actual_entry
                -
                exit_price

            )

        trades.append({

            "Signal":
                signal_data["signal"],

            "Strength":
                signal_data["strength"],

            "Entry Time":
                str(
                    data.index[position]
                ),

            "Exit Time":
                str(
                    data.index[
                        exit_position
                    ]
                ),

            "Entry":
                round(
                    actual_entry,
                    2
                ),

            "SL":
                round(
                    sl,
                    2
                ),

            "Target":
                round(
                    actual_target,
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

            "Exit Reason":
                exit_reason

        })

        # No overlapping trades
        position = (
            exit_position + 1
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# BACKTEST STATS
# ============================================================

def get_backtest_stats(df):

    if df is None or df.empty:

        return {

            "total": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0,
            "net_points": 0

        }

    total = len(df)

    wins = len(
        df[
            df["Points"] > 0
        ]
    )

    losses = len(
        df[
            df["Points"] < 0
        ]
    )

    breakeven = len(
        df[
            df["Points"] == 0
        ]
    )

    return {

        "total":
            total,

        "wins":
            wins,

        "losses":
            losses,

        "breakeven":
            breakeven,

        "win_rate":
            (
                wins / total * 100
                if total > 0
                else 0
            ),

        "net_points":
            df["Points"].sum()

    }


# ============================================================
# LIVE STATS
# ============================================================

def get_live_stats():

    trades = (
        st.session_state.closed_trades
    )

    if not trades:

        return {

            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0

        }

    closed = len(trades)

    wins = len([
        trade
        for trade in trades
        if trade["points"] > 0
    ])

    losses = len([
        trade
        for trade in trades
        if trade["points"] < 0
    ])

    net_points = sum(
        trade["points"]
        for trade in trades
    )

    return {

        "closed":
            closed,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            wins / closed * 100,

        "net_points":
            net_points

    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Personal 9-15 EMA Scalping Scanner"
)

st.caption(
    "Strict 9 EMA + 15 EMA Strategy | "
    "Trend + Slope + Separation | "
    "Pullback + Confirmation Candle + Break Entry | "
    "1:2 Risk:Reward"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns(
    [1, 1, 0.5]
)

with col1:

    index_name = st.selectbox(
        "Select Index",
        list(INDEX_CONFIG.keys()),
        index=0
    )

with col2:

    timeframe = st.selectbox(
        "Select Timeframe",
        list(TIMEFRAME_CONFIG.keys()),
        index=3
    )

with col3:

    st.write("")

    if st.button("🔄 Refresh"):

        get_market_data.clear()

        get_option_chain.clear()

        run_historical_backtest.clear()

        st.rerun()


# ============================================================
# MARKET DATA
# ============================================================

symbol = (
    INDEX_CONFIG[index_name]["yahoo"]
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
# INDICATORS
# ============================================================

data = calculate_indicators(df)

signal_data = get_signal(data)

if signal_data is None:

    st.warning(
        "Signal calculate करने के लिए पर्याप्त data नहीं है।"
    )

    st.stop()


signal = (
    signal_data["signal"]
)

last = data.iloc[-1]

price = safe_float(
    last["Close"]
)

ema9 = safe_float(
    last["EMA9"]
)

ema15 = safe_float(
    last["EMA15"]
)

levels = (
    calculate_trade_levels(
        signal_data
    )
)


# ============================================================
# OPTION DATA
# ============================================================

option_contract = None

option_levels = None

if signal in ["CALL", "PUT"]:

    option_type = (
        "CE"
        if signal == "CALL"
        else "PE"
    )

    option_chain = get_option_chain(
        index_name
    )

    option_contract = (
        find_option_contract(
            option_chain,
            price,
            option_type,
            index_name
        )
    )

    if option_contract is not None:

        option_levels = (
            calculate_option_levels(
                option_contract["premium"]
            )
        )


# ============================================================
# UPDATE RUNNING TRADE
# ============================================================

if (
    st.session_state.running_trade
    is not None
):

    running_trade = (
        st.session_state.running_trade
    )

    option_chain = get_option_chain(
        index_name
    )

    exact_option = (
        find_exact_option_contract(

            option_chain,

            running_trade[
                "option_strike"
            ],

            running_trade[
                "option_type"
            ],

            running_trade[
                "option_expiry"
            ]

        )
    )

    update_running_trade(
        price,
        exact_option
    )


# ============================================================
# CREATE NEW LIVE TRADE
# ============================================================

trade_allowed = (

    signal in ["CALL", "PUT"]

    and

    levels is not None

    and

    option_contract is not None

    and

    option_levels is not None

)

if (

    trade_allowed

    and

    st.session_state.running_trade
    is None
):

    signal_time = str(
        signal_data["time"]
    )

    if (
        st.session_state
        .last_trade_signal_time
        != signal_time
    ):

        st.session_state.running_trade = (

            create_trade(

                signal_data,

                levels,

                option_contract,

                option_levels

            )

        )

        st.session_state.last_trade_signal_time = (
            signal_time
        )


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.markdown(
    "## 🎯 Current Signal"
)

if signal == "CALL":

    signal_color_class = (
        "call-text"
    )

elif signal == "PUT":

    signal_color_class = (
        "put-text"
    )

else:

    signal_color_class = (
        "wait-text"
    )


left, right = st.columns(2)

with left:

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-label">
                Signal
            </div>
            <div class="signal-value {signal_color_class}">
                {signal} ({signal_data["strength"]})
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:

    if st.session_state.running_trade:

        status = "RUNNING"

    elif trade_allowed:

        status = "READY"

    else:

        status = "NO TRADE"

    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-label">
                Trade Status
            </div>
            <div class="signal-value">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CURRENT VALUES
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Price",
        format_number(price)
    )

with m2:

    st.metric(
        "EMA 9",
        format_number(ema9)
    )

with m3:

    st.metric(
        "EMA 15",
        format_number(ema15)
    )

with m4:

    st.metric(
        "EMA Separation",
        format_number(
            abs(ema9 - ema15)
        )
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
# INDEX LEVELS
# ============================================================

if levels is not None:

    st.markdown(
        "### 📍 Trade Levels"
    )

    l1, l2, l3, l4 = st.columns(4)

    with l1:

        st.metric(
            "Entry",
            format_number(
                levels["entry"]
            )
        )

    with l2:

        st.metric(
            "Stop Loss",
            format_number(
                levels["stop_loss"]
            )
        )

    with l3:

        st.metric(
            "Target (1:2)",
            format_number(
                levels["target"]
            )
        )

    with l4:

        st.metric(
            "Risk",
            format_number(
                levels["risk"]
            )
        )


# ============================================================
# LIVE OPTION PREMIUM
# ============================================================

st.markdown(
    "## 💰 Live Option Premium"
)

if option_contract is None:

    st.info(
        "कोई valid option setup उपलब्ध नहीं है।"
    )

else:

    o1, o2, o3, o4 = st.columns(4)

    with o1:

        st.metric(
            "Option",
            option_contract["symbol"]
        )

    with o2:

        st.metric(
            "Premium",
            format_number(
                option_contract["premium"]
            )
        )

    with o3:

        st.metric(
            "Option SL",
            format_number(
                option_levels["sl"]
            )
        )

    with o4:

        st.metric(
            "Option Target",
            format_number(
                option_levels["target"]
            )
        )


# ============================================================
# CHART
# ============================================================

st.markdown(
    "## 📊 Index Chart"
)

chart_data = (
    data.tail(150).copy()
)

fig = go.Figure()


# ------------------------------------------------------------
# CANDLESTICK
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# EMA 9
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA9"],

        mode="lines",

        name="EMA 9",

        line=dict(
            width=1.5
        )

    )

)


# ------------------------------------------------------------
# EMA 15
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA15"],

        mode="lines",

        name="EMA 15",

        line=dict(
            width=1.5
        )

    )

)


# ------------------------------------------------------------
# TRADE LEVELS
# ------------------------------------------------------------

if levels is not None:

    fig.add_hline(

        y=levels["entry"],

        line_dash="dot",

        annotation_text=(
            f"{signal} ENTRY "
            f"{levels['entry']:.2f}"
        )

    )

    fig.add_hline(

        y=levels["stop_loss"],

        line_dash="dash",

        annotation_text=(
            f"SL "
            f"{levels['stop_loss']:.2f}"
        )

    )

    fig.add_hline(

        y=levels["target"],

        line_dash="dash",

        annotation_text=(
            f"TARGET 1:2 "
            f"{levels['target']:.2f}"
        )

    )


# ------------------------------------------------------------
# RUNNING TRADE LEVELS
# ------------------------------------------------------------

running = (
    st.session_state.running_trade
)

if running is not None:

    fig.add_hline(

        y=running["index_entry"],

        line_dash="dot",

        annotation_text="RUNNING ENTRY"

    )

    fig.add_hline(

        y=running["index_sl"],

        line_dash="dash",

        annotation_text="RUNNING SL"

    )

    fig.add_hline(

        y=running["index_target"],

        line_dash="dash",

        annotation_text="RUNNING TARGET"

    )


fig.update_layout(

    height=620,

    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10
    ),

    xaxis_rangeslider_visible=False,

    template="plotly_dark",

    legend=dict(
        orientation="h"
    )

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# RUNNING TRADE
# ============================================================

st.markdown(
    "## 🔵 Running Trade"
)

running = (
    st.session_state.running_trade
)

if running is None:

    st.info(
        "No running trade."
    )

else:

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "Signal",
            running["signal"]
        )

    with r2:

        st.metric(
            "Option Entry",
            format_number(
                running["option_entry"]
            )
        )

    with r3:

        st.metric(
            "Current Premium",
            format_number(
                running["last_premium"]
            )
        )

    with r4:

        st.metric(
            "P/L Points",
            format_number(
                running["points"]
            )
        )

    if st.button(
        "🔴 Close Running Trade"
    ):

        trade = (
            st.session_state.running_trade
        )

        trade["exit_reason"] = (
            "MANUAL EXIT"
        )

        trade["exit_premium"] = (
            trade["last_premium"]
        )

        close_running_trade()

        st.rerun()


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

stats = (
    get_backtest_stats(
        backtest_df
    )
)


b1, b2, b3, b4 = st.columns(4)

with b1:

    st.metric(
        "Total Trades",
        stats["total"]
    )

with b2:

    st.metric(
        "Wins",
        stats["wins"]
    )

with b3:

    st.metric(
        "Losses",
        stats["losses"]
    )

with b4:

    st.metric(
        "Win Rate",
        f"{stats['win_rate']:.1f}%"
    )


b1, b2, b3 = st.columns(3)

with b1:

    st.metric(
        "Breakeven",
        stats["breakeven"]
    )

with b2:

    st.metric(
        "Net Points",
        f"{stats['net_points']:.2f}"
    )

with b3:

    st.metric(
        "Candles Tested",
        len(data)
    )


# ============================================================
# BACKTEST TABLE
# ============================================================

st.markdown(
    "### Recent Backtest Trades"
)

if backtest_df.empty:

    st.info(
        "इस strict strategy में कोई valid trade नहीं मिला।"
    )

else:

    st.dataframe(

        backtest_df.iloc[
            ::-1
        ].head(10),

        use_container_width=True

    )


# ============================================================
# LIVE CLOSED TRADES
# ============================================================

st.markdown(
    "## Recent Live Closed Trades"
)

live_stats = (
    get_live_stats()
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Closed Trades",
        live_stats["closed"]
    )

with c2:

    st.metric(
        "Wins",
        live_stats["wins"]
    )

with c3:

    st.metric(
        "Win Rate",
        f"{live_stats['win_rate']:.1f}%"
    )

with c4:

    st.metric(
        "Net Premium Points",
        f"{live_stats['net_points']:.2f}"
    )


closed_trades = (
    st.session_state.closed_trades
)

if not closed_trades:

    st.info(
        "No live closed trades yet."
    )

else:

    rows = []

    for trade in reversed(
        closed_trades[-10:]
    ):

        rows.append({

            "Signal":
                trade["signal"],

            "Option":
                trade["option_symbol"],

            "Entry":
                round(
                    trade["option_entry"],
                    2
                ),

            "Exit":
                round(
                    trade["exit_premium"],
                    2
                ),

            "Points":
                round(
                    trade["points"],
                    2
                ),

            "Exit Reason":
                trade["exit_reason"]

        })

    st.dataframe(

        pd.DataFrame(rows),

        use_container_width=True

    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "9 EMA + 15 EMA Strict Scalping Strategy | "
    "Trend + Slope + Separation + Pullback + "
    "Confirmation Candle + Break Entry | "
    "Fixed 1:2 Risk:Reward"
)
