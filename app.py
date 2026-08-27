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

.signal-call {
    color: #48d597;
    font-weight: 800;
}

.signal-put {
    color: #ff6b81;
    font-weight: 800;
}

.signal-wait {
    color: #f3c969;
    font-weight: 800;
}

.signal-strong {
    color: #48d597;
    font-weight: 800;
}

.signal-medium {
    color: #f3c969;
    font-weight: 800;
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
    },

    "1h": {
        "interval": "60m",
        "period": "1mo",
        "resample": None
    },

    "2h": {
        "interval": "60m",
        "period": "1mo",
        "resample": "2h"
    },

    "1d": {
        "interval": "1d",
        "period": "6mo",
        "resample": None
    }

}


# ============================================================
# STRATEGY SETTINGS
# ============================================================

RISK_REWARD = 2.0

BACKTEST_MAX_HOLD_CANDLES = 30

EMA_SLOPE_LOOKBACK = 3

MIN_EMA_SEPARATION_PCT = 0.0008

MIN_SLOPE_PCT = 0.0003

PULLBACK_LOOKBACK = 3

MAX_ENTRY_DISTANCE_PCT = 0.0025

MIN_BODY_TO_RANGE = 0.45

MIN_STRONG_BODY_TO_RANGE = 0.60

MAX_OPPOSITE_WICK_TO_BODY = 1.2

REJECTION_WICK_MULTIPLIER = 1.2

MIN_RISK_POINTS = 5

MAX_RISK_POINTS = 30


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
                df.resample(config["resample"])
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

    # EMA 9

    data["EMA9"] = (
        data["Close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # EMA 15

    data["EMA15"] = (
        data["Close"]
        .ewm(
            span=15,
            adjust=False
        )
        .mean()
    )

    # Typical Price

    data["TypicalPrice"] = (
        data["High"]
        + data["Low"]
        + data["Close"]
    ) / 3

    # VWAP - daily reset

    data["TradeDate"] = (
        pd.to_datetime(data.index).date
    )

    volume = (
        data["Volume"]
        .replace(0, np.nan)
        .fillna(1)
    )

    data["VWAP"] = (

        (
            data["TypicalPrice"]
            * volume
        )
        .groupby(data["TradeDate"])
        .cumsum()

        /

        volume
        .groupby(data["TradeDate"])
        .cumsum()

    )

    data["VWAP"] = (
        data["VWAP"]
        .fillna(data["TypicalPrice"])
    )

    # Candle values

    data["Body"] = (
        data["Close"]
        - data["Open"]
    ).abs()

    data["Range"] = (
        data["High"]
        - data["Low"]
    )

    data["UpperWick"] = (

        data["High"]

        -

        data[
            ["Open", "Close"]
        ].max(axis=1)

    )

    data["LowerWick"] = (

        data[
            ["Open", "Close"]
        ].min(axis=1)

        -

        data["Low"]

    )

    # ATR

    previous_close = (
        data["Close"].shift(1)
    )

    tr1 = (
        data["High"]
        - data["Low"]
    )

    tr2 = (
        data["High"]
        - previous_close
    ).abs()

    tr3 = (
        data["Low"]
        - previous_close
    ).abs()

    true_range = (
        pd.concat(
            [tr1, tr2, tr3],
            axis=1
        )
        .max(axis=1)
    )

    data["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # EMA slopes

    data["EMA9_Slope"] = (
        data["EMA9"]
        - data["EMA9"].shift(
            EMA_SLOPE_LOOKBACK
        )
    )

    data["EMA15_Slope"] = (
        data["EMA15"]
        - data["EMA15"].shift(
            EMA_SLOPE_LOOKBACK
        )
    )

    # EMA separation %

    data["EMA_Separation_Pct"] = (

        (
            data["EMA9"]
            - data["EMA15"]
        ).abs()

        /

        data["Close"]

    )

    # Body / range ratio

    data["Body_Range_Ratio"] = (

        data["Body"]

        /

        data["Range"]
        .replace(0, np.nan)

    ).fillna(0)

    return data


# ============================================================
# NSE SESSION
# ============================================================

def get_nse_session():

    session = requests.Session()

    session.headers.update({

        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",

        "Accept":
            "application/json, text/plain, */*",

        "Accept-Language":
            "en-US,en;q=0.9",

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
                    ).date()
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
# OPTION CONTRACT
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

        expiry = (
            get_nearest_expiry(
                option_chain
            )
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

        # ITM first, ATM second

        if option_type == "CE":

            preferred = [

                atm - step,

                atm,

                atm - (2 * step)

            ]

        else:

            preferred = [

                atm + step,

                atm,

                atm + (2 * step)

            ]

        available = {}

        for row in records:

            if row.get(
                "expiryDate"
            ) != expiry:
                continue

            if option_type not in row:
                continue

            strike = safe_float(
                row.get("strikePrice")
            )

            premium = safe_float(
                row[option_type]
                .get("lastPrice")
            )

            if (
                strike is None
                or premium is None
                or premium <= 0
            ):
                continue

            available[strike] = premium

        for strike in preferred:

            if strike in available:

                return {

                    "option_type":
                        option_type,

                    "strike":
                        strike,

                    "premium":
                        available[strike],

                    "expiry":
                        expiry,

                    "symbol":
                        f"{index_name} "
                        f"{int(strike)} "
                        f"{option_type}"

                }

        return None

    except Exception:
        return None


# ============================================================
# EXACT OPTION CONTRACT
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
                safe_float(
                    row.get("strikePrice")
                ) != strike
            ):
                continue

            if (
                row.get("expiryDate")
                != expiry
            ):
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

                "strike":
                    strike,

                "premium":
                    premium,

                "expiry":
                    expiry,

                "option_type":
                    option_type

            }

        return None

    except Exception:
        return None


# ============================================================
# CANDLE PATTERNS
# ============================================================

def is_bullish_rejection(row):

    body = safe_float(
        row["Body"],
        0
    )

    lower_wick = safe_float(
        row["LowerWick"],
        0
    )

    upper_wick = safe_float(
        row["UpperWick"],
        0
    )

    if body <= 0:
        return False

    return (

        safe_float(row["Close"])
        >
        safe_float(row["Open"])

        and

        lower_wick
        >=
        body * REJECTION_WICK_MULTIPLIER

        and

        lower_wick > upper_wick

    )


def is_bearish_rejection(row):

    body = safe_float(
        row["Body"],
        0
    )

    upper_wick = safe_float(
        row["UpperWick"],
        0
    )

    lower_wick = safe_float(
        row["LowerWick"],
        0
    )

    if body <= 0:
        return False

    return (

        safe_float(row["Close"])
        <
        safe_float(row["Open"])

        and

        upper_wick
        >=
        body * REJECTION_WICK_MULTIPLIER

        and

        upper_wick > lower_wick

    )


def is_bullish_engulfing(previous, current):

    return (

        safe_float(current["Close"])
        >
        safe_float(current["Open"])

        and

        safe_float(previous["Close"])
        <
        safe_float(previous["Open"])

        and

        safe_float(current["Close"])
        >=
        safe_float(previous["Open"])

        and

        safe_float(current["Open"])
        <=
        safe_float(previous["Close"])

    )


def is_bearish_engulfing(previous, current):

    return (

        safe_float(current["Close"])
        <
        safe_float(current["Open"])

        and

        safe_float(previous["Close"])
        >
        safe_float(previous["Open"])

        and

        safe_float(current["Close"])
        <=
        safe_float(previous["Open"])

        and

        safe_float(current["Open"])
        >=
        safe_float(previous["Close"])

    )


# ============================================================
# SIGNAL CALCULATION
# ============================================================

def calculate_signal_at(data, position):

    minimum_required = max(
        25,
        EMA_SLOPE_LOOKBACK + 15
    )

    if position < minimum_required:
        return None

    current = data.iloc[position]

    previous = data.iloc[
        position - 1
    ]

    price = safe_float(
        current["Close"]
    )

    ema9 = safe_float(
        current["EMA9"]
    )

    ema15 = safe_float(
        current["EMA15"]
    )

    vwap = safe_float(
        current["VWAP"]
    )

    atr = safe_float(
        current["ATR"],
        0
    )

    ema9_slope = safe_float(
        current["EMA9_Slope"],
        0
    )

    ema15_slope = safe_float(
        current["EMA15_Slope"],
        0
    )

    separation_pct = safe_float(
        current["EMA_Separation_Pct"],
        0
    )

    body_ratio = safe_float(
        current["Body_Range_Ratio"],
        0
    )

    current_range = safe_float(
        current["Range"],
        0
    )

    # ========================================================
    # TREND
    # ========================================================

    bullish_trend = (
        ema9 > ema15
        and price > ema15
    )

    bearish_trend = (
        ema9 < ema15
        and price < ema15
    )

    # ========================================================
    # SLOPE
    # ========================================================

    ema9_slope_pct = (

        ema9_slope / ema9

        if ema9 != 0

        else 0

    )

    ema15_slope_pct = (

        ema15_slope / ema15

        if ema15 != 0

        else 0

    )

    bullish_slope = (

        ema9_slope_pct
        >=
        MIN_SLOPE_PCT

        and

        ema15_slope_pct
        >=
        MIN_SLOPE_PCT * 0.50

    )

    bearish_slope = (

        ema9_slope_pct
        <=
        -MIN_SLOPE_PCT

        and

        ema15_slope_pct
        <=
        -MIN_SLOPE_PCT * 0.50

    )

    # ========================================================
    # SIDEWAYS FILTER
    # ========================================================

    enough_separation = (
        separation_pct
        >=
        MIN_EMA_SEPARATION_PCT
    )

    # ========================================================
    # PULLBACK / EMA TOUCH
    # ========================================================

    lookback_start = max(
        0,
        position - PULLBACK_LOOKBACK
    )

    pullback_data = data.iloc[
        lookback_start:
        position + 1
    ]

    bullish_touch = False
    bearish_touch = False

    for _, candle in pullback_data.iterrows():

        candle_low = safe_float(
            candle["Low"]
        )

        candle_high = safe_float(
            candle["High"]
        )

        candle_ema9 = safe_float(
            candle["EMA9"]
        )

        candle_ema15 = safe_float(
            candle["EMA15"]
        )

        if (

            candle_low <= candle_ema9
            and
            candle_high >= candle_ema15

        ):

            bullish_touch = True
            bearish_touch = True

        elif (

            candle_low <= candle_ema9
            and
            candle_high >= candle_ema9

        ):

            bullish_touch = True
            bearish_touch = True

    # ========================================================
    # CANDLE CONFIRMATION
    # ========================================================

    bullish_candle = (

        price
        >
        safe_float(current["Open"])

        and

        body_ratio
        >=
        MIN_BODY_TO_RANGE

    )

    bearish_candle = (

        price
        <
        safe_float(current["Open"])

        and

        body_ratio
        >=
        MIN_BODY_TO_RANGE

    )

    bullish_rejection = (
        is_bullish_rejection(
            current
        )
    )

    bearish_rejection = (
        is_bearish_rejection(
            current
        )
    )

    bullish_engulfing = (
        is_bullish_engulfing(
            previous,
            current
        )
    )

    bearish_engulfing = (
        is_bearish_engulfing(
            previous,
            current
        )
    )

    strong_bullish = (

        bullish_candle

        and

        body_ratio
        >=
        MIN_STRONG_BODY_TO_RANGE

        and

        safe_float(
            current["UpperWick"],
            0
        )

        <=

        safe_float(
            current["Body"],
            0
        )

        *
        MAX_OPPOSITE_WICK_TO_BODY

    )

    strong_bearish = (

        bearish_candle

        and

        body_ratio
        >=
        MIN_STRONG_BODY_TO_RANGE

        and

        safe_float(
            current["LowerWick"],
            0
        )

        <=

        safe_float(
            current["Body"],
            0
        )

        *
        MAX_OPPOSITE_WICK_TO_BODY

    )

    bullish_confirmation = (

        bullish_rejection

        or

        bullish_engulfing

        or

        strong_bullish

    )

    bearish_confirmation = (

        bearish_rejection

        or

        bearish_engulfing

        or

        strong_bearish

    )

    # ========================================================
    # LATE ENTRY FILTER
    # ========================================================

    distance_from_ema9 = (

        abs(
            price - ema9
        )

        /

        price

        if price != 0

        else 999

    )

    not_late_entry = (
        distance_from_ema9
        <=
        MAX_ENTRY_DISTANCE_PCT
    )

    # ========================================================
    # VWAP EXTRA FILTER
    # ========================================================

    bullish_vwap = (
        price >= vwap
    )

    bearish_vwap = (
        price <= vwap
    )

    # ========================================================
    # FINAL CALL
    # ========================================================

    call_valid = (

        bullish_trend

        and

        bullish_slope

        and

        enough_separation

        and

        bullish_touch

        and

        bullish_confirmation

        and

        not_late_entry

        and

        bullish_vwap

    )

    # ========================================================
    # FINAL PUT
    # ========================================================

    put_valid = (

        bearish_trend

        and

        bearish_slope

        and

        enough_separation

        and

        bearish_touch

        and

        bearish_confirmation

        and

        not_late_entry

        and

        bearish_vwap

    )

    # ========================================================
    # STRENGTH
    # ========================================================

    signal = "WAIT"
    strength = "LOW"
    reason = []

    if call_valid:

        signal = "CALL"

        if (
            strong_bullish
            or
            bullish_engulfing
        ):
            strength = "STRONG"
        else:
            strength = "MEDIUM"

        reason.append(
            "Bullish EMA 9 > EMA 15"
        )

        reason.append(
            "Upward EMA slope"
        )

        reason.append(
            "EMA pullback/touch"
        )

        if bullish_engulfing:
            reason.append(
                "Bullish engulfing"
            )

        elif bullish_rejection:
            reason.append(
                "Bullish rejection"
            )

        else:
            reason.append(
                "Strong bullish candle"
            )

    elif put_valid:

        signal = "PUT"

        if (
            strong_bearish
            or
            bearish_engulfing
        ):
            strength = "STRONG"
        else:
            strength = "MEDIUM"

        reason.append(
            "Bearish EMA 9 < EMA 15"
        )

        reason.append(
            "Downward EMA slope"
        )

        reason.append(
            "EMA pullback/touch"
        )

        if bearish_engulfing:
            reason.append(
                "Bearish engulfing"
            )

        elif bearish_rejection:
            reason.append(
                "Bearish rejection"
            )

        else:
            reason.append(
                "Strong bearish candle"
            )

    else:

        if not enough_separation:
            reason.append(
                "EMA flat / too close"
            )

        elif (
            not bullish_slope
            and
            not bearish_slope
        ):
            reason.append(
                "EMA slope weak"
            )

        elif not not_late_entry:
            reason.append(
                "Late entry - price far from EMA"
            )

        elif (
            not bullish_confirmation
            and
            not bearish_confirmation
        ):
            reason.append(
                "No strong candle confirmation"
            )

        else:
            reason.append(
                "No valid 9/15 EMA setup"
            )

    return {

        "signal":
            signal,

        "strength":
            strength,

        "price":
            price,

        "ema9":
            ema9,

        "ema15":
            ema15,

        "vwap":
            vwap,

        "atr":
            atr,

        "reason":
            " | ".join(reason),

        "time":
            data.index[position],

        "entry_trigger":
            safe_float(
                current["High"]
            )

            if signal == "CALL"

            else

            safe_float(
                current["Low"]
            )

            if signal == "PUT"

            else None,

        "signal_high":
            safe_float(
                current["High"]
            ),

        "signal_low":
            safe_float(
                current["Low"]
            )

    }


# ============================================================
# CURRENT SIGNAL
# ============================================================

def get_signal(data):

    if len(data) < 30:
        return None

    return calculate_signal_at(
        data,
        len(data) - 1
    )


# ============================================================
# INDEX LEVELS
# ============================================================

def calculate_index_levels(
    signal_data,
    candle
):

    signal = signal_data["signal"]

    if signal == "WAIT":

        return {

            "entry": None,
            "stop_loss": None,
            "target": None,
            "risk": None

        }

    # Entry = signal candle break

    if signal == "CALL":

        entry = (
            safe_float(
                candle["High"]
            )
        )

        stop_loss = (
            safe_float(
                candle["Low"]
            )
        )

    else:

        entry = (
            safe_float(
                candle["Low"]
            )
        )

        stop_loss = (
            safe_float(
                candle["High"]
            )
        )

    risk = abs(
        entry - stop_loss
    )

    # ATR fallback

    if risk <= 0:

        atr = safe_float(
            signal_data.get(
                "atr",
                0
            )
        )

        risk = max(
            MIN_RISK_POINTS,
            min(
                MAX_RISK_POINTS,
                atr * 0.5
            )
        )

        if signal == "CALL":
            stop_loss = entry - risk
        else:
            stop_loss = entry + risk

    if signal == "CALL":

        target = (
            entry
            +
            risk * RISK_REWARD
        )

    else:

        target = (
            entry
            -
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
# OPTION LEVELS
# ============================================================

def calculate_option_levels(
    option_premium
):

    risk_percent = 0.20

    risk_amount = (
        option_premium
        * risk_percent
    )

    option_sl = (
        option_premium
        - risk_amount
    )

    option_target = (
        option_premium
        +
        risk_amount
        * RISK_REWARD
    )

    return {

        "entry":
            option_premium,

        "sl":
            option_sl,

        "target":
            option_target,

        "risk":
            risk_amount

    }


# ============================================================
# CREATE TRADE
# ============================================================

def create_trade(
    signal_data,
    index_levels,
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
            index_levels["entry"],

        "index_sl":
            index_levels["stop_loss"],

        "index_target":
            index_levels["target"],

        "index_risk":
            index_levels["risk"],

        "option_symbol":
            option_contract["symbol"],

        "option_type":
            option_contract["option_type"],

        "option_strike":
            option_contract["strike"],

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
            0,

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

    exit_price = trade.get(
        "exit_premium"
    )

    if exit_price is None:

        exit_price = (
            trade["last_premium"]
        )

    trade["points"] = (
        exit_price
        -
        trade["option_entry"]
    )

    trade["status"] = "CLOSED"

    trade["exit_time"] = (
        datetime.now()
    )

    trade["exit_premium"] = (
        exit_price
    )

    st.session_state.closed_trades.append(
        trade.copy()
    )

    st.session_state.running_trade = None


# ============================================================
# UPDATE TRADE
# ============================================================

def update_running_trade(
    current_index_price,
    current_option_contract
):

    trade = (
        st.session_state.running_trade
    )

    if trade is None:
        return

    if current_option_contract is None:
        return

    premium = (
        current_option_contract["premium"]
    )

    trade["last_premium"] = premium

    trade["points"] = (
        premium
        -
        trade["option_entry"]
    )

    # OPTION TARGET

    if premium >= trade["option_target"]:

        trade["exit_reason"] = (
            "OPTION TARGET HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # OPTION SL

    if premium <= trade["option_sl"]:

        trade["exit_reason"] = (
            "OPTION STOP LOSS HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # INDEX TARGET

    if (

        trade["signal"] == "CALL"

        and

        current_index_price
        >=
        trade["index_target"]

    ):

        trade["exit_reason"] = (
            "INDEX TARGET HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    if (

        trade["signal"] == "PUT"

        and

        current_index_price
        <=
        trade["index_target"]

    ):

        trade["exit_reason"] = (
            "INDEX TARGET HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # INDEX SL

    if (

        trade["signal"] == "CALL"

        and

        current_index_price
        <=
        trade["index_sl"]

    ):

        trade["exit_reason"] = (
            "INDEX STOP LOSS HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    if (

        trade["signal"] == "PUT"

        and

        current_index_price
        >=
        trade["index_sl"]

    ):

        trade["exit_reason"] = (
            "INDEX STOP LOSS HIT"
        )

        trade["exit_premium"] = premium

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

    while position < len(data) - 2:

        signal_data = (
            calculate_signal_at(
                data,
                position
            )
        )

        if signal_data is None:

            position += 1
            continue

        signal = (
            signal_data["signal"]
        )

        if signal not in [
            "CALL",
            "PUT"
        ]:

            position += 1
            continue

        signal_candle = (
            data.iloc[position]
        )

        levels = (
            calculate_index_levels(
                signal_data,
                signal_candle
            )
        )

        # ----------------------------------------------------
        # WAIT FOR ACTUAL BREAKOUT
        # ----------------------------------------------------

        trigger = (
            levels["entry"]
        )

        setup_sl = (
            levels["stop_loss"]
        )

        entry_position = None
        entry = None

        trigger_end = min(
            position + 5,
            len(data) - 1
        )

        for check_position in range(
            position + 1,
            trigger_end + 1
        ):

            future = (
                data.iloc[
                    check_position
                ]
            )

            future_high = safe_float(
                future["High"]
            )

            future_low = safe_float(
                future["Low"]
            )

            # CALL:
            # first invalidate setup if low breaks SL

            if signal == "CALL":

                if future_low <= setup_sl:
                    break

                if future_high > trigger:

                    entry_position = (
                        check_position
                    )

                    entry = trigger

                    break

            # PUT

            else:

                if future_high >= setup_sl:
                    break

                if future_low < trigger:

                    entry_position = (
                        check_position
                    )

                    entry = trigger

                    break

        # No valid break

        if entry_position is None:

            position += 1
            continue

        risk = abs(
            entry - setup_sl
        )

        if risk <= 0:

            position = (
                entry_position + 1
            )

            continue

        sl = setup_sl

        if signal == "CALL":

            target = (
                entry
                +
                risk * RISK_REWARD
            )

        else:

            target = (
                entry
                -
                risk * RISK_REWARD
            )

        exit_found = False

        end_position = min(

            entry_position
            +
            BACKTEST_MAX_HOLD_CANDLES,

            len(data) - 1

        )

        # Start after entry candle
        # avoids false same-candle sequencing

        for future_position in range(

            entry_position + 1,

            end_position + 1

        ):

            future = (
                data.iloc[
                    future_position
                ]
            )

            high = safe_float(
                future["High"]
            )

            low = safe_float(
                future["Low"]
            )

            if signal == "CALL":

                # Conservative SL first

                if low <= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_found = True

                    break

                if high >= target:

                    exit_price = target

                    exit_reason = (
                        "TARGET HIT"
                    )

                    exit_found = True

                    break

            else:

                # Conservative SL first

                if high >= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS HIT"
                    )

                    exit_found = True

                    break

                if low <= target:

                    exit_price = target

                    exit_reason = (
                        "TARGET HIT"
                    )

                    exit_found = True

                    break

        # TIME EXIT

        if not exit_found:

            final_candle = (
                data.iloc[
                    end_position
                ]
            )

            exit_price = safe_float(
                final_candle["Close"]
            )

            exit_reason = (
                "TIME EXIT"
            )

            future_position = (
                end_position
            )

        # POINTS

        if signal == "CALL":

            points = (
                exit_price
                -
                entry
            )

        else:

            points = (
                entry
                -
                exit_price
            )

        trades.append({

            "Signal":
                signal,

            "Strength":
                signal_data["strength"],

            "Signal Time":
                str(
                    data.index[position]
                ),

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

            "Target":
                round(target, 2),

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
            future_position + 1
        )

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ============================================================
# HISTORICAL STATS
# ============================================================

def get_historical_stats(backtest_df):

    if (
        backtest_df is None
        or
        backtest_df.empty
    ):

        return {

            "total": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0,
            "net_points": 0

        }

    total = len(backtest_df)

    wins = len(
        backtest_df[
            backtest_df["Points"] > 0
        ]
    )

    losses = len(
        backtest_df[
            backtest_df["Points"] < 0
        ]
    )

    breakeven = len(
        backtest_df[
            backtest_df["Points"] == 0
        ]
    )

    win_rate = (

        wins / total * 100

        if total > 0

        else 0

    )

    net_points = (
        backtest_df["Points"]
        .sum()
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
            win_rate,

        "net_points":
            net_points

    }


# ============================================================
# LIVE TRADE STATS
# ============================================================

def get_live_trade_stats():

    trades = (
        st.session_state.closed_trades
    )

    if len(trades) == 0:

        return {

            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0

        }

    wins = len([

        x for x in trades

        if x.get(
            "points",
            0
        ) > 0

    ])

    losses = len([

        x for x in trades

        if x.get(
            "points",
            0
        ) < 0

    ])

    closed = len(trades)

    net_points = sum(

        x.get(
            "points",
            0
        )

        for x in trades

    )

    win_rate = (
        wins / closed * 100
    )

    return {

        "closed":
            closed,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

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
    "Strict EMA Pullback Strategy | "
    "EMA Slope + Separation | "
    "Candle Confirmation | "
    "Signal Candle Break | "
    "VWAP Filter | "
    "ITM/ATM Options | "
    "1:2 Risk:Reward"
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
        index=3
    )

with col3:

    st.write("")

    if st.button(
        "🔄 Refresh"
    ):

        get_market_data.clear()
        get_option_chain.clear()
        run_historical_backtest.clear()

        st.rerun()


# ============================================================
# MARKET DATA
# ============================================================

symbol = (
    INDEX_CONFIG[index_name]
    ["yahoo"]
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

strength = (
    signal_data["strength"]
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


# ============================================================
# INDEX LEVELS
# ============================================================

last_candle = data.iloc[-1]

index_levels = (
    calculate_index_levels(
        signal_data,
        last_candle
    )
)


# ============================================================
# OPTION DATA
# ============================================================

option_contract = None
option_levels = None

if signal in [
    "CALL",
    "PUT"
]:

    option_chain = (
        get_option_chain(
            index_name
        )
    )

    option_type = (

        "CE"

        if signal == "CALL"

        else "PE"

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
                option_contract[
                    "premium"
                ]
            )
        )


# ============================================================
# UPDATE RUNNING TRADE
# ============================================================

if (
    st.session_state.running_trade
    is not None
):

    running = (
        st.session_state.running_trade
    )

    option_chain_live = (
        get_option_chain(
            index_name
        )
    )

    exact_option = (
        find_exact_option_contract(

            option_chain_live,

            running["option_strike"],

            running["option_type"],

            running["option_expiry"]

        )
    )

    update_running_trade(
        price,
        exact_option
    )


# ============================================================
# CREATE NEW TRADE
# ============================================================

trade_allowed = (

    signal in [
        "CALL",
        "PUT"
    ]

    and

    strength in [
        "MEDIUM",
        "STRONG"
    ]

)

if (

    trade_allowed

    and

    st.session_state.running_trade
    is None

    and

    option_contract is not None

    and

    option_levels is not None

):

    signal_time = str(
        signal_data["time"]
    )

    if (

        st.session_state
        .last_trade_signal_time

        !=

        signal_time

    ):

        st.session_state.running_trade = (
            create_trade(

                signal_data,

                index_levels,

                option_contract,

                option_levels

            )
        )

        st.session_state.last_trade_signal_time = (
            signal_time
        )


# ============================================================
# SIGNAL COLOR
# ============================================================

if signal == "CALL":

    signal_class = "signal-call"

elif signal == "PUT":

    signal_class = "signal-put"

else:

    signal_class = "signal-wait"


# ============================================================
# CURRENT SIGNAL
# ============================================================

st.markdown(
    "## 🎯 Current Signal"
)

m1, m2 = st.columns(2)

with m1:

    st.markdown(

        metric_box(

            "Signal",

            f"{signal} ({strength})",

            signal_class

        ),

        unsafe_allow_html=True

    )

with m2:

    if st.session_state.running_trade:

        trade_status = "RUNNING"

    elif trade_allowed:

        trade_status = "SETUP READY"

    else:

        trade_status = "NO TRADE"

    st.markdown(

        metric_box(
            "Trade Status",
            trade_status
        ),

        unsafe_allow_html=True

    )


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Price",
        format_number(price)
    )

with c2:

    st.metric(
        "EMA 9",
        format_number(ema9)
    )

with c3:

    st.metric(
        "EMA 15",
        format_number(ema15)
    )

with c4:

    st.metric(
        "VWAP",
        format_number(vwap)
    )


# ============================================================
# SIGNAL REASON
# ============================================================

st.markdown(
    f"### Signal Condition\n"
    f"{signal_data['reason']}"
)


# ============================================================
# INDEX LEVELS
# ============================================================

if signal in [
    "CALL",
    "PUT"
]:

    st.markdown(
        "## 📍 Index Trade Levels"
    )

    l1, l2, l3, l4 = st.columns(4)

    with l1:

        st.metric(
            "Entry Trigger",
            format_number(
                index_levels["entry"]
            )
        )

    with l2:

        st.metric(
            "Stop Loss",
            format_number(
                index_levels["stop_loss"]
            )
        )

    with l3:

        st.metric(
            "Target 1:2",
            format_number(
                index_levels["target"]
            )
        )

    with l4:

        st.metric(
            "Risk",
            format_number(
                index_levels["risk"]
            )
        )


# ============================================================
# OPTION PREMIUM
# ============================================================

st.markdown(
    "## 💰 Live Option Premium"
)

if option_contract is not None:

    o1, o2, o3 = st.columns(3)

    with o1:

        st.metric(
            "Option",
            option_contract[
                "symbol"
            ]
        )

    with o2:

        st.metric(
            "Premium",
            format_number(
                option_contract[
                    "premium"
                ]
            )
        )

    with o3:

        st.metric(
            "Expiry",
            option_contract[
                "expiry"
            ]
        )

    if option_levels is not None:

        o1, o2, o3 = st.columns(3)

        with o1:

            st.metric(
                "Option Entry",
                format_number(
                    option_levels[
                        "entry"
                    ]
                )
            )

        with o2:

            st.metric(
                "Option SL",
                format_number(
                    option_levels[
                        "sl"
                    ]
                )
            )

        with o3:

            st.metric(
                "Option Target",
                format_number(
                    option_levels[
                        "target"
                    ]
                )
            )

else:

    st.info(
        "कोई valid option setup उपलब्ध नहीं है।"
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


# CANDLESTICK

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


# EMA 9

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA9"],

        mode="lines",

        name="EMA 9"

    )

)


# EMA 15

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA15"],

        mode="lines",

        name="EMA 15"

    )

)


# VWAP

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["VWAP"],

        mode="lines",

        name="VWAP"

    )

)


# SIGNAL LEVELS

if signal in [
    "CALL",
    "PUT"
]:

    fig.add_hline(

        y=index_levels["entry"],

        line_dash="dot",

        annotation_text=(
            f"{signal} ENTRY"
        )

    )

    fig.add_hline(

        y=index_levels["stop_loss"],

        line_dash="dash",

        annotation_text="SL"

    )

    fig.add_hline(

        y=index_levels["target"],

        line_dash="dash",

        annotation_text="TARGET 1:2"

    )

    arrow_symbol = (

        "triangle-up"

        if signal == "CALL"

        else "triangle-down"

    )

    fig.add_trace(

        go.Scatter(

            x=[
                chart_data.index[-1]
            ],

            y=[
                index_levels["entry"]
            ],

            mode="markers+text",

            marker=dict(

                size=14,

                symbol=arrow_symbol

            ),

            text=[
                f"{signal}"
            ],

            textposition="top center",

            name="Signal"

        )

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
                running[
                    "option_entry"
                ]
            )
        )

    with r3:

        st.metric(
            "Current Premium",
            format_number(
                running[
                    "last_premium"
                ]
            )
        )

    with r4:

        st.metric(
            "P/L Points",
            format_number(
                running[
                    "points"
                ]
            )
        )

    r1, r2 = st.columns(2)

    with r1:

        st.metric(
            "Fixed SL",
            format_number(
                running[
                    "option_sl"
                ]
            )
        )

    with r2:

        st.metric(
            "Target 1:2",
            format_number(
                running[
                    "option_target"
                ]
            )
        )


# ============================================================
# MANUAL CLOSE
# ============================================================

if running is not None:

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
    get_historical_stats(
        backtest_df
    )
)

h1, h2, h3, h4 = st.columns(4)

with h1:

    st.metric(
        "Total Trades",
        stats["total"]
    )

with h2:

    st.metric(
        "Wins",
        stats["wins"]
    )

with h3:

    st.metric(
        "Losses",
        stats["losses"]
    )

with h4:

    st.metric(
        "Win Rate",
        f"{stats['win_rate']:.1f}%"
    )


h1, h2, h3 = st.columns(3)

with h1:

    st.metric(
        "Breakeven",
        stats["breakeven"]
    )

with h2:

    st.metric(
        "Net Points",
        f"{stats['net_points']:.2f}"
    )

with h3:

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
        ].head(20),

        use_container_width=True

    )


# ============================================================
# LIVE CLOSED TRADES
# ============================================================

st.markdown(
    "## Recent Live Closed Trades"
)

live_stats = (
    get_live_trade_stats()
)

l1, l2, l3, l4 = st.columns(4)

with l1:

    st.metric(
        "Closed Trades",
        live_stats["closed"]
    )

with l2:

    st.metric(
        "Wins",
        live_stats["wins"]
    )

with l3:

    st.metric(
        "Win Rate",
        f"{live_stats['win_rate']:.1f}%"
    )

with l4:

    st.metric(
        "Net Premium Points",
        f"{live_stats['net_points']:.2f}"
    )


closed_trades = (
    st.session_state.closed_trades
)

if len(closed_trades) == 0:

    st.info(
        "No live closed trades yet."
    )

else:

    rows = []

    for trade in reversed(
        closed_trades[-20:]
    ):

        rows.append({

            "Signal":
                trade["signal"],

            "Strength":
                trade["strength"],

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

    live_df = pd.DataFrame(
        rows
    )

    st.dataframe(

        live_df,

        use_container_width=True

    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "9 EMA + 15 EMA Strict Pullback Strategy | "
    "Trend + Slope + Separation + Pullback + "
    "Candle Confirmation + Break Entry | "
    "Fixed 1:2 Risk:Reward"
)
