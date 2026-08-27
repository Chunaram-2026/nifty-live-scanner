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

MIN_RISK_POINTS = 8
MAX_RISK_POINTS = 15

BACKTEST_MAX_HOLD_CANDLES = 30

PULLBACK_LOOKBACK = 3
BREAK_ENTRY_LOOKAHEAD = 3

MIN_SEPARATION_FACTOR = 0.15


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

        <div class="metric-label">
            {label}
        </div>

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

        +

        data["Low"]

        +

        data["Close"]

    ) / 3

    # Daily VWAP

    data["TradeDate"] = (
        pd.to_datetime(data.index).date
    )

    volume = (

        data["Volume"]

        .replace(0, np.nan)

        .fillna(1)

    )

    tp_volume = (
        data["TypicalPrice"] * volume
    )

    cumulative_tp_volume = (

        tp_volume

        .groupby(data["TradeDate"])

        .cumsum()

    )

    cumulative_volume = (

        volume

        .groupby(data["TradeDate"])

        .cumsum()

    )

    data["VWAP"] = (

        cumulative_tp_volume

        /

        cumulative_volume

    )

    data["VWAP"] = (
        data["VWAP"]
        .fillna(data["TypicalPrice"])
    )

    # Candle body

    data["Body"] = (

        data["Close"]

        -

        data["Open"]

    ).abs()

    # Candle range

    data["Range"] = (

        data["High"]

        -

        data["Low"]

    )

    # Upper wick

    data["UpperWick"] = (

        data["High"]

        -

        data[
            ["Open", "Close"]
        ].max(axis=1)

    )

    # Lower wick

    data["LowerWick"] = (

        data[
            ["Open", "Close"]
        ].min(axis=1)

        -

        data["Low"]

    )

    # Average range

    data["AvgRange20"] = (

        data["Range"]

        .rolling(20)

        .mean()

    )

    # EMA slopes

    data["EMA9Slope"] = (
        data["EMA9"].diff()
    )

    data["EMA15Slope"] = (
        data["EMA15"].diff()
    )

    # EMA separation

    data["EMASeparation"] = (

        data["EMA9"]

        -

        data["EMA15"]

    ).abs()

    return data


# ============================================================
# NSE SESSION
# ============================================================

def get_nse_session():

    session = requests.Session()

    headers = {

        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),

        "Accept": (
            "application/json, text/plain, */*"
        ),

        "Accept-Language": (
            "en-US,en;q=0.9"
        ),

        "Referer": (
            "https://www.nseindia.com/"
        )

    }

    session.headers.update(headers)

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

        valid_dates = []

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

                    valid_dates.append(
                        (
                            expiry_date,
                            expiry
                        )
                    )

            except Exception:

                pass

        if valid_dates:

            valid_dates.sort(
                key=lambda x: x[0]
            )

            return valid_dates[0][1]

        return expiry_dates[0]

    except Exception:

        return None


# ============================================================
# FIND OPTION CONTRACT
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

        strike_step = (
            INDEX_CONFIG[index_name]["strike_step"]
        )

        atm_strike = (

            round(
                spot_price / strike_step
            )

            * strike_step

        )

        # ATM + 1 ITM + 2 ITM

        if option_type == "CE":

            preferred_strikes = [

                atm_strike,

                atm_strike - strike_step,

                atm_strike - (
                    2 * strike_step
                )

            ]

        else:

            preferred_strikes = [

                atm_strike,

                atm_strike + strike_step,

                atm_strike + (
                    2 * strike_step
                )

            ]

        available = []

        for row in records:

            if row.get("expiryDate") != expiry:
                continue

            if option_type not in row:
                continue

            strike = safe_float(
                row.get("strikePrice")
            )

            option_data = row.get(
                option_type,
                {}
            )

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

                "strike": strike,

                "premium": premium,

                "expiry": expiry,

                "option_type": option_type

            })

        for preferred_strike in preferred_strikes:

            for item in available:

                if item["strike"] == preferred_strike:

                    return {

                        "option_type":
                            option_type,

                        "strike":
                            item["strike"],

                        "premium":
                            item["premium"],

                        "expiry":
                            expiry,

                        "symbol":
                            f"{index_name} "
                            f"{int(item['strike'])} "
                            f"{option_type}"

                    }

        return None

    except Exception:

        return None


# ============================================================
# FIND EXACT RUNNING OPTION
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

            row_strike = safe_float(
                row.get("strikePrice")
            )

            if (

                row_strike != strike

                or row.get("expiryDate") != expiry

                or option_type not in row

            ):
                continue

            option_data = row[
                option_type
            ]

            premium = safe_float(
                option_data.get(
                    "lastPrice"
                )
            )

            if premium is None:
                return None

            return {

                "strike": strike,

                "premium": premium,

                "expiry": expiry,

                "option_type": option_type

            }

        return None

    except Exception:

        return None


# ============================================================
# RECENT PULLBACK
# ============================================================

def has_bullish_pullback(data, position):

    start = max(
        0,
        position - PULLBACK_LOOKBACK
    )

    pullback_data = data.iloc[
        start:position
    ]

    if pullback_data.empty:
        return False

    for _, candle in pullback_data.iterrows():

        low = safe_float(candle["Low"])
        high = safe_float(candle["High"])

        ema9 = safe_float(candle["EMA9"])
        ema15 = safe_float(candle["EMA15"])

        if None in [low, high, ema9, ema15]:
            continue

        zone_top = max(ema9, ema15)
        zone_bottom = min(ema9, ema15)

        # Candle touches EMA zone

        if (

            low <= zone_top

            and

            high >= zone_bottom

        ):

            return True

    return False


def has_bearish_pullback(data, position):

    start = max(
        0,
        position - PULLBACK_LOOKBACK
    )

    pullback_data = data.iloc[
        start:position
    ]

    if pullback_data.empty:
        return False

    for _, candle in pullback_data.iterrows():

        low = safe_float(candle["Low"])
        high = safe_float(candle["High"])

        ema9 = safe_float(candle["EMA9"])
        ema15 = safe_float(candle["EMA15"])

        if None in [low, high, ema9, ema15]:
            continue

        zone_top = max(ema9, ema15)
        zone_bottom = min(ema9, ema15)

        if (

            low <= zone_top

            and

            high >= zone_bottom

        ):

            return True

    return False


# ============================================================
# SIGNAL CALCULATION
# ============================================================

def calculate_signal_at(data, position):

    if position < 25:
        return None

    last = data.iloc[position]

    previous = data.iloc[position - 1]

    price = safe_float(last["Close"])

    open_price = safe_float(last["Open"])

    high = safe_float(last["High"])

    low = safe_float(last["Low"])

    ema9 = safe_float(last["EMA9"])

    ema15 = safe_float(last["EMA15"])

    vwap = safe_float(last["VWAP"])

    ema9_slope = safe_float(
        last["EMA9Slope"],
        0
    )

    ema15_slope = safe_float(
        last["EMA15Slope"],
        0
    )

    separation = safe_float(
        last["EMASeparation"],
        0
    )

    avg_range = safe_float(
        last["AvgRange20"],
        0
    )

    body = safe_float(
        last["Body"],
        0
    )

    candle_range = safe_float(
        last["Range"],
        0
    )

    previous_high = safe_float(
        previous["High"]
    )

    previous_low = safe_float(
        previous["Low"]
    )

    # --------------------------------------------------------
    # EMA SEPARATION
    #
    # Flat / very close EMA = no trade
    # --------------------------------------------------------

    minimum_separation = max(

        1.0,

        avg_range *
        MIN_SEPARATION_FACTOR

    )

    ema_separated = (
        separation >= minimum_separation
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    bullish_trend = (

        price > ema15

        and

        ema9 > ema15

        and

        ema9_slope > 0

        and

        ema15_slope >= 0

    )

    bearish_trend = (

        price < ema15

        and

        ema9 < ema15

        and

        ema9_slope < 0

        and

        ema15_slope <= 0

    )

    # --------------------------------------------------------
    # VWAP
    #
    # Soft confirmation
    # --------------------------------------------------------

    bullish_vwap = (

        price > vwap

        or

        ema9 > vwap

    )

    bearish_vwap = (

        price < vwap

        or

        ema9 < vwap

    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    bullish_pullback = (
        has_bullish_pullback(
            data,
            position
        )
    )

    bearish_pullback = (
        has_bearish_pullback(
            data,
            position
        )
    )

    # --------------------------------------------------------
    # CONFIRMATION CANDLE
    # --------------------------------------------------------

    bullish_candle = (
        price > open_price
    )

    bearish_candle = (
        price < open_price
    )

    valid_body = (

        candle_range > 0

        and

        body >= (
            candle_range * 0.25
        )

    )

    # Confirmation close should show momentum

    bullish_confirmation = (

        bullish_candle

        and

        valid_body

        and

        price > ema9

    )

    bearish_confirmation = (

        bearish_candle

        and

        valid_body

        and

        price < ema9

    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    bullish_momentum = (
        price > previous_high
    )

    bearish_momentum = (
        price < previous_low
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = "WAIT"

    strength = "LOW"

    reason = []

    call_conditions = [

        bullish_trend,

        ema_separated,

        bullish_pullback,

        bullish_confirmation,

        bullish_vwap

    ]

    put_conditions = [

        bearish_trend,

        ema_separated,

        bearish_pullback,

        bearish_confirmation,

        bearish_vwap

    ]

    call_score = sum(call_conditions)

    put_score = sum(put_conditions)

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if call_score >= 5:

        signal = "CALL"

        if bullish_momentum:

            strength = "STRONG"

        else:

            strength = "MEDIUM"

        reason = [

            "EMA bullish trend",

            "EMA separation",

            "Pullback confirmed",

            "Bullish confirmation",

            "VWAP confirmation"

        ]

        if bullish_momentum:

            reason.append(
                "Momentum breakout"
            )

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    elif put_score >= 5:

        signal = "PUT"

        if bearish_momentum:

            strength = "STRONG"

        else:

            strength = "MEDIUM"

        reason = [

            "EMA bearish trend",

            "EMA separation",

            "Pullback confirmed",

            "Bearish confirmation",

            "VWAP confirmation"

        ]

        if bearish_momentum:

            reason.append(
                "Momentum breakdown"
            )

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    else:

        if not ema_separated:

            reason.append(
                "EMA flat / too close"
            )

        elif not (
            bullish_pullback
            or bearish_pullback
        ):

            reason.append(
                "Valid pullback not found"
            )

        else:

            reason.append(
                "Setup confirmation pending"
            )

    return {

        "signal": signal,

        "strength": strength,

        "price": price,

        "ema9": ema9,

        "ema15": ema15,

        "vwap": vwap,

        "call_score": call_score,

        "put_score": put_score,

        "reason": " | ".join(reason),

        "time": data.index[position],

        "signal_high": high,

        "signal_low": low

    }


# ============================================================
# CURRENT SIGNAL
#
# LAST CLOSED CANDLE
# ============================================================

def get_signal(data):

    if len(data) < 30:
        return None

    # Use previous candle
    # Current candle can still change

    position = len(data) - 2

    return calculate_signal_at(
        data,
        position
    )


# ============================================================
# INDEX LEVELS
# ============================================================

def calculate_index_levels(
    signal_data,
    candle
):

    signal = signal_data["signal"]

    entry = signal_data["price"]

    candle_range = (

        safe_float(candle["High"])

        -

        safe_float(candle["Low"])

    )

    risk = max(

        MIN_RISK_POINTS,

        min(

            MAX_RISK_POINTS,

            candle_range * 0.50

        )

    )

    if signal == "CALL":

        stop_loss = entry - risk

        target = (

            entry

            +

            risk * RISK_REWARD

        )

    elif signal == "PUT":

        stop_loss = entry + risk

        target = (

            entry

            -

            risk * RISK_REWARD

        )

    else:

        return {

            "entry": entry,

            "stop_loss": None,

            "target": None,

            "risk": None

        }

    return {

        "entry": entry,

        "stop_loss": stop_loss,

        "target": target,

        "risk": risk

    }


# ============================================================
# OPTION LEVELS
# ============================================================

def calculate_option_levels(option_premium):

    risk_percent = 0.20

    risk_amount = (

        option_premium *

        risk_percent

    )

    option_sl = (

        option_premium

        -

        risk_amount

    )

    option_target = (

        option_premium

        +

        risk_amount *
        RISK_REWARD

    )

    return {

        "entry": option_premium,

        "sl": option_sl,

        "target": option_target,

        "risk": risk_amount

    }


# ============================================================
# CREATE LIVE TRADE
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

        "option_risk":
            option_levels["risk"],

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

    trade = st.session_state.running_trade

    if trade is None:
        return

    entry = trade["option_entry"]

    exit_price = trade.get(
        "exit_premium"
    )

    if exit_price is None:

        exit_price = trade[
            "last_premium"
        ]

    trade["points"] = (

        exit_price

        -

        entry

    )

    trade["status"] = "CLOSED"

    trade["exit_time"] = datetime.now()

    trade["exit_premium"] = exit_price

    st.session_state.closed_trades.append(
        trade.copy()
    )

    st.session_state.running_trade = None


# ============================================================
# UPDATE RUNNING TRADE
# ============================================================

def update_running_trade(

    current_index_price,

    current_option_contract

):

    trade = st.session_state.running_trade

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

    # Option target

    if premium >= trade["option_target"]:

        trade["exit_reason"] = (
            "OPTION TARGET HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # Option stop loss

    if premium <= trade["option_sl"]:

        trade["exit_reason"] = (
            "OPTION STOP LOSS HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # Index target

    if (

        trade["signal"] == "CALL"

        and

        current_index_price >=
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

        current_index_price <=
        trade["index_target"]

    ):

        trade["exit_reason"] = (
            "INDEX TARGET HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # Index stop loss

    if (

        trade["signal"] == "CALL"

        and

        current_index_price <=
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

        current_index_price >=
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

    while position < len(data) - 5:

        signal_data = (
            calculate_signal_at(
                data,
                position
            )
        )

        if signal_data is None:

            position += 1

            continue

        signal = signal_data["signal"]

        if signal not in ["CALL", "PUT"]:

            position += 1

            continue

        # ----------------------------------------------------
        # BREAK ENTRY
        #
        # Entry only after confirmation candle breaks
        # ----------------------------------------------------

        signal_candle = data.iloc[
            position
        ]

        signal_high = safe_float(
            signal_candle["High"]
        )

        signal_low = safe_float(
            signal_candle["Low"]
        )

        entry_position = None

        entry = None

        entry_end = min(

            position +
            BREAK_ENTRY_LOOKAHEAD,

            len(data) - 1

        )

        for future_position in range(

            position + 1,

            entry_end + 1

        ):

            future = data.iloc[
                future_position
            ]

            future_open = safe_float(
                future["Open"]
            )

            future_high = safe_float(
                future["High"]
            )

            future_low = safe_float(
                future["Low"]
            )

            # CALL breakout

            if signal == "CALL":

                if future_high > signal_high:

                    entry_position = future_position

                    if future_open > signal_high:

                        entry = future_open

                    else:

                        entry = signal_high

                    break

            # PUT breakdown

            else:

                if future_low < signal_low:

                    entry_position = future_position

                    if future_open < signal_low:

                        entry = future_open

                    else:

                        entry = signal_low

                    break

        # No break = no trade

        if entry_position is None:

            position += 1

            continue

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        candle_range = (

            signal_high

            -

            signal_low

        )

        risk = max(

            MIN_RISK_POINTS,

            min(

                MAX_RISK_POINTS,

                candle_range * 0.50

            )

        )

        if signal == "CALL":

            sl = entry - risk

            target = (

                entry

                +

                risk * RISK_REWARD

            )

        else:

            sl = entry + risk

            target = (

                entry

                -

                risk * RISK_REWARD

            )

        # ----------------------------------------------------
        # EXIT SEARCH
        # ----------------------------------------------------

        exit_found = False

        end_position = min(

            entry_position +

            BACKTEST_MAX_HOLD_CANDLES,

            len(data) - 1

        )

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

            # CALL

            if signal == "CALL":

                # Conservative:
                # SL checked first

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

            # PUT

            else:

                # Conservative:
                # SL checked first

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
        # POINTS
        # ----------------------------------------------------

        if signal == "CALL":

            points = (
                exit_price - entry
            )

        else:

            points = (
                entry - exit_price
            )

        trades.append({

            "Signal":
                signal,

            "Strength":
                signal_data["strength"],

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
                round(exit_price, 2),

            "Points":
                round(points, 2),

            "Exit Reason":
                exit_reason

        })

        # ----------------------------------------------------
        # NO OVERLAPPING TRADE
        # ----------------------------------------------------

        position = future_position + 1

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
        backtest_df["Points"].sum()
    )

    return {

        "total": total,

        "wins": wins,

        "losses": losses,

        "breakeven": breakeven,

        "win_rate": win_rate,

        "net_points": net_points

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

        if x.get("points", 0) > 0

    ])

    losses = len([

        x for x in trades

        if x.get("points", 0) < 0

    ])

    closed = len(trades)

    net_points = sum(

        x.get("points", 0)

        for x in trades

    )

    win_rate = (

        wins / closed * 100

    )

    return {

        "closed": closed,

        "wins": wins,

        "losses": losses,

        "win_rate": win_rate,

        "net_points": net_points

    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 Personal 9-15 EMA Scalping Scanner"
)

st.caption(
    "Strict Pullback Strategy | "
    "EMA Trend + Slope + Separation | "
    "Pullback + Candle Confirmation | "
    "Break Entry | VWAP Filter | "
    "ATM/ITM Options | 1:2 Risk:Reward"
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


signal = signal_data["signal"]

strength = signal_data["strength"]

price = signal_data["price"]

ema9 = signal_data["ema9"]

ema15 = signal_data["ema15"]

vwap = signal_data["vwap"]


# ============================================================
# INDEX LEVELS
# ============================================================

signal_position = (
    len(data) - 2
)

last_closed_candle = data.iloc[
    signal_position
]

index_levels = (
    calculate_index_levels(

        signal_data,

        last_closed_candle

    )
)


# ============================================================
# OPTION DATA
# ============================================================

option_contract = None

option_levels = None


if signal in ["CALL", "PUT"]:

    option_chain = get_option_chain(
        index_name
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

if st.session_state.running_trade is not None:

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
# CREATE LIVE TRADE
# ============================================================

trade_allowed = (

    signal in ["CALL", "PUT"]

    and

    strength in [
        "MEDIUM",
        "STRONG"
    ]

)


if (

    trade_allowed

    and

    st.session_state.running_trade is None

    and

    option_contract is not None

    and

    option_levels is not None

):

    signal_time = str(
        signal_data["time"]
    )

    if (

        st.session_state.last_trade_signal_time

        !=

        signal_time

    ):

        new_trade = (

            create_trade(

                signal_data,

                index_levels,

                option_contract,

                option_levels

            )

        )

        st.session_state.running_trade = (
            new_trade
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

        trade_status = "READY"

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
# SIGNAL CONDITION
# ============================================================

st.markdown(
    "### Signal Condition"
)

st.write(
    signal_data["reason"]
)

st.caption(

    f"CALL Conditions: "
    f"{signal_data['call_score']}/5 | "

    f"PUT Conditions: "
    f"{signal_data['put_score']}/5"

)


# ============================================================
# INDEX LEVELS
# ============================================================

if signal in ["CALL", "PUT"]:

    st.markdown(
        "### Index Trade Levels"
    )

    l1, l2, l3, l4 = st.columns(4)

    with l1:

        st.metric(
            "Entry",
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
            "Target",
            format_number(
                index_levels["target"]
            )
        )

    with l4:

        st.metric(
            "Risk : Reward",
            "1 : 2"
        )


# ============================================================
# LIVE OPTION PREMIUM
# ============================================================

st.markdown(
    "## 💰 Live Option Premium"
)


if option_contract is not None:

    o1, o2 = st.columns(2)

    with o1:

        st.markdown(

            metric_box(

                "Selected Option",

                option_contract["symbol"]

            ),

            unsafe_allow_html=True

        )

    with o2:

        st.markdown(

            metric_box(

                "Expiry",

                option_contract["expiry"]

            ),

            unsafe_allow_html=True

        )

    o1, o2 = st.columns(2)

    with o1:

        st.markdown(

            metric_box(

                "Actual NSE Premium",

                format_number(
                    option_contract["premium"]
                )

            ),

            unsafe_allow_html=True

        )

    with o2:

        st.markdown(

            metric_box(

                "Strike",

                format_number(

                    option_contract["strike"],

                    0

                )

            ),

            unsafe_allow_html=True

        )

    if option_levels is not None:

        o1, o2 = st.columns(2)

        with o1:

            st.markdown(

                metric_box(

                    "Fixed Option SL",

                    format_number(
                        option_levels["sl"]
                    )

                ),

                unsafe_allow_html=True

            )

        with o2:

            st.markdown(

                metric_box(

                    "Option Target (1:2)",

                    format_number(
                        option_levels["target"]
                    )

                ),

                unsafe_allow_html=True

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

chart_data = data.tail(150).copy()

fig = go.Figure()


# Candlestick

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

        name="EMA 9",

        line=dict(
            width=1.5
        )

    )

)


# EMA 15

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


# VWAP

fig.add_trace(

    go.Scatter(

        x=chart_data.index,

        y=chart_data["VWAP"],

        mode="lines",

        name="VWAP",

        line=dict(
            width=1.5
        )

    )

)


# Current signal levels

if signal in ["CALL", "PUT"]:

    entry = index_levels["entry"]

    sl = index_levels["stop_loss"]

    target = index_levels["target"]

    fig.add_hline(

        y=entry,

        line_dash="dot",

        annotation_text=(
            f"{signal} ENTRY "
            f"{entry:.2f}"
        )

    )

    fig.add_hline(

        y=sl,

        line_dash="dash",

        annotation_text=(
            f"SL {sl:.2f}"
        )

    )

    fig.add_hline(

        y=target,

        line_dash="dash",

        annotation_text=(
            f"TARGET 1:2 "
            f"{target:.2f}"
        )

    )

    arrow_symbol = (

        "triangle-up"

        if signal == "CALL"

        else "triangle-down"

    )

    fig.add_trace(

        go.Scatter(

            x=[
                data.index[
                    signal_position
                ]
            ],

            y=[
                entry
            ],

            mode="markers+text",

            marker=dict(

                size=14,

                symbol=arrow_symbol

            ),

            text=[
                f"{signal} ENTRY"
            ],

            textposition="top center",

            name=f"{signal} ENTRY"

        )

    )


# Running trade levels

running = (
    st.session_state.running_trade
)

if running is not None:

    fig.add_hline(

        y=running["index_entry"],

        line_dash="dot",

        annotation_text=(
            f"RUNNING "
            f"{running['signal']} ENTRY"
        )

    )

    fig.add_hline(

        y=running["index_sl"],

        line_dash="dash",

        annotation_text="RUNNING SL"

    )

    fig.add_hline(

        y=running["index_target"],

        line_dash="dash",

        annotation_text="RUNNING TARGET 1:2"

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

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(

            metric_box(
                "Signal",
                running["signal"]
            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(
                "Strength",
                running["strength"]
            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(
                "Option",
                running["option_symbol"]
            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(

                "Option Entry",

                format_number(
                    running["option_entry"]
                )

            ),

            unsafe_allow_html=True

        )

    with r2:

        st.markdown(

            metric_box(

                "Current Premium",

                format_number(
                    running["last_premium"]
                )

            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(

                "Fixed SL",

                format_number(
                    running["option_sl"]
                )

            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(

                "Target 1:2",

                format_number(
                    running["option_target"]
                )

            ),

            unsafe_allow_html=True

        )

        st.markdown(

            metric_box(

                "P/L Points",

                format_number(
                    running["points"]
                )

            ),

            unsafe_allow_html=True

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
    run_historical_backtest(data)
)

historical_stats = (
    get_historical_stats(backtest_df)
)


h1, h2, h3, h4 = st.columns(4)


with h1:

    st.metric(
        "Total Trades",
        historical_stats["total"]
    )


with h2:

    st.metric(
        "Wins",
        historical_stats["wins"]
    )


with h3:

    st.metric(
        "Losses",
        historical_stats["losses"]
    )


with h4:

    st.metric(

        "Win Rate",

        f"{historical_stats['win_rate']:.1f}%"

    )


h1, h2, h3 = st.columns(3)


with h1:

    st.metric(
        "Breakeven",
        historical_stats["breakeven"]
    )


with h2:

    st.metric(

        "Net Points",

        f"{historical_stats['net_points']:.2f}"

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
        "इस strict strategy में अभी valid trade नहीं मिला।"
    )

else:

    st.dataframe(

        backtest_df.iloc[::-1].head(10),

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
        closed_trades[-10:]
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

    live_df = pd.DataFrame(rows)

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
    "VWAP Filter | Fixed 1:2 Risk:Reward"
)
