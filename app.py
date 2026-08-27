import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import time
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal Scalping Scanner",
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

div[data-testid="stMetric"] {
    background: #172231;
    border: 1px solid #26384b;
    padding: 12px;
    border-radius: 10px;
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

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 8px;
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

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

if "last_trade_signal_time" not in st.session_state:
    st.session_state.last_trade_signal_time = None


# ============================================================
# CONSTANTS
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
        "nse": "SENSEX",
        "strike_step": 100
    }
}


TIMEFRAME_CONFIG = {
    "1m": {
        "interval": "1m",
        "period": "1d"
    },
    "2m": {
        "interval": "2m",
        "period": "5d"
    },
    "3m": {
        "interval": "5m",
        "period": "5d"
    },
    "5m": {
        "interval": "5m",
        "period": "5d"
    },
    "15m": {
        "interval": "15m",
        "period": "5d"
    },
    "1h": {
        "interval": "60m",
        "period": "1mo"
    },
    "2h": {
        "interval": "60m",
        "period": "1mo"
    },
    "1d": {
        "interval": "1d",
        "period": "6mo"
    },
    "1wk": {
        "interval": "1wk",
        "period": "2y"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "")

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

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=config["period"],
        interval=config["interval"],
        auto_adjust=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df.columns = [
        str(x).capitalize()
        for x in df.columns
    ]

    if "Datetime" in df.columns:
        df = df.set_index("Datetime")

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    data = df.copy()

    # EMA 9
    data["EMA9"] = (
        data["Close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # EMA 15
    data["EMA15"] = (
        data["Close"]
        .ewm(span=15, adjust=False)
        .mean()
    )

    # Typical Price
    data["TypicalPrice"] = (
        data["High"]
        + data["Low"]
        + data["Close"]
    ) / 3

    # VWAP
    volume = data["Volume"].replace(0, np.nan)

    cumulative_tp_volume = (
        data["TypicalPrice"] * volume
    ).cumsum()

    cumulative_volume = volume.cumsum()

    data["VWAP"] = (
        cumulative_tp_volume /
        cumulative_volume
    )

    # fallback
    data["VWAP"] = (
        data["VWAP"]
        .fillna(data["TypicalPrice"])
    )

    # Candle body
    data["Body"] = (
        data["Close"] -
        data["Open"]
    ).abs()

    # Candle range
    data["Range"] = (
        data["High"] -
        data["Low"]
    ).replace(0, np.nan)

    # Upper wick
    data["UpperWick"] = (
        data["High"] -
        data[["Open", "Close"]].max(axis=1)
    )

    # Lower wick
    data["LowerWick"] = (
        data[["Open", "Close"]].min(axis=1) -
        data["Low"]
    )

    return data


# ============================================================
# OPTION CHAIN
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
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9"
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


@st.cache_data(ttl=20)
def get_option_chain(index_name):

    nse_symbol = INDEX_CONFIG[index_name]["nse"]

    session = get_nse_session()

    try:

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

        data = response.json()

        return data

    except Exception:
        return None


# ============================================================
# EXPIRY DATE
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

                date_obj = datetime.strptime(
                    expiry,
                    "%d-%b-%Y"
                ).date()

                if date_obj >= today:
                    valid_dates.append(
                        (date_obj, expiry)
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
# FIND ATM / ITM OPTION
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

        strike_step = INDEX_CONFIG[
            index_name
        ]["strike_step"]

        atm_strike = round(
            spot_price / strike_step
        ) * strike_step

        # ----------------------------------------------------
        # IMPORTANT:
        # User requested:
        # OTM option नहीं चाहिए
        #
        # CALL:
        # ATM या थोड़ा ITM
        #
        # PUT:
        # ATM या थोड़ा ITM
        # ----------------------------------------------------

        if option_type == "CE":

            preferred_strikes = [
                atm_strike,
                atm_strike - strike_step,
                atm_strike - (2 * strike_step)
            ]

        else:

            preferred_strikes = [
                atm_strike,
                atm_strike + strike_step,
                atm_strike + (2 * strike_step)
            ]

        available = []

        for row in records:

            if row.get("expiryDate") != expiry:
                continue

            strike = safe_float(
                row.get("strikePrice")
            )

            if strike is None:
                continue

            if option_type not in row:
                continue

            option_data = row.get(
                option_type,
                {}
            )

            premium = safe_float(
                option_data.get("lastPrice")
            )

            if premium is None:
                continue

            if premium <= 0:
                continue

            available.append({
                "strike": strike,
                "premium": premium,
                "data": option_data,
                "expiry": expiry
            })

        if not available:
            return None

        # Prefer ATM / ITM only
        for strike in preferred_strikes:

            matches = [
                x for x in available
                if x["strike"] == strike
            ]

            if matches:

                selected = matches[0]

                option_data = selected["data"]

                return {
                    "option_type": option_type,
                    "strike": selected["strike"],
                    "premium": selected["premium"],
                    "expiry": selected["expiry"],
                    "symbol": (
                        f"{index_name} "
                        f"{int(selected['strike'])} "
                        f"{option_type}"
                    ),
                    "oi": safe_float(
                        option_data.get(
                            "openInterest"
                        ),
                        0
                    ),
                    "volume": safe_float(
                        option_data.get(
                            "totalTradedVolume"
                        ),
                        0
                    )
                }

        # Fallback nearest suitable ITM / ATM
        if option_type == "CE":

            valid = [
                x for x in available
                if x["strike"] <= spot_price
            ]

        else:

            valid = [
                x for x in available
                if x["strike"] >= spot_price
            ]

        if valid:

            valid.sort(
                key=lambda x:
                abs(
                    x["strike"] -
                    atm_strike
                )
            )

            selected = valid[0]

            option_data = selected["data"]

            return {
                "option_type": option_type,
                "strike": selected["strike"],
                "premium": selected["premium"],
                "expiry": selected["expiry"],
                "symbol": (
                    f"{index_name} "
                    f"{int(selected['strike'])} "
                    f"{option_type}"
                ),
                "oi": safe_float(
                    option_data.get(
                        "openInterest"
                    ),
                    0
                ),
                "volume": safe_float(
                    option_data.get(
                        "totalTradedVolume"
                    ),
                    0
                )
            }

        return None

    except Exception:
        return None


# ============================================================
# SIGNAL CONDITIONS
# ============================================================

def get_signal(data):

    if data is None:
        return None

    if len(data) < 20:
        return None

    last = data.iloc[-1]
    previous = data.iloc[-2]

    price = safe_float(last["Close"])
    ema9 = safe_float(last["EMA9"])
    ema15 = safe_float(last["EMA15"])
    vwap = safe_float(last["VWAP"])

    previous_close = safe_float(
        previous["Close"]
    )

    previous_ema9 = safe_float(
        previous["EMA9"]
    )

    previous_ema15 = safe_float(
        previous["EMA15"]
    )

    candle_range = (
        safe_float(last["High"]) -
        safe_float(last["Low"])
    )

    body = abs(
        safe_float(last["Close"]) -
        safe_float(last["Open"])
    )

    lower_wick = safe_float(
        last["LowerWick"],
        0
    )

    upper_wick = safe_float(
        last["UpperWick"],
        0
    )

    if candle_range is None:
        candle_range = 0

    # --------------------------------------------------------
    # TREND CONDITIONS
    # --------------------------------------------------------

    bullish_trend = (
        price > ema9 and
        ema9 > ema15 and
        price > vwap
    )

    bearish_trend = (
        price < ema9 and
        ema9 < ema15 and
        price < vwap
    )

    # --------------------------------------------------------
    # EMA CROSS
    # --------------------------------------------------------

    bullish_cross = (
        previous_ema9 <= previous_ema15
        and
        ema9 > ema15
    )

    bearish_cross = (
        previous_ema9 >= previous_ema15
        and
        ema9 < ema15
    )

    # --------------------------------------------------------
    # CANDLE DIRECTION
    # --------------------------------------------------------

    bullish_candle = (
        price >
        safe_float(last["Open"])
    )

    bearish_candle = (
        price <
        safe_float(last["Open"])
    )

    # --------------------------------------------------------
    # LIQUIDITY SWEEP / REJECTION
    # --------------------------------------------------------

    bullish_rejection = (
        lower_wick >
        body * 1.2
        and
        price >
        safe_float(last["Open"])
    )

    bearish_rejection = (
        upper_wick >
        body * 1.2
        and
        price <
        safe_float(last["Open"])
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    recent_high = (
        data["High"]
        .iloc[-6:-1]
        .max()
    )

    recent_low = (
        data["Low"]
        .iloc[-6:-1]
        .min()
    )

    bullish_breakout = (
        price > recent_high
    )

    bearish_breakdown = (
        price < recent_low
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    call_score = 0
    put_score = 0

    if bullish_trend:
        call_score += 3

    if bearish_trend:
        put_score += 3

    if bullish_candle:
        call_score += 1

    if bearish_candle:
        put_score += 1

    if bullish_cross:
        call_score += 2

    if bearish_cross:
        put_score += 2

    if bullish_rejection:
        call_score += 2

    if bearish_rejection:
        put_score += 2

    if bullish_breakout:
        call_score += 2

    if bearish_breakdown:
        put_score += 2

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    signal = "WAIT"
    strength = "LOW"
    reason = []

    if call_score >= 5:

        signal = "CALL"

        if call_score >= 7:
            strength = "STRONG"

        reason.append(
            "Bullish EMA + VWAP structure"
        )

        if bullish_rejection:
            reason.append(
                "Bullish rejection"
            )

        if bullish_breakout:
            reason.append(
                "Breakout"
            )

    elif put_score >= 5:

        signal = "PUT"

        if put_score >= 7:
            strength = "STRONG"

        reason.append(
            "Bearish EMA + VWAP structure"
        )

        if bearish_rejection:
            reason.append(
                "Bearish rejection"
            )

        if bearish_breakdown:
            reason.append(
                "Breakdown"
            )

    else:

        reason.append(
            "Signal conditions not strong enough"
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
        "time": data.index[-1]
    }


# ============================================================
# INDEX SL / TARGET CALCULATION
# ============================================================

def calculate_index_levels(
    signal_data,
    data
):

    signal = signal_data["signal"]
    entry = signal_data["price"]

    last = data.iloc[-1]

    candle_range = (
        safe_float(last["High"]) -
        safe_float(last["Low"])
    )

    # Maximum 10-15 point style risk
    risk = max(
        8,
        min(
            15,
            candle_range * 0.50
        )
    )

    if signal == "CALL":

        stop_loss = entry - risk

        target1 = entry + risk

        target2 = entry + (
            risk * 2
        )

    elif signal == "PUT":

        stop_loss = entry + risk

        target1 = entry - risk

        target2 = entry - (
            risk * 2
        )

    else:

        return {
            "entry": entry,
            "stop_loss": None,
            "target1": None,
            "target2": None,
            "risk": None
        }

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "risk": risk
    }


# ============================================================
# OPTION PREMIUM SL / TARGET
# ============================================================

def calculate_option_levels(
    option_premium
):

    # --------------------------------------------------------
    # OPTION PREMIUM RISK
    #
    # Premium SL = 20%
    #
    # Target 1 = +20%  -> 1:1
    # Target 2 = +40%  -> 1:2
    #
    # At Target 1:
    # SL moves to Entry Price
    # --------------------------------------------------------

    risk_percent = 0.20

    risk_amount = (
        option_premium *
        risk_percent
    )

    option_sl = (
        option_premium -
        risk_amount
    )

    target1 = (
        option_premium +
        risk_amount
    )

    target2 = (
        option_premium +
        risk_amount * 2
    )

    return {
        "entry": option_premium,
        "sl": option_sl,
        "target1": target1,
        "target2": target2,
        "risk": risk_amount
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

    if option_contract is None:
        return None

    return {
        "signal": signal_data["signal"],

        "status": "RUNNING",

        "entry_time":
            datetime.now(),

        "index_entry":
            index_levels["entry"],

        "index_sl":
            index_levels["stop_loss"],

        "index_target1":
            index_levels["target1"],

        "index_target2":
            index_levels["target2"],

        "index_risk":
            index_levels["risk"],

        # -----------------------------------------
        # OPTION
        # -----------------------------------------

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

        "option_original_sl":
            option_levels["sl"],

        "option_target1":
            option_levels["target1"],

        "option_target2":
            option_levels["target2"],

        "option_risk":
            option_levels["risk"],

        # -----------------------------------------
        # TRAILING
        # -----------------------------------------

        "target1_hit":
            False,

        "trailing_active":
            False,

        "trailing_sl":
            option_levels["sl"],

        "exit_reason":
            None,

        "exit_premium":
            None,

        "last_premium":
            option_levels["entry"],

        "points":
            0
    }


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

    current_option_premium = (
        current_option_contract["premium"]
    )

    trade["last_premium"] = (
        current_option_premium
    )

    # --------------------------------------------------------
    # OPTION PROFIT / LOSS
    # --------------------------------------------------------

    trade["points"] = (
        current_option_premium -
        trade["option_entry"]
    )

    # --------------------------------------------------------
    # TARGET 1 HIT
    #
    # IMPORTANT:
    #
    # When 1:1 achieved:
    #
    # Trailing SL -> Entry Price
    # --------------------------------------------------------

    if (
        not trade["target1_hit"]
        and
        current_option_premium >=
        trade["option_target1"]
    ):

        trade["target1_hit"] = True

        trade["trailing_active"] = True

        # BREAKEVEN STOP LOSS
        trade["trailing_sl"] = (
            trade["option_entry"]
        )

        trade["option_sl"] = (
            trade["option_entry"]
        )

    # --------------------------------------------------------
    # TARGET 2
    # --------------------------------------------------------

    if (
        current_option_premium >=
        trade["option_target2"]
    ):

        trade["exit_reason"] = (
            "OPTION TARGET 2 HIT"
        )

        trade["exit_premium"] = (
            current_option_premium
        )

        close_running_trade()

        return

    # --------------------------------------------------------
    # OPTION STOP LOSS
    #
    # If Target 1 was achieved,
    # this will now be Entry Price
    # --------------------------------------------------------

    if (
        current_option_premium <=
        trade["option_sl"]
    ):

        if trade["trailing_active"]:

            trade["exit_reason"] = (
                "TRAILING SL HIT - BREAKEVEN"
            )

        else:

            trade["exit_reason"] = (
                "OPTION STOP LOSS HIT"
            )

        trade["exit_premium"] = (
            current_option_premium
        )

        close_running_trade()

        return

    # --------------------------------------------------------
    # INDEX TARGET 2
    # --------------------------------------------------------

    if trade["signal"] == "CALL":

        if (
            current_index_price >=
            trade["index_target2"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET 2 HIT"
            )

            trade["exit_premium"] = (
                current_option_premium
            )

            close_running_trade()

            return

    if trade["signal"] == "PUT":

        if (
            current_index_price <=
            trade["index_target2"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET 2 HIT"
            )

            trade["exit_premium"] = (
                current_option_premium
            )

            close_running_trade()

            return

    # --------------------------------------------------------
    # INDEX STOP LOSS
    # --------------------------------------------------------

    if trade["signal"] == "CALL":

        if (
            current_index_price <=
            trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = (
                current_option_premium
            )

            close_running_trade()

            return

    if trade["signal"] == "PUT":

        if (
            current_index_price >=
            trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = (
                current_option_premium
            )

            close_running_trade()

            return


# ============================================================
# CLOSE TRADE
# ============================================================

def close_running_trade():

    trade = st.session_state.running_trade

    if trade is None:
        return

    exit_premium = (
        trade.get("exit_premium")
    )

    entry = trade["option_entry"]

    points = (
        exit_premium -
        entry
    )

    trade["points"] = points

    trade["status"] = "CLOSED"

    trade["exit_time"] = (
        datetime.now()
    )

    closed_trade = trade.copy()

    st.session_state.closed_trades.append(
        closed_trade
    )

    st.session_state.running_trade = None


# ============================================================
# BACKTEST STATISTICS
# ============================================================

def get_backtest_stats():

    trades = (
        st.session_state.closed_trades
    )

    if len(trades) == 0:

        return {
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0,
            "target1": 0,
            "target2": 0
        }

    wins = len([
        x for x in trades
        if x.get("points", 0) > 0
    ])

    losses = len([
        x for x in trades
        if x.get("points", 0) < 0
    ])

    breakeven = len([
        x for x in trades
        if abs(
            x.get("points", 0)
        ) < 0.01
    ])

    closed = len(trades)

    net_points = sum(
        x.get("points", 0)
        for x in trades
    )

    target1 = len([
        x for x in trades
        if x.get("target1_hit", False)
    ])

    target2 = len([
        x for x in trades
        if x.get("exit_reason") ==
        "OPTION TARGET 2 HIT"
    ])

    win_rate = 0

    if closed > 0:

        win_rate = (
            wins /
            closed
        ) * 100

    return {
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "net_points": net_points,
        "target1": target1,
        "target2": target2
    }


# ============================================================
# HEADER
# ============================================================

st.title("📈 Personal Scalping Scanner")


# ============================================================
# CONTROLS
# ============================================================

col1, col2 = st.columns(2)

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
        index=2
    )


# ============================================================
# GET MARKET DATA
# ============================================================

symbol = INDEX_CONFIG[
    index_name
]["yahoo"]

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

data = calculate_indicators(df)

signal_data = get_signal(data)

if signal_data is None:

    st.warning(
        "Signal calculate करने के लिए पर्याप्त data नहीं है।"
    )

    st.stop()


signal = signal_data["signal"]

price = signal_data["price"]

ema9 = signal_data["ema9"]

ema15 = signal_data["ema15"]

vwap = signal_data["vwap"]


# ============================================================
# INDEX LEVELS
# ============================================================

index_levels = (
    calculate_index_levels(
        signal_data,
        data
    )
)


# ============================================================
# GET OPTION CONTRACT
# ============================================================

option_contract = None
option_levels = None

if signal == "CALL":

    option_chain = get_option_chain(
        index_name
    )

    option_contract = (
        find_option_contract(
            option_chain,
            price,
            "CE",
            index_name
        )
    )

elif signal == "PUT":

    option_chain = get_option_chain(
        index_name
    )

    option_contract = (
        find_option_contract(
            option_chain,
            price,
            "PE",
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

if st.session_state.running_trade is not None:

    running = (
        st.session_state.running_trade
    )

    current_type = (
        running["option_type"]
    )

    option_chain_live = get_option_chain(
        index_name
    )

    current_option = (
        find_option_contract(
            option_chain_live,
            price,
            current_type,
            index_name
        )
    )

    update_running_trade(
        price,
        current_option
    )


# ============================================================
# CREATE NEW TRADE
# ============================================================

if (
    signal in ["CALL", "PUT"]
    and
    st.session_state.running_trade is None
    and
    option_contract is not None
    and
    option_levels is not None
):

    last_signal = (
        st.session_state.last_signal
    )

    signal_time = str(
        signal_data["time"]
    )

    # New trade only if signal changes
    if (
        last_signal != signal
        or
        st.session_state.last_trade_signal_time
        != signal_time
    ):

        new_trade = create_trade(
            signal_data,
            index_levels,
            option_contract,
            option_levels
        )

        if new_trade is not None:

            st.session_state.running_trade = (
                new_trade
            )

            st.session_state.last_signal = (
                signal
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
# TOP METRICS
# ============================================================

st.markdown("### Current Signal")

m1, m2 = st.columns(2)

with m1:

    st.markdown(
        metric_box(
            "Signal",
            f"{signal} ({signal_data['strength']})",
            signal_class
        ),
        unsafe_allow_html=True
    )

with m2:

    trade_status = (
        "RUNNING"
        if st.session_state.running_trade
        else "WAIT"
    )

    st.markdown(
        metric_box(
            "Trade Status",
            trade_status
        ),
        unsafe_allow_html=True
    )


c1, c2 = st.columns(2)

with c1:

    st.markdown(
        metric_box(
            "Price",
            format_number(price)
        ),
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        metric_box(
            "EMA 9",
            format_number(ema9)
        ),
        unsafe_allow_html=True
    )


c1, c2 = st.columns(2)

with c1:

    st.markdown(
        metric_box(
            "EMA 15",
            format_number(ema15)
        ),
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        metric_box(
            "VWAP (OHLC/4)",
            format_number(vwap)
        ),
        unsafe_allow_html=True
    )


c1, c2 = st.columns(2)

with c1:

    index_sl_text = (
        format_number(
            index_levels["stop_loss"]
        )
        if index_levels["stop_loss"]
        else "-"
    )

    st.markdown(
        metric_box(
            "Index Stop Loss",
            index_sl_text
        ),
        unsafe_allow_html=True
    )

with c2:

    risk_text = (
        format_number(
            index_levels["risk"]
        )
        if index_levels["risk"]
        else "-"
    )

    st.markdown(
        metric_box(
            "Risk Points",
            risk_text
        ),
        unsafe_allow_html=True
    )


c1, c2 = st.columns(2)

with c1:

    target1_text = (
        format_number(
            index_levels["target1"]
        )
        if index_levels["target1"]
        else "-"
    )

    st.markdown(
        metric_box(
            "Target 1 (1:1)",
            target1_text
        ),
        unsafe_allow_html=True
    )

with c2:

    target2_text = (
        format_number(
            index_levels["target2"]
        )
        if index_levels["target2"]
        else "-"
    )

    st.markdown(
        metric_box(
            "Target 2 (1:2)",
            target2_text
        ),
        unsafe_allow_html=True
    )


# ============================================================
# SIGNAL CONDITIONS
# ============================================================

st.markdown(
    f"**Signal Condition:** "
    f"{signal_data['reason']}"
)

st.caption(
    f"CALL Score: {signal_data['call_score']} | "
    f"PUT Score: {signal_data['put_score']}"
)


# ============================================================
# OPTION PREMIUM DETAILS
# ============================================================

st.markdown("### 🎯 Live Option Premium")

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
                    "Option Premium SL",
                    format_number(
                        option_levels["sl"]
                    )
                ),
                unsafe_allow_html=True
            )

        with o2:

            st.markdown(
                metric_box(
                    "Option Premium Target 1",
                    format_number(
                        option_levels["target1"]
                    )
                ),
                unsafe_allow_html=True
            )


        o1, o2 = st.columns(2)

        with o1:

            st.markdown(
                metric_box(
                    "Option Premium Target 2",
                    format_number(
                        option_levels["target2"]
                    )
                ),
                unsafe_allow_html=True
            )

        with o2:

            st.markdown(
                metric_box(
                    "Trailing Rule",
                    "1:1 → SL = Entry Price"
                ),
                unsafe_allow_html=True
            )

else:

    st.warning(
        "NSE option-chain से option premium अभी उपलब्ध नहीं है।"
    )


# ============================================================
# CHART
# ============================================================

st.markdown("### 📊 Index Chart")

chart_data = data.tail(150).copy()

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
        line=dict(width=1.5)
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
        line=dict(width=1.5)
    )
)


# ------------------------------------------------------------
# VWAP
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["VWAP"],
        mode="lines",
        name="VWAP",
        line=dict(width=1.5)
    )
)


# ============================================================
# CHART LINES
# ============================================================

if signal in ["CALL", "PUT"]:

    entry = index_levels["entry"]
    sl = index_levels["stop_loss"]
    t1 = index_levels["target1"]
    t2 = index_levels["target2"]

    # ENTRY
    fig.add_hline(
        y=entry,
        line_dash="dot",
        annotation_text=f"{signal} ENTRY {entry:.2f}"
    )

    # STOP LOSS
    fig.add_hline(
        y=sl,
        line_dash="dash",
        annotation_text=f"SL {sl:.2f}"
    )

    # TARGET 1
    fig.add_hline(
        y=t1,
        line_dash="dash",
        annotation_text=f"T1 {t1:.2f}"
    )

    # TARGET 2
    fig.add_hline(
        y=t2,
        line_dash="dash",
        annotation_text=f"T2 {t2:.2f}"
    )

    # --------------------------------------------------------
    # ENTRY ARROW
    # --------------------------------------------------------

    arrow_symbol = (
        "triangle-up"
        if signal == "CALL"
        else "triangle-down"
    )

    fig.add_trace(
        go.Scatter(
            x=[chart_data.index[-1]],
            y=[entry],
            mode="markers+text",
            marker=dict(
                size=16,
                symbol=arrow_symbol
            ),
            text=[signal],
            textposition="top center",
            name=f"{signal} ENTRY"
        )
    )


# ============================================================
# RUNNING TRADE LEVELS ON CHART
# ============================================================

running = (
    st.session_state.running_trade
)

if running is not None:

    fig.add_hline(
        y=running["index_entry"],
        line_dash="dot",
        annotation_text=(
            f"RUNNING {running['signal']} "
            f"ENTRY"
        )
    )

    fig.add_hline(
        y=running["index_sl"],
        line_dash="dash",
        annotation_text="INDEX SL"
    )

    if running["signal"] == "CALL":

        fig.add_hline(
            y=running["index_target1"],
            line_dash="dot",
            annotation_text="INDEX T1"
        )

        fig.add_hline(
            y=running["index_target2"],
            line_dash="dash",
            annotation_text="INDEX T2"
        )

    else:

        fig.add_hline(
            y=running["index_target1"],
            line_dash="dot",
            annotation_text="INDEX T1"
        )

        fig.add_hline(
            y=running["index_target2"],
            line_dash="dash",
            annotation_text="INDEX T2"
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

st.markdown("### 🔵 Running Trade")

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
                "Trade",
                running["signal"]
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
                "Option Entry Premium",
                format_number(
                    running["option_entry"]
                )
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            metric_box(
                "Current Premium",
                format_number(
                    running["last_premium"]
                )
            ),
            unsafe_allow_html=True
        )

    with r2:

        st.markdown(
            metric_box(
                "Current SL",
                format_number(
                    running["option_sl"]
                )
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            metric_box(
                "Target 1",
                format_number(
                    running["option_target1"]
                )
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            metric_box(
                "Target 2",
                format_number(
                    running["option_target2"]
                )
            ),
            unsafe_allow_html=True
        )

        trailing_text = (
            "ACTIVE → SL = ENTRY PRICE"
            if running["trailing_active"]
            else "Waiting for 1:1"
        )

        st.markdown(
            metric_box(
                "Trailing Stop Loss",
                trailing_text
            ),
            unsafe_allow_html=True
        )


    if running["target1_hit"]:

        st.success(
            "✅ Target 1 / 1:1 achieved — "
            "Stop Loss अब Entry Price पर है "
            "(Breakeven Trailing SL)."
        )

    else:

        st.warning(
            "⏳ 1:1 का इंतजार — "
            "Target 1 hit होते ही "
            "SL Entry Price पर shift होगा।"
        )


# ============================================================
# BACKTEST
# ============================================================

st.markdown(
    "### 📉 Backtest (Closed Trades Only)"
)

stats = get_backtest_stats()

b1, b2 = st.columns(2)

with b1:

    st.markdown(
        metric_box(
            "Closed Trades",
            stats["closed"]
        ),
        unsafe_allow_html=True
    )

with b2:

    st.markdown(
        metric_box(
            "Wins",
            stats["wins"]
        ),
        unsafe_allow_html=True
    )


b1, b2 = st.columns(2)

with b1:

    st.markdown(
        metric_box(
            "Losses",
            stats["losses"]
        ),
        unsafe_allow_html=True
    )

with b2:

    st.markdown(
        metric_box(
            "Win Rate",
            f"{stats['win_rate']:.1f}%"
        ),
        unsafe_allow_html=True
    )


b1, b2 = st.columns(2)

with b1:

    st.markdown(
        metric_box(
            "Net Premium Points",
            f"{stats['net_points']:.2f}"
        ),
        unsafe_allow_html=True
    )

with b2:

    st.markdown(
        metric_box(
            "Target 1 / Target 2",
            f"{stats['target1']} / {stats['target2']}"
        ),
        unsafe_allow_html=True
    )


# ============================================================
# RECENT CLOSED TRADES
# ============================================================

st.markdown("### Recent Closed Trades")

closed_trades = (
    st.session_state.closed_trades
)

if len(closed_trades) == 0:

    st.info(
        "No closed trades found."
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

            "Target 1":
                "YES"
                if trade["target1_hit"]
                else "NO",

            "Exit Reason":
                trade["exit_reason"]
        })

    trades_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        trades_df,
        use_container_width=True
    )


# ============================================================
# MANUAL CLOSE BUTTON
# ============================================================

if (
    st.session_state.running_trade
    is not None
):

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
# FOOTER
# ============================================================

st.caption(
    "Scalping scanner | "
    "EMA 9 + EMA 15 + VWAP | "
    "ATM/ITM Options Only | "
    "1:1 → Breakeven Trailing SL"
)
