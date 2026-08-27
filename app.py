import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
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

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

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
        "nse": "SENSEX",
        "strike_step": 100
    }
}


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

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

    try:

        df = yf.download(
            symbol,
            period=config["period"],
            interval=config["interval"],
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Fix MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):

            df.columns = [
                col[0]
                for col in df.columns
            ]

        df.columns = [
            str(col).capitalize()
            for col in df.columns
        ]

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
        .ewm(span=9, adjust=False)
        .mean()
    )

    # EMA 15
    data["EMA15"] = (
        data["Close"]
        .ewm(span=15, adjust=False)
        .mean()
    )

    # Typical price
    data["TypicalPrice"] = (
        data["High"]
        + data["Low"]
        + data["Close"]
    ) / 3

    # VWAP
    volume = data["Volume"].replace(0, np.nan)

    tp_volume = (
        data["TypicalPrice"] * volume
    )

    data["VWAP"] = (
        tp_volume.cumsum()
        /
        volume.cumsum()
    )

    data["VWAP"] = (
        data["VWAP"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(data["TypicalPrice"])
    )

    # Candle body
    data["Body"] = (
        data["Close"]
        - data["Open"]
    ).abs()

    # Upper wick
    data["UpperWick"] = (
        data["High"]
        - data[["Open", "Close"]].max(axis=1)
    )

    # Lower wick
    data["LowerWick"] = (
        data[["Open", "Close"]].min(axis=1)
        - data["Low"]
    )

    return data


# ============================================================
# NSE OPTION CHAIN
# ============================================================

def get_nse_session():

    session = requests.Session()

    session.headers.update({

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

    })

    return session


@st.cache_data(ttl=20)
def get_option_chain(index_name):

    nse_symbol = INDEX_CONFIG[index_name]["nse"]

    try:

        session = get_nse_session()

        session.get(
            "https://www.nseindia.com",
            timeout=15
        )

        time.sleep(0.5)

        url = (
            "https://www.nseindia.com/api/"
            f"option-chain-indices?symbol={nse_symbol}"
        )

        response = session.get(
            url,
            timeout=20
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

                expiry_date = datetime.strptime(
                    expiry,
                    "%d-%b-%Y"
                ).date()

                if expiry_date >= today:

                    valid_dates.append(
                        (expiry_date, expiry)
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
    index_name,
    locked_strike=None
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

        atm_strike = (
            round(
                spot_price / strike_step
            )
            * strike_step
        )

        available = []

        for row in records:

            if row.get("expiryDate") != expiry:
                continue

            if option_type not in row:
                continue

            strike = safe_float(
                row.get("strikePrice")
            )

            if strike is None:
                continue

            option_data = row.get(
                option_type,
                {}
            )

            premium = safe_float(
                option_data.get("lastPrice")
            )

            if premium is None or premium <= 0:
                continue

            available.append({

                "strike": strike,

                "premium": premium,

                "data": option_data,

                "expiry": expiry
            })

        if not available:
            return None

        # IMPORTANT:
        # Running trade must keep SAME strike
        if locked_strike is not None:

            matches = [

                x for x in available

                if x["strike"] == locked_strike

            ]

            if matches:

                selected = matches[0]

                return {

                    "option_type":
                        option_type,

                    "strike":
                        selected["strike"],

                    "premium":
                        selected["premium"],

                    "expiry":
                        selected["expiry"],

                    "symbol":
                        (
                            f"{index_name} "
                            f"{int(selected['strike'])} "
                            f"{option_type}"
                        )
                }

        # ATM / ITM preference
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

        for strike in preferred_strikes:

            matches = [

                x for x in available

                if x["strike"] == strike

            ]

            if matches:

                selected = matches[0]

                return {

                    "option_type":
                        option_type,

                    "strike":
                        selected["strike"],

                    "premium":
                        selected["premium"],

                    "expiry":
                        selected["expiry"],

                    "symbol":
                        (
                            f"{index_name} "
                            f"{int(selected['strike'])} "
                            f"{option_type}"
                        )
                }

        return None

    except Exception:

        return None


# ============================================================
# SIGNAL ENGINE
# ============================================================

def get_signal(data):

    if len(data) < 20:
        return None

    last = data.iloc[-1]

    previous = data.iloc[-2]

    price = safe_float(last["Close"])

    ema9 = safe_float(last["EMA9"])

    ema15 = safe_float(last["EMA15"])

    vwap = safe_float(last["VWAP"])

    previous_ema9 = safe_float(
        previous["EMA9"]
    )

    previous_ema15 = safe_float(
        previous["EMA15"]
    )

    body = safe_float(
        last["Body"],
        0
    )

    lower_wick = safe_float(
        last["LowerWick"],
        0
    )

    upper_wick = safe_float(
        last["UpperWick"],
        0
    )

    bullish_trend = (

        price > ema9

        and ema9 > ema15

        and price > vwap
    )

    bearish_trend = (

        price < ema9

        and ema9 < ema15

        and price < vwap
    )

    bullish_cross = (

        previous_ema9
        <= previous_ema15

        and ema9 > ema15
    )

    bearish_cross = (

        previous_ema9
        >= previous_ema15

        and ema9 < ema15
    )

    bullish_candle = (
        last["Close"]
        > last["Open"]
    )

    bearish_candle = (
        last["Close"]
        < last["Open"]
    )

    bullish_rejection = (

        lower_wick
        > body * 1.2

        and bullish_candle
    )

    bearish_rejection = (

        upper_wick
        > body * 1.2

        and bearish_candle
    )

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

    signal = "WAIT"
    strength = "LOW"
    reason = []

    if call_score >= 5:

        signal = "CALL"

        strength = (
            "STRONG"
            if call_score >= 7
            else "MEDIUM"
        )

        reason.append(
            "Bullish EMA + VWAP"
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

        strength = (
            "STRONG"
            if put_score >= 7
            else "MEDIUM"
        )

        reason.append(
            "Bearish EMA + VWAP"
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
            "Conditions not strong enough"
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

        "reason":
            " | ".join(reason),

        "time":
            data.index[-1]
    }


# ============================================================
# INDEX LEVELS
# ============================================================

def calculate_index_levels(
    signal_data,
    data
):

    signal = signal_data["signal"]

    entry = signal_data["price"]

    last = data.iloc[-1]

    candle_range = (
        safe_float(last["High"])
        -
        safe_float(last["Low"])
    )

    risk = max(
        8,
        min(
            15,
            candle_range * 0.5
        )
    )

    if signal == "CALL":

        sl = entry - risk

        t1 = entry + risk

        t2 = entry + (
            risk * 2
        )

    elif signal == "PUT":

        sl = entry + risk

        t1 = entry - risk

        t2 = entry - (
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

        "stop_loss": sl,

        "target1": t1,

        "target2": t2,

        "risk": risk
    }


# ============================================================
# OPTION PREMIUM LEVELS
# ============================================================

def calculate_option_levels(option_premium):

    risk_percent = 0.20

    risk = (
        option_premium
        * risk_percent
    )

    return {

        "entry":
            option_premium,

        "sl":
            option_premium - risk,

        "target1":
            option_premium + risk,

        "target2":
            option_premium + (
                risk * 2
            ),

        "risk":
            risk
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

        "status":
            "RUNNING",

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

        "target1_hit":
            False,

        "trailing_active":
            False,

        "highest_premium":
            option_levels["entry"],

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
# CLOSE RUNNING TRADE
# ============================================================

def close_running_trade():

    trade = (
        st.session_state.running_trade
    )

    if trade is None:
        return

    exit_premium = (
        trade["exit_premium"]
    )

    trade["points"] = (
        exit_premium
        -
        trade["option_entry"]
    )

    trade["status"] = "CLOSED"

    trade["exit_time"] = (
        datetime.now()
    )

    st.session_state.closed_trades.append(
        trade.copy()
    )

    st.session_state.running_trade = None


# ============================================================
# UPDATE LIVE TRADE
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

    # ========================================================
    # UPDATE HIGHEST PREMIUM
    # ========================================================

    if premium > trade["highest_premium"]:

        trade["highest_premium"] = premium

    # ========================================================
    # TARGET 1 = BREAKEVEN
    # ========================================================

    if (

        not trade["target1_hit"]

        and premium
        >= trade["option_target1"]

    ):

        trade["target1_hit"] = True

        trade["trailing_active"] = True

        # SL moves to entry
        trade["option_sl"] = (
            trade["option_entry"]
        )

        trade["trailing_sl"] = (
            trade["option_entry"]
        )

    # ========================================================
    # DYNAMIC TRAILING SL
    #
    # After 1:1:
    # Every new premium high
    # SL trails 10% below highest premium
    # But never below Entry
    # ========================================================

    if trade["trailing_active"]:

        highest = (
            trade["highest_premium"]
        )

        trailing_percent = 0.10

        new_trailing_sl = (
            highest
            * (1 - trailing_percent)
        )

        new_trailing_sl = max(
            new_trailing_sl,
            trade["option_entry"]
        )

        if new_trailing_sl > trade["option_sl"]:

            trade["option_sl"] = (
                new_trailing_sl
            )

            trade["trailing_sl"] = (
                new_trailing_sl
            )

    # ========================================================
    # TARGET 2
    # ========================================================

    if premium >= trade["option_target2"]:

        trade["exit_reason"] = (
            "OPTION TARGET 2 HIT"
        )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # ========================================================
    # OPTION TRAILING SL / SL
    # ========================================================

    if premium <= trade["option_sl"]:

        if trade["trailing_active"]:

            trade["exit_reason"] = (
                "TRAILING SL HIT"
            )

        else:

            trade["exit_reason"] = (
                "OPTION STOP LOSS HIT"
            )

        trade["exit_premium"] = premium

        close_running_trade()

        return

    # ========================================================
    # INDEX STOP / TARGET
    # ========================================================

    if trade["signal"] == "CALL":

        if (
            current_index_price
            <= trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = premium

            close_running_trade()

            return

        if (
            current_index_price
            >= trade["index_target2"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET 2 HIT"
            )

            trade["exit_premium"] = premium

            close_running_trade()

            return

    elif trade["signal"] == "PUT":

        if (
            current_index_price
            >= trade["index_sl"]
        ):

            trade["exit_reason"] = (
                "INDEX STOP LOSS HIT"
            )

            trade["exit_premium"] = premium

            close_running_trade()

            return

        if (
            current_index_price
            <= trade["index_target2"]
        ):

            trade["exit_reason"] = (
                "INDEX TARGET 2 HIT"
            )

            trade["exit_premium"] = premium

            close_running_trade()

            return


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

def run_backtest(data):

    trades = []

    position = None

    for i in range(20, len(data)):

        current = data.iloc[i]

        previous = data.iloc[i - 1]

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

        prev_ema9 = safe_float(
            previous["EMA9"]
        )

        prev_ema15 = safe_float(
            previous["EMA15"]
        )

        bullish_trend = (

            price > ema9

            and ema9 > ema15

            and price > vwap
        )

        bearish_trend = (

            price < ema9

            and ema9 < ema15

            and price < vwap
        )

        bullish_candle = (
            current["Close"]
            > current["Open"]
        )

        bearish_candle = (
            current["Close"]
            < current["Open"]
        )

        bullish_cross = (

            prev_ema9
            <= prev_ema15

            and ema9 > ema15
        )

        bearish_cross = (

            prev_ema9
            >= prev_ema15

            and ema9 < ema15
        )

        recent_high = (
            data["High"]
            .iloc[i - 5:i]
            .max()
        )

        recent_low = (
            data["Low"]
            .iloc[i - 5:i]
            .min()
        )

        bullish_breakout = (
            price > recent_high
        )

        bearish_breakdown = (
            price < recent_low
        )

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

        if bullish_breakout:
            call_score += 2

        if bearish_breakdown:
            put_score += 2

        # ====================================================
        # NEW ENTRY
        # ====================================================

        if position is None:

            candle_range = (
                current["High"]
                -
                current["Low"]
            )

            risk = max(
                8,
                min(
                    15,
                    candle_range * 0.5
                )
            )

            if call_score >= 5:

                position = {

                    "signal":
                        "CALL",

                    "entry_time":
                        data.index[i],

                    "entry":
                        price,

                    "sl":
                        price - risk,

                    "target1":
                        price + risk,

                    "target2":
                        price + (
                            risk * 2
                        ),

                    "risk":
                        risk,

                    "target1_hit":
                        False,

                    "highest_price":
                        price
                }

            elif put_score >= 5:

                position = {

                    "signal":
                        "PUT",

                    "entry_time":
                        data.index[i],

                    "entry":
                        price,

                    "sl":
                        price + risk,

                    "target1":
                        price - risk,

                    "target2":
                        price - (
                            risk * 2
                        ),

                    "risk":
                        risk,

                    "target1_hit":
                        False,

                    "lowest_price":
                        price
                }

        # ====================================================
        # CALL MANAGEMENT
        # ====================================================

        elif position["signal"] == "CALL":

            high = current["High"]
            low = current["Low"]

            if high > position["highest_price"]:

                position["highest_price"] = high

            # Target 1
            if (

                not position["target1_hit"]

                and high
                >= position["target1"]

            ):

                position["target1_hit"] = True

                position["sl"] = (
                    position["entry"]
                )

            # Dynamic trailing
            if position["target1_hit"]:

                trailing_sl = max(

                    position["entry"],

                    position["highest_price"]
                    -
                    position["risk"]
                )

                if trailing_sl > position["sl"]:

                    position["sl"] = trailing_sl

            # Target 2
            if high >= position["target2"]:

                position["exit"] = (
                    position["target2"]
                )

                position["exit_reason"] = (
                    "TARGET 2 HIT"
                )

                position["exit_time"] = (
                    data.index[i]
                )

                position["points"] = (
                    position["exit"]
                    -
                    position["entry"]
                )

                trades.append(
                    position.copy()
                )

                position = None

            # Stop loss
            elif low <= position["sl"]:

                position["exit"] = (
                    position["sl"]
                )

                position["exit_reason"] = (

                    "TRAILING SL HIT"

                    if position["target1_hit"]

                    else "STOP LOSS HIT"
                )

                position["exit_time"] = (
                    data.index[i]
                )

                position["points"] = (
                    position["exit"]
                    -
                    position["entry"]
                )

                trades.append(
                    position.copy()
                )

                position = None

        # ====================================================
        # PUT MANAGEMENT
        # ====================================================

        elif position["signal"] == "PUT":

            high = current["High"]
            low = current["Low"]

            if low < position["lowest_price"]:

                position["lowest_price"] = low

            # Target 1
            if (

                not position["target1_hit"]

                and low
                <= position["target1"]

            ):

                position["target1_hit"] = True

                position["sl"] = (
                    position["entry"]
                )

            # Dynamic trailing
            if position["target1_hit"]:

                trailing_sl = min(

                    position["entry"],

                    position["lowest_price"]
                    +
                    position["risk"]
                )

                if trailing_sl < position["sl"]:

                    position["sl"] = trailing_sl

            # Target 2
            if low <= position["target2"]:

                position["exit"] = (
                    position["target2"]
                )

                position["exit_reason"] = (
                    "TARGET 2 HIT"
                )

                position["exit_time"] = (
                    data.index[i]
                )

                position["points"] = (
                    position["entry"]
                    -
                    position["exit"]
                )

                trades.append(
                    position.copy()
                )

                position = None

            # Stop loss
            elif high >= position["sl"]:

                position["exit"] = (
                    position["sl"]
                )

                position["exit_reason"] = (

                    "TRAILING SL HIT"

                    if position["target1_hit"]

                    else "STOP LOSS HIT"
                )

                position["exit_time"] = (
                    data.index[i]
                )

                position["points"] = (
                    position["entry"]
                    -
                    position["exit"]
                )

                trades.append(
                    position.copy()
                )

                position = None

    return trades


# ============================================================
# HEADER
# ============================================================

st.title("📈 Personal Scalping Scanner")

st.caption(
    "EMA 9 + EMA 15 + VWAP | "
    "ATM/ITM Options | "
    "1:1 Breakeven + Dynamic Trailing SL"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    index_name = st.selectbox(
        "Select Index",
        list(INDEX_CONFIG.keys())
    )

with col2:

    timeframe = st.selectbox(
        "Select Timeframe",
        list(TIMEFRAME_CONFIG.keys()),
        index=2
    )


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
        "Signal के लिए पर्याप्त data नहीं है।"
    )

    st.stop()


signal = signal_data["signal"]

price = signal_data["price"]


# ============================================================
# INDEX LEVELS
# ============================================================

index_levels = calculate_index_levels(
    signal_data,
    data
)


# ============================================================
# OPTION CONTRACT
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

    option_contract = find_option_contract(

        option_chain,

        price,

        option_type,

        index_name
    )

    if option_contract:

        option_levels = (
            calculate_option_levels(
                option_contract["premium"]
            )
        )


# ============================================================
# UPDATE RUNNING TRADE
# ============================================================

if st.session_state.running_trade:

    running = (
        st.session_state.running_trade
    )

    option_chain_live = (
        get_option_chain(index_name)
    )

    current_option = find_option_contract(

        option_chain_live,

        price,

        running["option_type"],

        index_name,

        locked_strike=
            running["option_strike"]
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

    and st.session_state.running_trade is None

    and option_contract is not None

    and option_levels is not None

):

    signal_time = str(
        signal_data["time"]
    )

    if (

        st.session_state.last_signal
        != signal

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
# CURRENT SIGNAL
# ============================================================

st.markdown("### Current Signal")

if signal == "CALL":

    signal_class = "signal-call"

elif signal == "PUT":

    signal_class = "signal-put"

else:

    signal_class = "signal-wait"


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

    status = (

        "RUNNING"

        if st.session_state.running_trade

        else "WAIT"
    )

    st.markdown(
        metric_box(
            "Trade Status",
            status
        ),
        unsafe_allow_html=True
    )


m1, m2 = st.columns(2)

with m1:

    st.markdown(
        metric_box(
            "Price",
            format_number(price)
        ),
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        metric_box(
            "EMA 9",
            format_number(
                signal_data["ema9"]
            )
        ),
        unsafe_allow_html=True
    )


m1, m2 = st.columns(2)

with m1:

    st.markdown(
        metric_box(
            "EMA 15",
            format_number(
                signal_data["ema15"]
            )
        ),
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        metric_box(
            "VWAP",
            format_number(
                signal_data["vwap"]
            )
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

    f"CALL Score: "
    f"{signal_data['call_score']} | "

    f"PUT Score: "
    f"{signal_data['put_score']}"
)


# ============================================================
# OPTION PREMIUM
# ============================================================

st.markdown(
    "### 🎯 Live Option Premium"
)

if option_contract:

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


    if option_levels:

        o1, o2 = st.columns(2)

        with o1:

            st.markdown(
                metric_box(
                    "Option SL",
                    format_number(
                        option_levels["sl"]
                    )
                ),
                unsafe_allow_html=True
            )

        with o2:

            st.markdown(
                metric_box(
                    "Target 1 (1:1)",
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
                    "Target 2 (1:2)",
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
                    "After 1:1 → Dynamic SL"
                ),
                unsafe_allow_html=True
            )

else:

    st.warning(
        "NSE option-chain से option premium उपलब्ध नहीं है।"
    )


# ============================================================
# CHART
# ============================================================

st.markdown("### 📊 Index Chart")

chart_data = (
    data.tail(150).copy()
)

fig = go.Figure()


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


fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA9"],

        mode="lines",

        name="EMA 9"
    )
)


fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["EMA15"],

        mode="lines",

        name="EMA 15"
    )
)


fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["VWAP"],

        mode="lines",

        name="VWAP"
    )
)


if signal in ["CALL", "PUT"]:

    entry = index_levels["entry"]
    sl = index_levels["stop_loss"]
    t1 = index_levels["target1"]
    t2 = index_levels["target2"]

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

        y=t1,

        line_dash="dot",

        annotation_text=(
            f"T1 {t1:.2f}"
        )
    )

    fig.add_hline(

        y=t2,

        line_dash="dash",

        annotation_text=(
            f"T2 {t2:.2f}"
        )
    )

    arrow = (

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
                entry
            ],

            mode="markers+text",

            marker=dict(
                size=16,
                symbol=arrow
            ),

            text=[
                signal
            ],

            textposition="top center",

            name=f"{signal} ENTRY"
        )
    )


fig.update_layout(

    height=620,

    template="plotly_dark",

    xaxis_rangeslider_visible=False,

    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10
    ),

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
    "### 🔵 Running Trade"
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
                "Entry Premium",
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
                "Highest Premium",
                format_number(
                    running["highest_premium"]
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

    if running["target1_hit"]:

        st.success(
            "✅ 1:1 achieved. "
            "SL Entry पर गया और अब "
            "highest premium के साथ trail हो रहा है।"
        )

    else:

        st.warning(
            "⏳ Target 1 hit होने पर "
            "SL पहले Entry Price पर जाएगा।"
        )


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

st.markdown(
    "### 📉 Historical Backtest"
)

backtest_trades = (
    run_backtest(data)
)

total = len(
    backtest_trades
)

wins = len([

    x for x in backtest_trades

    if x["points"] > 0
])

losses = len([

    x for x in backtest_trades

    if x["points"] < 0
])

breakeven = len([

    x for x in backtest_trades

    if abs(x["points"]) < 0.01
])

win_rate = (

    wins / total * 100

    if total > 0

    else 0
)

net_points = sum(

    x["points"]

    for x in backtest_trades
)


b1, b2, b3, b4 = st.columns(4)

with b1:

    st.metric(
        "Total Trades",
        total
    )

with b2:

    st.metric(
        "Wins",
        wins
    )

with b3:

    st.metric(
        "Losses",
        losses
    )

with b4:

    st.metric(
        "Win Rate",
        f"{win_rate:.1f}%"
    )


b1, b2, b3 = st.columns(3)

with b1:

    st.metric(
        "Breakeven",
        breakeven
    )

with b2:

    st.metric(
        "Net Points",
        f"{net_points:.2f}"
    )

with b3:

    st.metric(
        "Backtest Candles",
        len(data)
    )


# ============================================================
# BACKTEST TABLE
# ============================================================

st.markdown(
    "### Recent Backtest Trades"
)

if len(backtest_trades) == 0:

    st.info(
        "इस timeframe के available data में "
        "कोई completed trade नहीं मिला।"
    )

else:

    rows = []

    for trade in reversed(
        backtest_trades[-20:]
    ):

        rows.append({

            "Signal":
                trade["signal"],

            "Entry Time":
                str(
                    trade["entry_time"]
                ),

            "Entry":
                round(
                    trade["entry"],
                    2
                ),

            "Exit":
                round(
                    trade["exit"],
                    2
                ),

            "Points":
                round(
                    trade["points"],
                    2
                ),

            "T1 Hit":
                (
                    "YES"

                    if trade["target1_hit"]

                    else "NO"
                ),

            "Exit Reason":
                trade["exit_reason"]
        })

    backtest_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        backtest_df,
        use_container_width=True
    )


# ============================================================
# LIVE CLOSED TRADES
# ============================================================

st.markdown(
    "### Recent Live Closed Trades"
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

            "T1 Hit":
                (
                    "YES"

                    if trade["target1_hit"]

                    else "NO"
                ),

            "Exit Reason":
                trade["exit_reason"]
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True
    )


# ============================================================
# MANUAL CLOSE
# ============================================================

if st.session_state.running_trade:

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
# REFRESH
# ============================================================

if st.button(
    "🔄 Refresh Market Data"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "EMA 9 + EMA 15 + VWAP | "
    "ATM/ITM Options Only | "
    "1:1 → Breakeven SL → "
    "Dynamic Trailing SL"
)
