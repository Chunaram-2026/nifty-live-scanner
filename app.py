from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from datetime import time as dt_time
from zoneinfo import ZoneInfo


app = Flask(__name__)


# =========================================================
# SETTINGS
# =========================================================

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}


TIMEFRAMES = [
    "1m",
    "2m",
    "3m",
    "5m",
    "15m",
    "1h",
    "2h",
    "1d",
    "1wk",
]


# =========================================================
# SCALPING SETTINGS
# =========================================================

# केवल इन timeframe पर trade signals / backtest होंगे
SCALPING_TIMEFRAMES = [
    "1m",
    "2m",
    "3m",
    "5m",
    "15m",
]


# Maximum trades per day
MAX_TRADES_PER_DAY = 3


# =========================================================
# STOP LOSS RULE
# =========================================================

# Signal candle के हिसाब से SL निकलेगा।
# Entry और SL के बीच risk इससे ज्यादा हुआ
# तो trade नहीं लिया जाएगा.

MAX_STOP_LOSS_POINTS = 15.0


# SL में extra buffer
SL_BUFFER_POINTS = 0.0


# =========================================================
# MARKET TIME RULES
# =========================================================

MARKET_TZ = ZoneInfo(
    "Asia/Kolkata"
)


# NSE market open
MARKET_OPEN_TIME = dt_time(
    9,
    15
)


# शुरुआती volatility settle होने तक
# नई trade नहीं

TRADE_START_TIME = dt_time(
    9,
    25
)


# Market close से पहले
# नई trade बंद

TRADE_END_TIME = dt_time(
    15,
    10
)


# =========================================================
# MAXIMUM SCALPING HOLDING TIME
#
# इतने समय के अंदर SL / Target नहीं लगा
# तो TIME EXIT
# =========================================================

MAX_HOLDING_BARS = {

    "1m": 15,

    "2m": 10,

    "3m": 7,

    "5m": 6,

    "15m": 3
}


# =========================================================
# EMA SETTINGS
# =========================================================

EMA_SLOPE_LOOKBACK = 5


# Normalized EMA slope
MIN_EMA_SLOPE = 0.20


# =========================================================
# CANDLE QUALITY
# =========================================================

MIN_BODY_PERCENT = 0.45

WICK_RATIO = 1.5


# =========================================================
# RISK REWARD
# =========================================================

RR_1 = 2.0

RR_2 = 3.0


# =========================================================
# TIMEFRAME SETTINGS
# =========================================================

def timeframe_settings(tf):

    if tf == "1m":
        return "1m", "7d", None

    if tf == "2m":
        return "2m", "60d", None

    if tf == "3m":
        return "1m", "7d", "3min"

    if tf == "5m":
        return "5m", "60d", None

    if tf == "15m":
        return "15m", "60d", None

    if tf == "1h":
        return "1h", "730d", None

    if tf == "2h":
        return "1h", "730d", "2h"

    if tf == "1d":
        return "1d", "5y", None

    if tf == "1wk":
        return "1wk", "10y", None

    return "5m", "60d", None


# =========================================================
# CLEAN DATA
# =========================================================

def clean_columns(data):

    if data is None:
        return None

    if data.empty:
        return None

    data = data.copy()


    # MultiIndex columns fix
    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )


    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]


    for col in required:

        if col not in data.columns:
            return None


    if "Volume" not in data.columns:

        data["Volume"] = 0


    data = data.dropna(
        subset=required
    )


    for col in required + [
        "Volume"
    ]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )


    data = data.dropna(
        subset=required
    )


    return data


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data(symbol, tf):

    interval, period, resample_rule = (
        timeframe_settings(tf)
    )


    try:

        data = yf.download(

            symbol,

            period=period,

            interval=interval,

            progress=False,

            auto_adjust=False,

            threads=False,

            group_by="column"
        )


        data = clean_columns(data)


        if data is None:
            return None


        # =================================================
        # CUSTOM TIMEFRAME RESAMPLE
        # =================================================

        if resample_rule:

            data = data.resample(
                resample_rule
            ).agg({

                "Open": "first",

                "High": "max",

                "Low": "min",

                "Close": "last",

                "Volume": "sum"
            })


            data = data.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close"
                ]
            )


        return data


    except Exception as e:

        print(
            "DOWNLOAD ERROR:",
            e
        )

        traceback.print_exc()

        return None


# =========================================================
# TIMESTAMP TO IST
# =========================================================

def to_ist(timestamp):

    ts = pd.Timestamp(
        timestamp
    )


    if ts.tzinfo is None:

        ts = ts.tz_localize(
            MARKET_TZ
        )

    else:

        ts = ts.tz_convert(
            MARKET_TZ
        )


    return ts


# =========================================================
# CHECK MARKET ENTRY TIME
#
# नई trade:
#
# 09:25 से पहले नहीं
# 15:10 के बाद नहीं
# Weekend नहीं
# =========================================================

def is_valid_entry_time(
    timestamp,
    tf
):

    # केवल scalping TF
    if tf not in SCALPING_TIMEFRAMES:
        return False


    try:

        ts = to_ist(
            timestamp
        )

    except Exception:

        return False


    # Saturday / Sunday
    if ts.weekday() >= 5:
        return False


    current_time = ts.time()


    if current_time < TRADE_START_TIME:
        return False


    if current_time >= TRADE_END_TIME:
        return False


    return True


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(data):

    if data is None:
        return None

    if data.empty:
        return None


    data = data.copy()


    close = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )


    high = pd.to_numeric(
        data["High"],
        errors="coerce"
    )


    low = pd.to_numeric(
        data["Low"],
        errors="coerce"
    )


    open_price = pd.to_numeric(
        data["Open"],
        errors="coerce"
    )


    volume = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    ).fillna(0)


    # =====================================================
    # EMA 9
    # =====================================================

    data["EMA9"] = (
        close
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    # =====================================================
    # EMA 15
    # =====================================================

    data["EMA15"] = (
        close
        .ewm(
            span=15,
            adjust=False
        )
        .mean()
    )


    # =====================================================
    # VWAP
    #
    # User requested:
    #
    # (O + H + L + C) / 4
    # =====================================================

    typical_price = (

        open_price

        +

        high

        +

        low

        +

        close

    ) / 4


    # =====================================================
    # DAILY VWAP RESET
    # =====================================================

    dates = pd.Series(

        pd.to_datetime(
            data.index
        ).date,

        index=data.index
    )


    cumulative_pv = (

        typical_price

        *

        volume

    ).groupby(
        dates
    ).cumsum()


    cumulative_volume = (

        volume
        .groupby(
            dates
        )
        .cumsum()
    )


    data["VWAP"] = np.where(

        cumulative_volume > 0,

        cumulative_pv /
        cumulative_volume,

        typical_price
    )


    # =====================================================
    # EMA SLOPE
    # =====================================================

    data["EMA9_SLOPE"] = (

        data["EMA9"]

        -

        data["EMA9"].shift(
            EMA_SLOPE_LOOKBACK
        )
    )


    data["EMA15_SLOPE"] = (

        data["EMA15"]

        -

        data["EMA15"].shift(
            EMA_SLOPE_LOOKBACK
        )
    )


    return data


# =========================================================
# CANDLE QUALITY
# =========================================================

def candle_info(row):

    try:

        o = float(
            row["Open"]
        )

        h = float(
            row["High"]
        )

        l = float(
            row["Low"]
        )

        c = float(
            row["Close"]
        )

    except Exception:

        return {
            "valid": False
        }


    candle_range = (
        h - l
    )


    body = abs(
        c - o
    )


    if candle_range <= 0:

        return {
            "valid": False
        }


    body_ratio = (
        body /
        candle_range
    )


    upper_wick = (

        h

        -

        max(
            o,
            c
        )
    )


    lower_wick = (

        min(
            o,
            c
        )

        -

        l
    )


    bullish = (
        c > o
    )


    bearish = (
        c < o
    )


    # Strong body
    strong_body = (

        body_ratio >=
        MIN_BODY_PERCENT
    )


    # Bullish hammer / rejection
    bullish_hammer = (

        bullish

        and

        lower_wick >=
        body * WICK_RATIO

        and

        upper_wick <=
        candle_range * 0.35
    )


    # Bearish rejection
    bearish_hammer = (

        bearish

        and

        upper_wick >=
        body * WICK_RATIO

        and

        lower_wick <=
        candle_range * 0.35
    )


    good_bullish = (

        bullish

        and

        (
            strong_body
            or
            bullish_hammer
        )
    )


    good_bearish = (

        bearish

        and

        (
            strong_body
            or
            bearish_hammer
        )
    )


    return {

        "valid": True,

        "bullish": bullish,

        "bearish": bearish,

        "body_ratio": body_ratio,

        "upper_wick": upper_wick,

        "lower_wick": lower_wick,

        "strong_body": strong_body,

        "bullish_hammer": bullish_hammer,

        "bearish_hammer": bearish_hammer,

        "good_bullish": good_bullish,

        "good_bearish": good_bearish
    }


# =========================================================
# EMA TREND
# =========================================================

def ema_trend_ok(row):

    try:

        close = float(
            row["Close"]
        )


        ema9 = float(
            row["EMA9"]
        )


        ema15 = float(
            row["EMA15"]
        )


        ema9_slope = float(
            row["EMA9_SLOPE"]
        )


        ema15_slope = float(
            row["EMA15_SLOPE"]
        )


    except Exception:

        return {

            "bullish": False,

            "bearish": False
        }


    if close <= 0:

        return {

            "bullish": False,

            "bearish": False
        }


    ema9_slope_pct = (

        ema9_slope

        /

        close

    ) * 100


    ema15_slope_pct = (

        ema15_slope

        /

        close

    ) * 100


    bullish = (

        ema9 > ema15

        and

        ema9_slope_pct >
        MIN_EMA_SLOPE

        and

        ema15_slope_pct > 0
    )


    bearish = (

        ema9 < ema15

        and

        ema9_slope_pct <
        -MIN_EMA_SLOPE

        and

        ema15_slope_pct < 0
    )


    return {

        "bullish": bullish,

        "bearish": bearish,

        "ema9_slope_pct":
        ema9_slope_pct,

        "ema15_slope_pct":
        ema15_slope_pct
    }


# =========================================================
# GET RAW SIGNAL
# =========================================================

def get_raw_signal(row):

    try:

        price = float(
            row["Close"]
        )


        ema9 = float(
            row["EMA9"]
        )


        ema15 = float(
            row["EMA15"]
        )


        vwap = float(
            row["VWAP"]
        )


    except Exception:

        return "WAIT"


    if any(

        pd.isna(x)

        for x in [

            price,

            ema9,

            ema15,

            vwap
        ]
    ):

        return "WAIT"


    candle = candle_info(
        row
    )


    if not candle.get(
        "valid"
    ):

        return "WAIT"


    trend = ema_trend_ok(
        row
    )


    # =====================================================
    # CALL
    # =====================================================

    bullish_structure = (

        ema9 > ema15

        and

        trend["bullish"]

        and

        price > ema9

        and

        price > ema15

        and

        price > vwap

        and

        ema9 > vwap
    )


    if (

        bullish_structure

        and

        candle["good_bullish"]
    ):

        return "CALL"


    # =====================================================
    # PUT
    # =====================================================

    bearish_structure = (

        ema9 < ema15

        and

        trend["bearish"]

        and

        price < ema9

        and

        price < ema15

        and

        price < vwap

        and

        ema9 < vwap
    )


    if (

        bearish_structure

        and

        candle["good_bearish"]
    ):

        return "PUT"


    return "WAIT"


# =========================================================
# GET TRADE SETUP
#
# यहाँ signal के साथ
# Maximum SL rule भी check होगा
# =========================================================

def get_trade_setup(
    row,
    timestamp,
    tf
):

    empty = {

        "signal": "WAIT",

        "entry": None,

        "stop_loss": None,

        "target_1": None,

        "target_2": None,

        "risk": None
    }


    # केवल scalping TF
    if tf not in SCALPING_TIMEFRAMES:
        return empty


    # Valid market entry time
    if not is_valid_entry_time(
        timestamp,
        tf
    ):
        return empty


    signal = get_raw_signal(
        row
    )


    if signal == "WAIT":
        return empty


    try:

        entry = float(
            row["Close"]
        )


        low = float(
            row["Low"]
        )


        high = float(
            row["High"]
        )

    except Exception:

        return empty


    # =====================================================
    # CALL
    # =====================================================

    if signal == "CALL":

        stop_loss = (

            low

            -

            SL_BUFFER_POINTS
        )


        risk = (

            entry

            -

            stop_loss
        )


        # SL invalid
        if risk <= 0:
            return empty


        # Maximum 15 points
        if risk > MAX_STOP_LOSS_POINTS:
            return empty


        target_1 = (

            entry

            +

            risk * RR_1
        )


        target_2 = (

            entry

            +

            risk * RR_2
        )


        return {

            "signal": "CALL",

            "entry": entry,

            "stop_loss": stop_loss,

            "target_1": target_1,

            "target_2": target_2,

            "risk": risk
        }


    # =====================================================
    # PUT
    # =====================================================

    if signal == "PUT":

        stop_loss = (

            high

            +

            SL_BUFFER_POINTS
        )


        risk = (

            stop_loss

            -

            entry
        )


        if risk <= 0:
            return empty


        # Maximum 15 points
        if risk > MAX_STOP_LOSS_POINTS:
            return empty


        target_1 = (

            entry

            -

            risk * RR_1
        )


        target_2 = (

            entry

            -

            risk * RR_2
        )


        return {

            "signal": "PUT",

            "entry": entry,

            "stop_loss": stop_loss,

            "target_1": target_1,

            "target_2": target_2,

            "risk": risk
        }


    return empty


# =========================================================
# ADD SIGNAL MARKERS
#
# केवल valid scalping setup marker
# =========================================================

def add_signal_markers(
    data,
    tf
):

    if data is None:
        return data


    if data.empty:
        return data


    data = data.copy()


    markers = []


    previous_signal = "WAIT"


    for timestamp, row in data.iterrows():

        setup = get_trade_setup(

            row,

            timestamp,

            tf
        )


        signal = setup[
            "signal"
        ]


        marker = ""


        if (

            signal == "CALL"

            and

            previous_signal != "CALL"
        ):

            marker = "CALL"


        elif (

            signal == "PUT"

            and

            previous_signal != "PUT"
        ):

            marker = "PUT"


        markers.append(
            marker
        )


        previous_signal = signal


    data["MARKER"] = (
        markers
    )


    return data


# =========================================================
# CALCULATE ALL
# =========================================================

def calculate_all_signals(
    data,
    tf
):

    if data is None:
        return None


    if data.empty:
        return None


    data = calculate_indicators(
        data
    )


    if data is None:
        return None


    if data.empty:
        return None


    data = add_signal_markers(
        data,
        tf
    )


    return data


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(
    data,
    tf
):

    empty_result = {

        "signal": "NO DATA",

        "price": None,

        "ema9": None,

        "ema15": None,

        "vwap": None,

        "stop_loss": None,

        "target_1": None,

        "target_2": None,

        "time": None
    }


    if data is None:
        return empty_result


    if len(data) < 20:
        return empty_result


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None:
        return empty_result


    if data.empty:
        return empty_result


    row = data.iloc[-1]

    timestamp = data.index[-1]


    try:

        price = float(
            row["Close"]
        )

    except Exception:

        return empty_result


    setup = get_trade_setup(

        row,

        timestamp,

        tf
    )


    signal = setup[
        "signal"
    ]


    return {

        "signal": signal,

        "price": round(
            price,
            2
        ),

        "ema9": round(
            float(
                row["EMA9"]
            ),
            2
        ),

        "ema15": round(
            float(
                row["EMA15"]
            ),
            2
        ),

        "vwap": round(
            float(
                row["VWAP"]
            ),
            2
        ),

        "stop_loss":

        round(
            setup["stop_loss"],
            2
        )

        if setup["stop_loss"]
        is not None

        else None,


        "target_1":

        round(
            setup["target_1"],
            2
        )

        if setup["target_1"]
        is not None

        else None,


        "target_2":

        round(
            setup["target_2"],
            2
        )

        if setup["target_2"]
        is not None

        else None,


        "time":
        str(timestamp)
    }


# =========================================================
# CHART JSON
# =========================================================

def chart_json(
    data,
    tf
):

    if data is None:
        return []


    if data.empty:
        return []


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None:
        return []


    if data.empty:
        return []


    result = []


    for timestamp, row in data.iterrows():

        try:

            ts = int(

                pd.Timestamp(
                    timestamp
                ).timestamp()
            )


            marker = row.get(
                "MARKER",
                ""
            )


            result.append({

                "time": ts,

                "open": round(
                    float(
                        row["Open"]
                    ),
                    2
                ),

                "high": round(
                    float(
                        row["High"]
                    ),
                    2
                ),

                "low": round(
                    float(
                        row["Low"]
                    ),
                    2
                ),

                "close": round(
                    float(
                        row["Close"]
                    ),
                    2
                ),

                "ema9": round(
                    float(
                        row["EMA9"]
                    ),
                    2
                ),

                "ema15": round(
                    float(
                        row["EMA15"]
                    ),
                    2
                ),

                "vwap": round(
                    float(
                        row["VWAP"]
                    ),
                    2
                ),

                "marker":
                marker
            })


        except Exception as e:

            print(
                "CHART ROW ERROR:",
                e
            )


    return result


# =========================================================
# BACKTEST
#
# RULES:
#
# 1. केवल scalping TF
# 2. Market opening settlement के बाद
# 3. Market closing से पहले no entry
# 4. Max 15 point SL
# 5. Max 3 trades/day
# 6. Max holding bars
# 7. केवल closed trades count होंगे
# =========================================================

def run_backtest(
    data,
    tf
):

    empty = {

        "trades": [],

        "total_trades": 0,

        "closed_trades": 0,

        "wins": 0,

        "losses": 0,

        "win_rate": 0,

        "net_points": 0,

        "target_1_hits": 0,

        "target_2_hits": 0
    }


    # =====================================================
    # NON-SCALPING TIMEFRAME
    # =====================================================

    if tf not in SCALPING_TIMEFRAMES:
        return empty


    if data is None:
        return empty


    if len(data) < 30:
        return empty


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None:
        return empty


    if len(data) < 30:
        return empty


    trades = []

    open_trade = None

    daily_trade_count = {}


    max_holding_bars = (
        MAX_HOLDING_BARS.get(
            tf,
            5
        )
    )


    # =====================================================
    # BACKTEST LOOP
    # =====================================================

    for i in range(
        20,
        len(data)
    ):


        row = data.iloc[i]

        timestamp = data.index[i]


        date_key = str(

            pd.Timestamp(
                timestamp
            ).date()
        )


        # =================================================
        # FIRST CHECK OPEN TRADE
        # =================================================

        if open_trade is not None:


            trade_type = (
                open_trade["type"]
            )


            entry = (
                open_trade["entry"]
            )


            sl = (
                open_trade["stop_loss"]
            )


            target1 = (
                open_trade["target_1"]
            )


            target2 = (
                open_trade["target_2"]
            )


            high = float(
                row["High"]
            )


            low = float(
                row["Low"]
            )


            close = float(
                row["Close"]
            )


            exit_price = None

            exit_reason = None


            # =============================================
            # CALL EXIT
            # =============================================

            if trade_type == "CALL":


                # Conservative:
                # Same candle SL + target
                # तो SL पहले

                if low <= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS"
                    )


                elif high >= target2:

                    exit_price = target2

                    exit_reason = (
                        "TARGET 1:3"
                    )


                elif high >= target1:

                    exit_price = target1

                    exit_reason = (
                        "TARGET 1:2"
                    )


            # =============================================
            # PUT EXIT
            # =============================================

            elif trade_type == "PUT":


                if high >= sl:

                    exit_price = sl

                    exit_reason = (
                        "STOP LOSS"
                    )


                elif low <= target2:

                    exit_price = target2

                    exit_reason = (
                        "TARGET 1:3"
                    )


                elif low <= target1:

                    exit_price = target1

                    exit_reason = (
                        "TARGET 1:2"
                    )


            # =============================================
            # TIME EXIT
            # =============================================

            bars_held = (

                i

                -

                open_trade["entry_bar"]
            )


            if (

                exit_price is None

                and

                bars_held >=
                max_holding_bars
            ):

                exit_price = close

                exit_reason = (
                    "TIME EXIT"
                )


            # =============================================
            # MARKET END EXIT
            #
            # 15:10 के बाद नई trade नहीं,
            # open trade को भी scalping के लिए
            # close कर देंगे
            # =============================================

            try:

                ts_ist = to_ist(
                    timestamp
                )


                current_time = (
                    ts_ist.time()
                )


                if (

                    exit_price is None

                    and

                    current_time >=
                    TRADE_END_TIME
                ):

                    exit_price = close

                    exit_reason = (
                        "MARKET TIME EXIT"
                    )


            except Exception:

                pass


            # =============================================
            # CLOSE TRADE
            # =============================================

            if exit_price is not None:


                if trade_type == "CALL":

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


                result = (

                    "WIN"

                    if points > 0

                    else "LOSS"
                )


                trades.append({

                    "type":
                    trade_type,


                    "entry":
                    round(
                        entry,
                        2
                    ),


                    "exit":
                    round(
                        exit_price,
                        2
                    ),


                    "stop_loss":
                    round(
                        sl,
                        2
                    ),


                    "target_1":
                    round(
                        target1,
                        2
                    ),


                    "target_2":
                    round(
                        target2,
                        2
                    ),


                    "points":
                    round(
                        points,
                        2
                    ),


                    "result":
                    result,


                    "exit_reason":
                    exit_reason,


                    "entry_time":
                    open_trade[
                        "entry_time"
                    ],


                    "exit_time":
                    str(
                        timestamp
                    )
                })


                open_trade = None


        # =================================================
        # NEW TRADE ONLY
        # IF NO OPEN TRADE
        # =================================================

        if open_trade is not None:
            continue


        # =================================================
        # MAX 3 TRADES PER DAY
        # =================================================

        today_count = (
            daily_trade_count.get(
                date_key,
                0
            )
        )


        if today_count >= MAX_TRADES_PER_DAY:
            continue


        # =================================================
        # GET VALID SCALPING SETUP
        #
        # इसमें:
        #
        # Signal
        # Market time
        # Maximum 15 point SL
        #
        # सब check होगा
        # =================================================

        setup = get_trade_setup(

            row,

            timestamp,

            tf
        )


        signal = setup[
            "signal"
        ]


        if signal == "WAIT":
            continue


        entry = setup[
            "entry"
        ]


        stop_loss = setup[
            "stop_loss"
        ]


        target_1 = setup[
            "target_1"
        ]


        target_2 = setup[
            "target_2"
        ]


        # =================================================
        # OPEN TRADE
        # =================================================

        open_trade = {

            "type":
            signal,


            "entry":
            entry,


            "stop_loss":
            stop_loss,


            "target_1":
            target_1,


            "target_2":
            target_2,


            "entry_time":
            str(
                timestamp
            ),


            "entry_bar":
            i
        }


        daily_trade_count[
            date_key
        ] = (

            today_count + 1
        )


    # =====================================================
    # IMPORTANT
    #
    # END OF DATA पर trade force close नहीं करेंगे
    #
    # अधूरी trade backtest statistics में नहीं जाएगी
    # =====================================================


    # =====================================================
    # STATISTICS
    #
    # केवल वास्तव में CLOSED trades
    # =====================================================

    total_trades = len(
        trades
    )


    closed_trades_count = (
        total_trades
    )


    wins = sum(

        1

        for trade in trades

        if trade["result"] == "WIN"
    )


    losses = sum(

        1

        for trade in trades

        if trade["result"] == "LOSS"
    )


    # =====================================================
    # WIN RATE
    # =====================================================

    win_rate = 0


    if closed_trades_count > 0:

        win_rate = round(

            (

                wins

                /

                closed_trades_count

            )

            *

            100,

            2
        )


    target_1_hits = sum(

        1

        for trade in trades

        if trade["exit_reason"] ==
        "TARGET 1:2"
    )


    target_2_hits = sum(

        1

        for trade in trades

        if trade["exit_reason"] ==
        "TARGET 1:3"
    )


    net_points = round(

        sum(

            trade["points"]

            for trade in trades

        ),

        2
    )


    return {

        "trades":
        trades,


        "total_trades":
        total_trades,


        "closed_trades":
        closed_trades_count,


        "wins":
        wins,


        "losses":
        losses,


        "win_rate":
        win_rate,


        "net_points":
        net_points,


        "target_1_hits":
        target_1_hits,


        "target_2_hits":
        target_2_hits
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
Personal Scalping Scanner
</title>


<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    padding: 12px;

    background: #080c12;

    color: #ffffff;

    font-family:
    Arial,
    sans-serif;
}


h1 {

    font-size: 22px;

    margin:
    8px 0 15px;
}


h2 {

    font-size: 18px;

    margin:
    20px 0 10px;
}


.card {

    background: #111923;

    border:
    1px solid #263241;

    border-radius: 12px;

    padding: 14px;

    margin-bottom: 12px;
}


.tf {

    display: flex;

    gap: 6px;

    overflow-x: auto;

    padding-bottom: 6px;
}


button {

    padding:
    9px 12px;

    border-radius: 8px;

    border:
    1px solid #34465a;

    background:
    #172331;

    color: white;

    cursor: pointer;

    white-space: nowrap;
}


button.active {

    background:
    #2463eb;
}


button:active {

    transform:
    scale(0.97);
}


#chart {

    width: 100%;

    height: 500px;
}


.grid {

    display: grid;

    grid-template-columns:
    repeat(2, 1fr);

    gap: 10px;
}


.box {

    background:
    #172331;

    padding: 12px;

    border-radius: 8px;
}


.label {

    font-size: 12px;

    color:
    #aab7c4;
}


.value {

    font-size: 18px;

    margin-top: 5px;

    font-weight: bold;
}


.call {

    color:
    #4ade80;
}


.put {

    color:
    #fb7185;
}


.wait {

    color:
    #facc15;
}


table {

    width: 100%;

    border-collapse:
    collapse;

    font-size: 12px;
}


th,
td {

    padding: 8px;

    border-bottom:
    1px solid #263241;

    text-align: left;
}


.good {

    color:
    #4ade80;
}


.bad {

    color:
    #fb7185;
}


.small {

    font-size: 12px;

    color:
    #aab7c4;
}


</style>

</head>

<body>


<h1>
📈 Personal Scalping Scanner
</h1>


<div class="card">

<div
id="indices"
class="tf">
</div>

<div
id="timeframes"
class="tf">
</div>

</div>


<div class="card">

<div class="grid">


<div class="box">

<div class="label">
Signal
</div>

<div
id="signal"
class="value">
Loading...
</div>

</div>


<div class="box">

<div class="label">
Price
</div>

<div
id="price"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
EMA 9
</div>

<div
id="ema9"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
EMA 15
</div>

<div
id="ema15"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
VWAP (OHLC/4)
</div>

<div
id="vwap"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Stop Loss
</div>

<div
id="sl"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 1
</div>

<div
id="target1"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 2
</div>

<div
id="target2"
class="value">
-
</div>

</div>


</div>

</div>


<h2>
📊 Index Chart
</h2>


<div class="card">

<div id="chart">
</div>

</div>


<h2>
📈 Backtest
</h2>


<div class="card">

<div class="grid">


<div class="box">

<div class="label">
Closed Trades
</div>

<div
id="totalTrades"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Wins
</div>

<div
id="wins"
class="value good">
-
</div>

</div>


<div class="box">

<div class="label">
Losses
</div>

<div
id="losses"
class="value bad">
-
</div>

</div>


<div class="box">

<div class="label">
Win Rate
</div>

<div
id="winRate"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Net Points
</div>

<div
id="netPoints"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
1:2 / 1:3 Targets
</div>

<div
id="targets"
class="value">
-
</div>

</div>


</div>

</div>


<div class="card">

<h2>
Recent Closed Trades
</h2>

<div
id="trades">
</div>

</div>


<script>


let selectedIndex =
"NIFTY 50";


let selectedTF =
"5m";


let chart = null;

let candleSeries = null;

let ema9Series = null;

let ema15Series = null;

let vwapSeries = null;


const indices = [

    "NIFTY 50",

    "BANK NIFTY",

    "SENSEX"
];


const timeframes = [

    "1m",

    "2m",

    "3m",

    "5m",

    "15m",

    "1h",

    "2h",

    "1d",

    "1wk"
];


function createButtons() {


    const indexDiv =
    document.getElementById(
        "indices"
    );


    indexDiv.innerHTML = "";


    indices.forEach(
        name => {


            const button =
            document.createElement(
                "button"
            );


            button.textContent =
            name;


            if (
                name === selectedIndex
            ) {

                button.classList.add(
                    "active"
                );
            }


            button.onclick =
            () => {


                selectedIndex =
                name;


                createButtons();


                loadData();
            };


            indexDiv.appendChild(
                button
            );
        }
    );


    const tfDiv =
    document.getElementById(
        "timeframes"
    );


    tfDiv.innerHTML = "";


    timeframes.forEach(
        tf => {


            const button =
            document.createElement(
                "button"
            );


            button.textContent =
            tf;


            if (
                tf === selectedTF
            ) {

                button.classList.add(
                    "active"
                );
            }


            button.onclick =
            () => {


                selectedTF =
                tf;


                createButtons();


                loadData();
            };


            tfDiv.appendChild(
                button
            );
        }
    );
}


function createChart() {


    const container =
    document.getElementById(
        "chart"
    );


    container.innerHTML = "";


    chart =
    LightweightCharts.createChart(

        container,

        {

            width:
            container.clientWidth,

            height:
            500,


            layout: {

                background: {

                    type:
                    "solid",

                    color:
                    "#111923"
                },


                textColor:
                "#d1d4dc"
            },


            grid: {

                vertLines: {

                    color:
                    "#202b38"
                },


                horzLines: {

                    color:
                    "#202b38"
                }
            },


            rightPriceScale: {

                borderColor:
                "#263241"
            },


            timeScale: {

                borderColor:
                "#263241",

                timeVisible:
                true
            }
        }
    );


    candleSeries =
    chart.addCandlestickSeries({

        upColor:
        "#22c55e",

        downColor:
        "#ef4444",

        borderUpColor:
        "#22c55e",

        borderDownColor:
        "#ef4444",

        wickUpColor:
        "#22c55e",

        wickDownColor:
        "#ef4444"
    });


    ema9Series =
    chart.addLineSeries({

        color:
        "#3b82f6",

        lineWidth:
        2,

        title:
        "EMA 9"
    });


    ema15Series =
    chart.addLineSeries({

        color:
        "#f59e0b",

        lineWidth:
        2,

        title:
        "EMA 15"
    });


    vwapSeries =
    chart.addLineSeries({

        color:
        "#a855f7",

        lineWidth:
        2,

        title:
        "VWAP OHLC/4"
    });


    window.addEventListener(

        "resize",

        () => {


            if (
                chart
            ) {

                chart.applyOptions({

                    width:
                    container.clientWidth
                });
            }
        }
    );
}


function formatNumber(
    value
) {


    if (

        value === null

        ||

        value === undefined

    ) {

        return "-";
    }


    return Number(
        value
    ).toFixed(2);
}


async function loadData() {


    try {


        document.getElementById(
            "signal"
        ).textContent =
        "Loading...";


        const response =
        await fetch(

            "/api/data?index="

            +

            encodeURIComponent(
                selectedIndex
            )

            +

            "&tf="

            +

            encodeURIComponent(
                selectedTF
            )
        );


        const data =
        await response.json();


        if (
            data.error
        ) {

            alert(
                data.error
            );

            return;
        }


        updateScanner(
            data.scanner
        );


        updateChart(
            data.chart
        );


        updateBacktest(
            data.backtest
        );

    }


    catch (
        error
    ) {


        console.error(
            error
        );


        document.getElementById(
            "signal"
        ).textContent =
        "ERROR";
    }
}


function updateScanner(
    scanner
) {


    const signal =

    scanner.signal

    ||

    "WAIT";


    const signalElement =
    document.getElementById(
        "signal"
    );


    signalElement.textContent =
    signal;


    signalElement.className =

    "value "

    +

    (

        signal === "CALL"

        ?

        "call"

        :

        signal === "PUT"

        ?

        "put"

        :

        "wait"
    );


    document.getElementById(
        "price"
    ).textContent =
    formatNumber(
        scanner.price
    );


    document.getElementById(
        "ema9"
    ).textContent =
    formatNumber(
        scanner.ema9
    );


    document.getElementById(
        "ema15"
    ).textContent =
    formatNumber(
        scanner.ema15
    );


    document.getElementById(
        "vwap"
    ).textContent =
    formatNumber(
        scanner.vwap
    );


    document.getElementById(
        "sl"
    ).textContent =
    formatNumber(
        scanner.stop_loss
    );


    document.getElementById(
        "target1"
    ).textContent =
    formatNumber(
        scanner.target_1
    );


    document.getElementById(
        "target2"
    ).textContent =
    formatNumber(
        scanner.target_2
    );
}


function updateChart(
    chartData
) {


    if (
        !chart
    ) {

        createChart();
    }


    const candles =
    chartData.map(

        x => ({

            time:
            x.time,

            open:
            x.open,

            high:
            x.high,

            low:
            x.low,

            close:
            x.close
        })
    );


    const ema9 =
    chartData.map(

        x => ({

            time:
            x.time,

            value:
            x.ema9
        })
    );


    const ema15 =
    chartData.map(

        x => ({

            time:
            x.time,

            value:
            x.ema15
        })
    );


    const vwap =
    chartData.map(

        x => ({

            time:
            x.time,

            value:
            x.vwap
        })
    );


    candleSeries.setData(
        candles
    );


    ema9Series.setData(
        ema9
    );


    ema15Series.setData(
        ema15
    );


    vwapSeries.setData(
        vwap
    );


    const markers = [];


    chartData.forEach(

        x => {


            if (
                x.marker === "CALL"
            ) {

                markers.push({

                    time:
                    x.time,

                    position:
                    "belowBar",

                    color:
                    "#22c55e",

                    shape:
                    "arrowUp",

                    text:
                    "CALL"
                });
            }


            if (
                x.marker === "PUT"
            ) {

                markers.push({

                    time:
                    x.time,

                    position:
                    "aboveBar",

                    color:
                    "#ef4444",

                    shape:
                    "arrowDown",

                    text:
                    "PUT"
                });
            }
        }
    );


    candleSeries.setMarkers(
        markers
    );


    chart.timeScale()
    .fitContent();
}


function updateBacktest(
    backtest
) {


    document.getElementById(
        "totalTrades"
    ).textContent =
    backtest.closed_trades;


    document.getElementById(
        "wins"
    ).textContent =
    backtest.wins;


    document.getElementById(
        "losses"
    ).textContent =
    backtest.losses;


    document.getElementById(
        "winRate"
    ).textContent =

    backtest.win_rate

    +

    "%";


    const netElement =
    document.getElementById(
        "netPoints"
    );


    netElement.textContent =
    backtest.net_points;


    netElement.className =

    "value "

    +

    (

        backtest.net_points >= 0

        ?

        "good"

        :

        "bad"
    );


    document.getElementById(
        "targets"
    ).textContent =

    backtest.target_1_hits

    +

    " / "

    +

    backtest.target_2_hits;


    const tradesDiv =
    document.getElementById(
        "trades"
    );


    tradesDiv.innerHTML = "";


    const trades =

    [...backtest.trades]

    .reverse()

    .slice(
        0,
        30
    );


    if (
        trades.length === 0
    ) {

        tradesDiv.innerHTML =

        "<div class='small'>No closed trades found.</div>";

        return;
    }


    trades.forEach(

        trade => {


            const div =
            document.createElement(
                "div"
            );


            div.style.padding =
            "8px 0";


            div.style.borderBottom =

            "1px solid #263241";


            const resultClass =

            trade.result === "WIN"

            ?

            "good"

            :

            "bad";


            div.innerHTML =

            "<b>"

            +

            trade.type

            +

            "</b><br>"

            +

            "Entry: "

            +

            trade.entry

            +

            " | SL: "

            +

            trade.stop_loss

            +

            "<br>"

            +

            "Exit: "

            +

            trade.exit

            +

            " | Points: "

            +

            trade.points

            +

            " | "

            +

            "<span class='"

            +

            resultClass

            +

            "'>"

            +

            trade.result

            +

            "</span>"

            +

            "<br>"

            +

            "<span class='small'>"

            +

            trade.exit_reason

            +

            "</span>";


            tradesDiv.appendChild(
                div
            );
        }
    );
}


createButtons();

createChart();

loadData();


setInterval(

    loadData,

    60000
);


</script>

</body>

</html>
"""


# =========================================================
# API
# =========================================================

@app.route("/api/data")
def api_data():

    index_name = request.args.get(
        "index",
        "NIFTY 50"
    )


    tf = request.args.get(
        "tf",
        "5m"
    )


    if index_name not in INDICES:

        return jsonify({

            "error":
            "Invalid index"

        }), 400


    if tf not in TIMEFRAMES:

        return jsonify({

            "error":
            "Invalid timeframe"

        }), 400


    symbol = INDICES[
        index_name
    ]


    data = download_data(
        symbol,
        tf
    )


    if data is None:

        return jsonify({

            "error":
            "Market data not available"

        }), 500


    if data.empty:

        return jsonify({

            "error":
            "Market data not available"

        }), 500


    scanner = calculate_scanner(
        data,
        tf
    )


    chart = chart_json(
        data,
        tf
    )


    backtest = run_backtest(
        data,
        tf
    )


    return jsonify({

        "index":
        index_name,

        "timeframe":
        tf,

        "scanner":
        scanner,

        "chart":
        chart,

        "backtest":
        backtest
    })


# =========================================================
# HEALTH
# =========================================================

@app.route("/api/health")
def api_health():

    return jsonify({

        "status":
        "ok"
    })


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False
    )
