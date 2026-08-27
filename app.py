from flask import Flask, jsonify, request
import os
import traceback
from datetime import time
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# APP
# =========================================================

app = Flask(__name__)


# =========================================================
# TIMEZONE
# =========================================================

IST = ZoneInfo("Asia/Kolkata")


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
# TRADING RULES
# =========================================================

MAX_TRADES_PER_DAY = 3

MAX_STOP_LOSS_POINTS = 15.0
MIN_STOP_LOSS_POINTS = 1.0

RR_1 = 2.0
RR_2 = 3.0

SL_BUFFER_POINTS = 0.0


# =========================================================
# 30 DEGREE APPROXIMATION
#
# Real chart angle depends on zoom and screen size.
# Therefore normalized EMA slope is used as a practical
# approximation.
# =========================================================

EMA_SLOPE_LOOKBACK = 5
MIN_EMA_SLOPE_PERCENT = 0.05


# =========================================================
# CANDLE QUALITY
# =========================================================

MIN_BODY_RATIO = 0.45
WICK_RATIO = 1.5


# =========================================================
# NSE MARKET TIME
# =========================================================

MARKET_OPEN_TIME = time(9, 15)

NO_TRADE_AFTER_OPEN_MINUTES = 10

MARKET_CLOSE_TIME = time(15, 30)

NO_TRADE_BEFORE_CLOSE_MINUTES = 20


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
# TIMEZONE HELPER
# =========================================================

def timestamp_to_ist(timestamp):

    ts = pd.Timestamp(timestamp)

    if ts.tzinfo is None:

        # NSE data should be treated as Indian time
        ts = ts.tz_localize(IST)

    else:

        ts = ts.tz_convert(IST)

    return ts


# =========================================================
# CLEAN DATA
# =========================================================

def clean_columns(data):

    if data is None:
        return None

    if data.empty:
        return None

    data = data.copy()


    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)


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


    for col in required + ["Volume"]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )


    data = data.dropna(
        subset=required
    )


    if data.empty:
        return None


    # Remove duplicate timestamps
    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]


    data = data.sort_index()


    return data


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data(symbol, tf):

    interval, period, resample_rule = (
        timeframe_settings(tf)
    )

    try:

        ticker = yf.Ticker(symbol)


        data = ticker.history(

            period=period,

            interval=interval,

            auto_adjust=False,

            prepost=False,

            actions=False
        )


        data = clean_columns(data)


        if data is None:

            print(
                f"NO DATA: {symbol} | {tf}"
            )

            return None


        # =================================================
        # CUSTOM RESAMPLING
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


        data = clean_columns(data)


        if data is None:
            return None


        print(

            f"DATA OK | "
            f"Symbol={symbol} | "
            f"TF={tf} | "
            f"Rows={len(data)}"

        )


        return data


    except Exception as e:

        print(

            f"DOWNLOAD ERROR | "
            f"Symbol={symbol} | "
            f"TF={tf}"

        )

        print(str(e))

        traceback.print_exc()


        return None


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(data):

    if data is None or data.empty:
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

    data["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()


    # =====================================================
    # EMA 15
    # =====================================================

    data["EMA15"] = close.ewm(
        span=15,
        adjust=False
    ).mean()


    # =====================================================
    # VWAP
    #
    # (Open + High + Low + Close) / 4
    # =====================================================

    average_price = (

        open_price

        +

        high

        +

        low

        +

        close

    ) / 4


    # =====================================================
    # DAILY VWAP RESET USING IST
    # =====================================================

    index_time = pd.DatetimeIndex(
        data.index
    )


    if index_time.tz is None:

        index_time = index_time.tz_localize(
            IST
        )

    else:

        index_time = index_time.tz_convert(
            IST
        )


    dates = pd.Series(
        index_time.date,
        index=data.index
    )


    price_volume = (
        average_price * volume
    )


    cumulative_pv = (
        price_volume
        .groupby(dates)
        .cumsum()
    )


    cumulative_volume = (
        volume
        .groupby(dates)
        .cumsum()
    )


    # If volume unavailable,
    # use OHLC/4 fallback

    data["VWAP"] = np.where(

        cumulative_volume > 0,

        cumulative_pv /
        cumulative_volume,

        average_price
    )


    # =====================================================
    # EMA SLOPE
    #
    # Used as 30-degree trend approximation
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

        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

    except Exception:

        return {
            "valid": False
        }


    candle_range = h - l


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

        max(o, c)
    )


    lower_wick = (

        min(o, c)

        -

        l
    )


    bullish = c > o
    bearish = c < o


    strong_body = (

        body_ratio >=
        MIN_BODY_RATIO
    )


    # =====================================================
    # BULLISH REJECTION
    # =====================================================

    bullish_hammer = (

        bullish

        and

        body > 0

        and

        lower_wick >=
        body * WICK_RATIO

        and

        upper_wick <=
        candle_range * 0.35
    )


    # =====================================================
    # BEARISH REJECTION
    # =====================================================

    bearish_hammer = (

        bearish

        and

        body > 0

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
# EMA TREND FILTER
#
# 30 DEGREE APPROXIMATION
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


    if any(

        pd.isna(x)

        for x in [

            ema9,

            ema15,

            ema9_slope,

            ema15_slope
        ]

    ):

        return {

            "bullish": False,

            "bearish": False
        }


    ema9_slope_percent = (

        ema9_slope /

        close

    ) * 100


    ema15_slope_percent = (

        ema15_slope /

        close

    ) * 100


    bullish = (

        ema9 > ema15

        and

        ema9_slope_percent >=
        MIN_EMA_SLOPE_PERCENT

        and

        ema15_slope_percent > 0
    )


    bearish = (

        ema9 < ema15

        and

        ema9_slope_percent <=
        -MIN_EMA_SLOPE_PERCENT

        and

        ema15_slope_percent < 0
    )


    return {

        "bullish": bullish,

        "bearish": bearish,

        "ema9_slope_percent":
        ema9_slope_percent,

        "ema15_slope_percent":
        ema15_slope_percent
    }


# =========================================================
# MARKET TIME FILTER
# =========================================================

def is_market_time_allowed(timestamp, tf):

    if tf in [
        "1d",
        "1wk"
    ]:

        return True


    try:

        ts = timestamp_to_ist(
            timestamp
        )


        current_minutes = (

            ts.hour * 60

            +

            ts.minute
        )


        market_open_minutes = (
            9 * 60 + 15
        )


        market_close_minutes = (
            15 * 60 + 30
        )


        first_allowed = (

            market_open_minutes

            +

            NO_TRADE_AFTER_OPEN_MINUTES
        )


        last_allowed = (

            market_close_minutes

            -

            NO_TRADE_BEFORE_CLOSE_MINUTES
        )


        if (
            current_minutes <
            first_allowed
        ):

            return False


        if (
            current_minutes >=
            last_allowed
        ):

            return False


        return True


    except Exception as e:

        print(
            "TIME FILTER ERROR:",
            e
        )

        return False


# =========================================================
# GET SIGNAL
# =========================================================

def get_signal(row):

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
        "valid",
        False
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

        trend.get(
            "bullish",
            False
        )

        and

        price > ema9

        and

        price > ema15

        and

        price > vwap

        and

        vwap < ema9

        and

        vwap < ema15
    )


    if (

        bullish_structure

        and

        candle.get(
            "good_bullish",
            False
        )

    ):

        return "CALL"


    # =====================================================
    # PUT
    # =====================================================

    bearish_structure = (

        ema9 < ema15

        and

        trend.get(
            "bearish",
            False
        )

        and

        price < ema9

        and

        price < ema15

        and

        price < vwap

        and

        vwap > ema9

        and

        vwap > ema15
    )


    if (

        bearish_structure

        and

        candle.get(
            "good_bearish",
            False
        )

    ):

        return "PUT"


    return "WAIT"


# =========================================================
# ADD SIGNAL MARKERS
#
# Same direction repeated signal
# gets only one marker.
# =========================================================

def add_signal_markers(data, tf):

    if data is None or data.empty:
        return data


    data = data.copy()


    markers = []


    previous_signal = "WAIT"


    for timestamp, row in data.iterrows():

        signal = get_signal(
            row
        )


        marker = ""


        if not is_market_time_allowed(
            timestamp,
            tf
        ):

            signal = "WAIT"


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


    data["MARKER"] = markers


    return data


# =========================================================
# CALCULATE ALL SIGNALS
# =========================================================

def calculate_all_signals(data, tf):

    if data is None or data.empty:
        return None


    data = calculate_indicators(
        data
    )


    if data is None or data.empty:
        return None


    data = add_signal_markers(
        data,
        tf
    )


    return data


# =========================================================
# CREATE TRADE LEVELS
# =========================================================

def create_trade_levels(signal, row):

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

        return None


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


        if risk <= 0:
            return None


        if risk < MIN_STOP_LOSS_POINTS:
            return None


        if risk > MAX_STOP_LOSS_POINTS:
            return None


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

            "type": "CALL",

            "entry": entry,

            "stop_loss": stop_loss,

            "risk": risk,

            "target_1": target_1,

            "target_2": target_2
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
            return None


        if risk < MIN_STOP_LOSS_POINTS:
            return None


        if risk > MAX_STOP_LOSS_POINTS:
            return None


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

            "type": "PUT",

            "entry": entry,

            "stop_loss": stop_loss,

            "risk": risk,

            "target_1": target_1,

            "target_2": target_2
        }


    return None


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(data, tf):

    empty_result = {

        "signal": "WAIT",

        "price": None,

        "ema9": None,

        "ema15": None,

        "vwap": None,

        "stop_loss": None,

        "risk": None,

        "target_1": None,

        "target_2": None,

        "trade_allowed": False,

        "message": "No trade",

        "time": None
    }


    if data is None or len(data) < 30:

        empty_result["signal"] = (
            "NO DATA"
        )

        empty_result["message"] = (
            "Not enough market data"
        )

        return empty_result


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None or data.empty:
        return empty_result


    row = data.iloc[-1]

    timestamp = data.index[-1]


    price = float(
        row["Close"]
    )


    signal = get_signal(
        row
    )


    allowed_time = is_market_time_allowed(
        timestamp,
        tf
    )


    if not allowed_time:

        signal = "WAIT"


    levels = None


    if signal in [
        "CALL",
        "PUT"
    ]:

        levels = create_trade_levels(
            signal,
            row
        )


        if levels is None:

            signal = "WAIT"


    result = {

        "signal": signal,

        "price": round(
            price,
            2
        ),

        "ema9": round(
            float(row["EMA9"]),
            2
        ),

        "ema15": round(
            float(row["EMA15"]),
            2
        ),

        "vwap": round(
            float(row["VWAP"]),
            2
        ),

        "stop_loss": None,

        "risk": None,

        "target_1": None,

        "target_2": None,

        "trade_allowed": False,

        "message": "WAIT",

        "time": str(
            timestamp_to_ist(
                timestamp
            )
        )
    }


    if levels is not None:

        result["stop_loss"] = round(
            levels["stop_loss"],
            2
        )

        result["risk"] = round(
            levels["risk"],
            2
        )

        result["target_1"] = round(
            levels["target_1"],
            2
        )

        result["target_2"] = round(
            levels["target_2"],
            2
        )

        result["trade_allowed"] = True

        result["message"] = (
            "TRADE READY"
        )


    return result


# =========================================================
# CHART JSON
# =========================================================

def chart_json(data, tf):

    if data is None or data.empty:
        return []


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None or data.empty:
        return []


    result = []


    for timestamp, row in data.iterrows():

        try:

            ts = timestamp_to_ist(
                timestamp
            )


            unix_time = int(
                ts.timestamp()
            )


            result.append({

                "time": unix_time,

                "open": round(
                    float(row["Open"]),
                    2
                ),

                "high": round(
                    float(row["High"]),
                    2
                ),

                "low": round(
                    float(row["Low"]),
                    2
                ),

                "close": round(
                    float(row["Close"]),
                    2
                ),

                "ema9": round(
                    float(row["EMA9"]),
                    2
                ),

                "ema15": round(
                    float(row["EMA15"]),
                    2
                ),

                "vwap": round(
                    float(row["VWAP"]),
                    2
                ),

                "marker": str(
                    row.get(
                        "MARKER",
                        ""
                    )
                )
            })


        except Exception as e:

            print(
                "CHART ERROR:",
                e
            )


    return result


# =========================================================
# BACKTEST
#
# ONLY CLOSED TRADES ARE COUNTED
# RUNNING TRADE IS SEPARATE
# =========================================================

def run_backtest(data, tf):

    empty = {

        "trades": [],

        "total_trades": 0,

        "wins": 0,

        "losses": 0,

        "win_rate": 0,

        "net_points": 0,

        "target_1_hits": 0,

        "target_2_hits": 0,

        "running_trade": None
    }


    if data is None or len(data) < 30:
        return empty


    data = calculate_all_signals(
        data,
        tf
    )


    if data is None or len(data) < 30:
        return empty


    trades = []

    open_trade = None

    daily_trade_count = {}


    # Start after enough indicator history

    for i in range(
        20,
        len(data)
    ):

        row = data.iloc[i]

        timestamp = data.index[i]

        date_key = str(
            timestamp_to_ist(
                timestamp
            ).date()
        )


        # =================================================
        # CHECK RUNNING TRADE
        # =================================================

        if open_trade is not None:

            trade_type = (
                open_trade["type"]
            )

            entry = (
                open_trade["entry"]
            )

            stop_loss = (
                open_trade["stop_loss"]
            )

            target_1 = (
                open_trade["target_1"]
            )

            target_2 = (
                open_trade["target_2"]
            )

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )


            exit_price = None
            exit_reason = None


            # =============================================
            # CALL EXIT
            # =============================================

            if trade_type == "CALL":

                # Conservative:
                # If SL and target both hit
                # in same candle, SL assumed first.

                if low <= stop_loss:

                    exit_price = stop_loss
                    exit_reason = "STOP LOSS"


                elif high >= target_2:

                    exit_price = target_2
                    exit_reason = "TARGET 1:3"


                elif high >= target_1:

                    exit_price = target_1
                    exit_reason = "TARGET 1:2"


            # =============================================
            # PUT EXIT
            # =============================================

            elif trade_type == "PUT":

                if high >= stop_loss:

                    exit_price = stop_loss
                    exit_reason = "STOP LOSS"


                elif low <= target_2:

                    exit_price = target_2
                    exit_reason = "TARGET 1:3"


                elif low <= target_1:

                    exit_price = target_1
                    exit_reason = "TARGET 1:2"


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

                    else

                    "LOSS"
                )


                trades.append({

                    "type": trade_type,

                    "entry": round(
                        entry,
                        2
                    ),

                    "exit": round(
                        exit_price,
                        2
                    ),

                    "stop_loss": round(
                        stop_loss,
                        2
                    ),

                    "risk": round(
                        open_trade["risk"],
                        2
                    ),

                    "target_1": round(
                        target_1,
                        2
                    ),

                    "target_2": round(
                        target_2,
                        2
                    ),

                    "points": round(
                        points,
                        2
                    ),

                    "result": result,

                    "exit_reason": exit_reason,

                    "entry_time":
                    open_trade[
                        "entry_time"
                    ],

                    "exit_time": str(
                        timestamp_to_ist(
                            timestamp
                        )
                    )
                })


                open_trade = None


        # =================================================
        # DON'T OPEN NEW TRADE
        # WHILE CURRENT TRADE RUNNING
        # =================================================

        if open_trade is not None:
            continue


        # =================================================
        # MARKET TIME FILTER
        # =================================================

        if not is_market_time_allowed(
            timestamp,
            tf
        ):

            continue


        # =================================================
        # DAILY TRADE LIMIT
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
        # USE ONLY ACTUAL SIGNAL MARKER
        #
        # Prevents repeated entries
        # during the same continuous trend.
        # =================================================

        signal = str(
            row.get(
                "MARKER",
                ""
            )
        )


        if signal not in [
            "CALL",
            "PUT"
        ]:

            continue


        # =================================================
        # CREATE TRADE LEVELS
        # =================================================

        levels = create_trade_levels(
            signal,
            row
        )


        if levels is None:
            continue


        # =================================================
        # OPEN TRADE
        # =================================================

        open_trade = {

            "type": levels["type"],

            "entry": levels["entry"],

            "stop_loss":
            levels["stop_loss"],

            "risk":
            levels["risk"],

            "target_1":
            levels["target_1"],

            "target_2":
            levels["target_2"],

            "entry_time": str(
                timestamp_to_ist(
                    timestamp
                )
            )
        }


        daily_trade_count[
            date_key
        ] = (

            today_count + 1
        )


    # =====================================================
    # RUNNING TRADE
    # =====================================================

    running_trade = None


    if open_trade is not None:

        last_row = data.iloc[-1]

        current_price = float(
            last_row["Close"]
        )


        if open_trade["type"] == "CALL":

            running_points = (

                current_price

                -

                open_trade["entry"]
            )

        else:

            running_points = (

                open_trade["entry"]

                -

                current_price
            )


        running_trade = {

            "type": open_trade["type"],

            "entry": round(
                open_trade["entry"],
                2
            ),

            "current_price": round(
                current_price,
                2
            ),

            "stop_loss": round(
                open_trade["stop_loss"],
                2
            ),

            "risk": round(
                open_trade["risk"],
                2
            ),

            "target_1": round(
                open_trade["target_1"],
                2
            ),

            "target_2": round(
                open_trade["target_2"],
                2
            ),

            "running_points": round(
                running_points,
                2
            ),

            "entry_time":
            open_trade[
                "entry_time"
            ],

            "status": "RUNNING"
        }


    # =====================================================
    # CLOSED TRADE STATISTICS ONLY
    # =====================================================

    total_trades = len(
        trades
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


    target_1_hits = sum(

        1

        for trade in trades

        if (
            trade["exit_reason"]
            ==
            "TARGET 1:2"
        )
    )


    target_2_hits = sum(

        1

        for trade in trades

        if (
            trade["exit_reason"]
            ==
            "TARGET 1:3"
        )
    )


    net_points = sum(

        trade["points"]

        for trade in trades
    )


    win_rate = 0


    if total_trades > 0:

        win_rate = (

            wins

            /

            total_trades

        ) * 100


    return {

        "trades": trades,

        "total_trades": total_trades,

        "wins": wins,

        "losses": losses,

        "win_rate": round(
            win_rate,
            2
        ),

        "net_points": round(
            net_points,
            2
        ),

        "target_1_hits":
        target_1_hits,

        "target_2_hits":
        target_2_hits,

        "running_trade":
        running_trade
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return r"""
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


<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>


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

    word-break: break-word;
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


.good {

    color:
    #4ade80;
}


.bad {

    color:
    #fb7185;
}


.running {

    color:
    #38bdf8;
}


.small {

    font-size: 12px;

    color:
    #aab7c4;
}


.trade-row {

    padding:
    10px 0;

    border-bottom:
    1px solid #263241;
}


.trade-row:last-child {

    border-bottom: none;
}


@media (max-width: 450px) {

    body {

        padding: 8px;
    }


    .card {

        padding: 10px;
    }


    #chart {

        height: 450px;
    }


    .value {

        font-size: 15px;
    }

}

</style>

</head>


<body>


<h1>
📈 Personal Scalping Scanner
</h1>


<!-- ========================================= -->
<!-- INDEX / TIMEFRAME -->
<!-- ========================================= -->

<div class="card">

<div
id="indices"
class="tf">
</div>


<br>


<div
id="timeframes"
class="tf">
</div>

</div>


<!-- ========================================= -->
<!-- SCANNER -->
<!-- ========================================= -->

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
Trade Status
</div>

<div
id="tradeStatus"
class="value">
-
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
Risk Points
</div>

<div
id="risk"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 1 (1:2)
</div>

<div
id="target1"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 2 (1:3)
</div>

<div
id="target2"
class="value">
-
</div>

</div>


</div>

</div>


<!-- ========================================= -->
<!-- CHART -->
<!-- ========================================= -->

<h2>
📊 Index Chart
</h2>


<div class="card">

<div id="chart"></div>

</div>


<!-- ========================================= -->
<!-- RUNNING TRADE -->
<!-- ========================================= -->

<h2>
🔵 Running Trade
</h2>


<div class="card">

<div
id="runningTrade"
class="small">

No running trade.

</div>

</div>


<!-- ========================================= -->
<!-- BACKTEST -->
<!-- ========================================= -->

<h2>
📈 Backtest (Closed Trades Only)
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
Target 1 / Target 2
</div>

<div
id="targets"
class="value">
-
</div>

</div>


</div>

</div>


<!-- ========================================= -->
<!-- RECENT CLOSED TRADES -->
<!-- ========================================= -->

<div class="card">

<h2>
Recent Closed Trades
</h2>

<div id="trades"></div>

</div>


<script>


// =========================================================
// SETTINGS
// =========================================================

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


// =========================================================
// BUTTONS
// =========================================================

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


// =========================================================
// CREATE CHART
// =========================================================

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
            container.clientHeight,


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
                true,

                secondsVisible:
                false
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
        "VWAP"
    });
}


// =========================================================
// RESIZE
// =========================================================

window.addEventListener(

    "resize",

    () => {


        if (

            chart

            &&

            document.getElementById(
                "chart"
            )

        ) {

            const container =

            document.getElementById(
                "chart"
            );


            chart.applyOptions({

                width:
                container.clientWidth,

                height:
                container.clientHeight
            });
        }
    }
);


// =========================================================
// FORMAT NUMBER
// =========================================================

function formatNumber(value) {


    if (

        value === null

        ||

        value === undefined

        ||

        value === ""

        ||

        Number.isNaN(
            Number(value)
        )

    ) {

        return "-";
    }


    return Number(
        value
    ).toFixed(2);
}


// =========================================================
// LOAD DATA
// =========================================================

async function loadData() {


    try {


        document.getElementById(
            "signal"
        ).textContent =
        "Loading...";


        document.getElementById(
            "tradeStatus"
        ).textContent =
        "Loading...";


        const url =

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
        );


        const response =

        await fetch(

            url,

            {

                cache:
                "no-store"
            }
        );


        let data;


        try {

            data =
            await response.json();

        }

        catch (jsonError) {

            throw new Error(
                "Server returned invalid response"
            );
        }


        if (
            !response.ok
        ) {

            throw new Error(

                data.error

                ||

                "Server error " +

                response.status
            );
        }


        if (
            data.error
        ) {

            throw new Error(
                data.error
            );
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

    catch (error) {


        console.error(
            "LOAD ERROR:",
            error
        );


        const signalElement =

        document.getElementById(
            "signal"
        );


        signalElement.textContent =
        "ERROR";


        signalElement.className =
        "value bad";


        const statusElement =

        document.getElementById(
            "tradeStatus"
        );


        statusElement.textContent =

        error.message

        ||

        "Unable to load market data";


        statusElement.className =
        "value bad";
    }
}


// =========================================================
// UPDATE SCANNER
// =========================================================

function updateScanner(scanner) {


    if (!scanner) {
        return;
    }


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

        signal === "ERROR"

        ?

        "bad"

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
        "risk"
    ).textContent =

    formatNumber(
        scanner.risk
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


    const statusElement =

    document.getElementById(
        "tradeStatus"
    );


    statusElement.textContent =

    scanner.message

    ||

    "WAIT";


    if (
        scanner.trade_allowed
    ) {

        statusElement.className =
        "value call";

    }

    else {

        statusElement.className =
        "value wait";
    }
}


// =========================================================
// UPDATE CHART
// =========================================================

function updateChart(chartData) {


    if (

        !chart

        ||

        !candleSeries
    ) {

        createChart();
    }


    if (

        !chartData

        ||

        chartData.length === 0
    ) {

        return;
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


    if (
        typeof candleSeries.setMarkers
        === "function"
    ) {

        candleSeries.setMarkers(
            markers
        );
    }


    chart.timeScale().fitContent();
}


// =========================================================
// UPDATE BACKTEST
// =========================================================

function updateBacktest(backtest) {


    if (!backtest) {
        return;
    }


    document.getElementById(
        "totalTrades"
    ).textContent =

    backtest.total_trades
    ?? 0;


    document.getElementById(
        "wins"
    ).textContent =

    backtest.wins
    ?? 0;


    document.getElementById(
        "losses"
    ).textContent =

    backtest.losses
    ?? 0;


    document.getElementById(
        "winRate"
    ).textContent =

    formatNumber(
        backtest.win_rate
    )

    +

    "%";


    const netElement =

    document.getElementById(
        "netPoints"
    );


    const netPoints =

    Number(
        backtest.net_points
        ?? 0
    );


    netElement.textContent =

    formatNumber(
        netPoints
    );


    netElement.className =

    "value "

    +

    (

        netPoints >= 0

        ?

        "good"

        :

        "bad"
    );


    document.getElementById(
        "targets"
    ).textContent =

    (

        backtest.target_1_hits
        ?? 0

    )

    +

    " / "

    +

    (

        backtest.target_2_hits
        ?? 0

    );


    // ==============================================
    // RUNNING TRADE
    // ==============================================

    const runningDiv =

    document.getElementById(
        "runningTrade"
    );


    const running =

    backtest.running_trade;


    if (
        running
    ) {


        const pointsClass =

        Number(
            running.running_points
        ) >= 0

        ?

        "good"

        :

        "bad";


        runningDiv.innerHTML =

        "<div class='trade-row'>"

        +

        "<b class='running'>"

        +

        "● "

        +

        running.status

        +

        " "

        +

        running.type

        +

        "</b><br><br>"

        +

        "Entry: "

        +

        formatNumber(
            running.entry
        )

        +

        " | Current: "

        +

        formatNumber(
            running.current_price
        )

        +

        "<br>"

        +

        "SL: "

        +

        formatNumber(
            running.stop_loss
        )

        +

        " | Risk: "

        +

        formatNumber(
            running.risk
        )

        +

        "<br>"

        +

        "Target 1: "

        +

        formatNumber(
            running.target_1
        )

        +

        " | Target 2: "

        +

        formatNumber(
            running.target_2
        )

        +

        "<br><br>"

        +

        "Running Points: "

        +

        "<span class='"

        +

        pointsClass

        +

        "'>"

        +

        formatNumber(
            running.running_points
        )

        +

        "</span>"

        +

        "</div>";

    }


    else {

        runningDiv.innerHTML =

        "No running trade.";
    }


    // ==============================================
    // CLOSED TRADES
    // ==============================================

    const tradesDiv =

    document.getElementById(
        "trades"
    );


    tradesDiv.innerHTML = "";


    const trades =

    [

        ...(backtest.trades || [])

    ]

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


            div.className =
            "trade-row";


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

            "</b> | "

            +

            "Entry: "

            +

            formatNumber(
                trade.entry
            )

            +

            " | SL: "

            +

            formatNumber(
                trade.stop_loss
            )

            +

            " | Exit: "

            +

            formatNumber(
                trade.exit
            )

            +

            "<br>"

            +

            "Risk: "

            +

            formatNumber(
                trade.risk
            )

            +

            " | Points: "

            +

            formatNumber(
                trade.points
            )

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


// =========================================================
// START
// =========================================================

createButtons();

createChart();

loadData();


// Refresh every 60 seconds

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

    try:

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


        if (

            data is None

            or

            data.empty

        ):

            return jsonify({

                "error":

                "Market data not available. "
                "Please try another timeframe or refresh later."

            }), 503


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


    except Exception as e:

        print(
            "API ERROR:",
            e
        )

        traceback.print_exc()


        return jsonify({

            "error":

            "Internal server error: "

            +

            str(e)

        }), 500


# =========================================================
# HEALTH
# =========================================================

@app.route("/api/health")
def api_health():

    return jsonify({

        "status": "ok",

        "message":
        "Personal Scalping Scanner is running"

    })


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False,

        threaded=True
    )
