from flask import Flask, jsonify, request

import yfinance as yf
import pandas as pd
import numpy as np
import traceback


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
    "1wk"
]


# =========================================================
# TRADING RULES
# =========================================================

MAX_TRADES_PER_DAY = 3


# Maximum allowed stop loss

MAX_STOP_LOSS_POINTS = 15.0


# Scalping trade maximum duration

MAX_TRADE_MINUTES = 30


# NSE market time

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15

MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


# No new trade first 10 minutes

OPENING_NO_TRADE_MINUTES = 10


# No new trade last 20 minutes

CLOSING_NO_TRADE_MINUTES = 20


# EMA slope

EMA_SLOPE_LOOKBACK = 5

MIN_EMA_SLOPE = 0.05


# Candle quality

MIN_BODY_PERCENT = 0.40

WICK_RATIO = 1.5


# Risk reward

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
# TIME HELPERS
# =========================================================

def get_market_time(timestamp):

    ts = pd.Timestamp(timestamp)

    if ts.tzinfo is not None:

        try:
            ts = ts.tz_convert(
                "Asia/Kolkata"
            )

        except Exception:
            pass

    return ts


def is_intraday_timeframe(tf):

    return tf in [
        "1m",
        "2m",
        "3m",
        "5m",
        "15m",
        "1h",
        "2h"
    ]


def is_trading_day(timestamp):

    ts = get_market_time(timestamp)

    return ts.weekday() < 5


def can_open_new_trade(timestamp):

    ts = get_market_time(timestamp)

    if ts.weekday() >= 5:
        return False

    current_minutes = (
        ts.hour * 60
        +
        ts.minute
    )


    market_open = (
        MARKET_OPEN_HOUR * 60
        +
        MARKET_OPEN_MINUTE
    )


    market_close = (
        MARKET_CLOSE_HOUR * 60
        +
        MARKET_CLOSE_MINUTE
    )


    allowed_start = (
        market_open
        +
        OPENING_NO_TRADE_MINUTES
    )


    allowed_end = (
        market_close
        -
        CLOSING_NO_TRADE_MINUTES
    )


    if current_minutes < allowed_start:
        return False


    if current_minutes >= allowed_end:
        return False


    return True


def force_close_time(timestamp):

    ts = get_market_time(timestamp)

    current_minutes = (
        ts.hour * 60
        +
        ts.minute
    )


    market_close = (
        MARKET_CLOSE_HOUR * 60
        +
        MARKET_CLOSE_MINUTE
    )


    close_cutoff = (
        market_close
        -
        CLOSING_NO_TRADE_MINUTES
    )


    return current_minutes >= close_cutoff


# =========================================================
# CLEAN DATA
# =========================================================

def clean_columns(data):

    if data is None:
        return None

    if data.empty:
        return None


    data = data.copy()


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


        # Custom timeframe

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
    # OHLC / 4
    #
    # (OPEN + HIGH + LOW + CLOSE) / 4
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


    # Daily VWAP reset

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

        volume.groupby(
            dates
        ).cumsum()

    )


    data["VWAP"] = np.where(

        cumulative_volume > 0,

        cumulative_pv
        /
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
        body
        /
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


    bullish = c > o

    bearish = c < o


    strong_body = (

        body_ratio
        >=
        MIN_BODY_PERCENT

    )


    bullish_hammer = (

        bullish

        and

        lower_wick
        >=
        body * WICK_RATIO

    )


    bearish_hammer = (

        bearish

        and

        upper_wick
        >=
        body * WICK_RATIO

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

        "good_bullish":
        good_bullish,

        "good_bearish":
        good_bearish,

        "body_ratio":
        body_ratio
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

        ema9_slope_pct
        >
        MIN_EMA_SLOPE

        and

        ema15_slope_pct
        >
        0

    )


    bearish = (

        ema9 < ema15

        and

        ema9_slope_pct
        <
        -MIN_EMA_SLOPE

        and

        ema15_slope_pct
        <
        0

    )


    return {

        "bullish": bullish,

        "bearish": bearish

    }


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


    candle = candle_info(row)


    if not candle.get("valid"):

        return "WAIT"


    trend = ema_trend_ok(row)


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
# CALCULATE ALL
# =========================================================

def calculate_all_signals(data):

    if data is None:
        return None


    if data.empty:
        return None


    data = calculate_indicators(
        data
    )


    return data


# =========================================================
# CREATE TRADE
# =========================================================

def create_trade(
    row,
    timestamp,
    signal
):

    entry = float(
        row["Close"]
    )


    low = float(
        row["Low"]
    )


    high = float(
        row["High"]
    )


    if signal == "CALL":

        stop_loss = low

        risk = (
            entry
            -
            stop_loss
        )


        if risk <= 0:
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

            "stop_loss":
            stop_loss,

            "target_1":
            target_1,

            "target_2":
            target_2,

            "risk":
            risk,

            "entry_time":
            str(timestamp),

            "entry_timestamp":
            timestamp
        }


    if signal == "PUT":

        stop_loss = high

        risk = (

            stop_loss

            -

            entry

        )


        if risk <= 0:
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

            "stop_loss":
            stop_loss,

            "target_1":
            target_1,

            "target_2":
            target_2,

            "risk":
            risk,

            "entry_time":
            str(timestamp),

            "entry_timestamp":
            timestamp
        }


    return None


# =========================================================
# CLOSE TRADE
# =========================================================

def close_trade(
    open_trade,
    exit_price,
    exit_reason,
    timestamp
):

    if (
        open_trade["type"]
        ==
        "CALL"
    ):

        points = (

            exit_price

            -

            open_trade["entry"]

        )

    else:

        points = (

            open_trade["entry"]

            -

            exit_price

        )


    result = (

        "WIN"

        if points > 0

        else "LOSS"

    )


    return {

        "type":
        open_trade["type"],

        "entry":
        round(
            open_trade["entry"],
            2
        ),

        "exit":
        round(
            exit_price,
            2
        ),

        "stop_loss":
        round(
            open_trade["stop_loss"],
            2
        ),

        "target_1":
        round(
            open_trade["target_1"],
            2
        ),

        "target_2":
        round(
            open_trade["target_2"],
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
        open_trade["entry_time"],

        "exit_time":
        str(timestamp)
    }


# =========================================================
# BACKTEST
#
# IMPORTANT:
#
# Only CLOSED trades are included
# in win rate and statistics.
#
# Running trade is separate.
# =========================================================

def run_backtest(
    data,
    tf
):

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


    if data is None:
        return empty


    if len(data) < 30:
        return empty


    data = calculate_all_signals(
        data
    )


    if data is None:
        return empty


    if len(data) < 30:
        return empty


    trades = []


    open_trade = None


    daily_trade_count = {}


    # =====================================================
    # LOOP
    # =====================================================

    for i in range(
        20,
        len(data)
    ):

        row = data.iloc[i]

        timestamp = data.index[i]

        market_ts = get_market_time(
            timestamp
        )


        date_key = str(
            market_ts.date()
        )


        # =================================================
        # CHECK RUNNING TRADE
        # =================================================

        if open_trade is not None:

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


            trade_type = (
                open_trade["type"]
            )


            # ---------------------------------------------
            # CALL
            # ---------------------------------------------

            if trade_type == "CALL":

                # Conservative:
                # SL first if same candle hits both

                if low <= open_trade["stop_loss"]:

                    exit_price = (
                        open_trade[
                            "stop_loss"
                        ]
                    )

                    exit_reason = (
                        "STOP LOSS"
                    )


                elif high >= open_trade["target_2"]:

                    exit_price = (
                        open_trade[
                            "target_2"
                        ]
                    )

                    exit_reason = (
                        "TARGET 1:3"
                    )


                elif high >= open_trade["target_1"]:

                    exit_price = (
                        open_trade[
                            "target_1"
                        ]
                    )

                    exit_reason = (
                        "TARGET 1:2"
                    )


            # ---------------------------------------------
            # PUT
            # ---------------------------------------------

            elif trade_type == "PUT":

                if high >= open_trade["stop_loss"]:

                    exit_price = (
                        open_trade[
                            "stop_loss"
                        ]
                    )

                    exit_reason = (
                        "STOP LOSS"
                    )


                elif low <= open_trade["target_2"]:

                    exit_price = (
                        open_trade[
                            "target_2"
                        ]
                    )

                    exit_reason = (
                        "TARGET 1:3"
                    )


                elif low <= open_trade["target_1"]:

                    exit_price = (
                        open_trade[
                            "target_1"
                        ]
                    )

                    exit_reason = (
                        "TARGET 1:2"
                    )


            # =================================================
            # SCALPING TIME LIMIT
            # =================================================

            entry_ts = pd.Timestamp(
                open_trade[
                    "entry_timestamp"
                ]
            )


            elapsed_minutes = (

                pd.Timestamp(timestamp)

                -

                entry_ts

            ).total_seconds() / 60


            if (

                exit_price is None

                and

                elapsed_minutes
                >=
                MAX_TRADE_MINUTES

            ):

                exit_price = close

                exit_reason = (
                    "SCALPING TIME EXIT"
                )


            # =================================================
            # MARKET CLOSE EXIT
            # =================================================

            if (

                exit_price is None

                and

                is_intraday_timeframe(tf)

                and

                force_close_time(
                    timestamp
                )

            ):

                exit_price = close

                exit_reason = (
                    "MARKET TIME EXIT"
                )


            # =================================================
            # CLOSE TRADE
            # =================================================

            if exit_price is not None:

                trade = close_trade(

                    open_trade,

                    exit_price,

                    exit_reason,

                    timestamp

                )


                trades.append(
                    trade
                )


                open_trade = None


        # =================================================
        # DO NOT OPEN IF RUNNING TRADE EXISTS
        # =================================================

        if open_trade is not None:
            continue


        # =================================================
        # INTRADAY MARKET TIME FILTER
        # =================================================

        if is_intraday_timeframe(tf):

            if not can_open_new_trade(
                timestamp
            ):
                continue


        # =================================================
        # MAX TRADES PER DAY
        # =================================================

        today_count = (
            daily_trade_count.get(
                date_key,
                0
            )
        )


        if (
            today_count
            >=
            MAX_TRADES_PER_DAY
        ):

            continue


        # =================================================
        # GET SIGNAL
        # =================================================

        signal = get_signal(
            row
        )


        if signal == "WAIT":
            continue


        # =================================================
        # CREATE TRADE
        # =================================================

        new_trade = create_trade(

            row,

            timestamp,

            signal

        )


        if new_trade is None:
            continue


        open_trade = new_trade


        daily_trade_count[
            date_key
        ] = (

            today_count
            +
            1

        )


    # =====================================================
    # RUNNING TRADE
    #
    # Do NOT add it to backtest statistics.
    # =====================================================

    running_trade = None


    if open_trade is not None:

        last_row = data.iloc[-1]

        current_price = float(
            last_row["Close"]
        )


        if (
            open_trade["type"]
            ==
            "CALL"
        ):

            live_points = (

                current_price

                -

                open_trade["entry"]

            )

        else:

            live_points = (

                open_trade["entry"]

                -

                current_price

            )


        entry_ts = pd.Timestamp(
            open_trade[
                "entry_timestamp"
            ]
        )


        last_ts = pd.Timestamp(
            data.index[-1]
        )


        elapsed_minutes = (

            last_ts

            -

            entry_ts

        ).total_seconds() / 60


        running_trade = {

            "status":
            "RUNNING",

            "type":
            open_trade["type"],

            "entry":
            round(
                open_trade["entry"],
                2
            ),

            "current_price":
            round(
                current_price,
                2
            ),

            "stop_loss":
            round(
                open_trade["stop_loss"],
                2
            ),

            "target_1":
            round(
                open_trade["target_1"],
                2
            ),

            "target_2":
            round(
                open_trade["target_2"],
                2
            ),

            "risk":
            round(
                open_trade["risk"],
                2
            ),

            "live_points":
            round(
                live_points,
                2
            ),

            "entry_time":
            open_trade["entry_time"],

            "minutes_running":
            round(
                elapsed_minutes,
                1
            )
        }


    # =====================================================
    # STATISTICS
    #
    # ONLY CLOSED TRADES
    # =====================================================

    total_trades = len(
        trades
    )


    wins = sum(

        1

        for trade in trades

        if (
            trade["result"]
            ==
            "WIN"
        )

    )


    losses = sum(

        1

        for trade in trades

        if (
            trade["result"]
            ==
            "LOSS"
        )

    )


    net_points = sum(

        trade["points"]

        for trade in trades

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


    win_rate = 0


    if total_trades > 0:

        win_rate = (

            wins
            /
            total_trades

        ) * 100


    return {

        "trades":
        trades,

        "total_trades":
        total_trades,

        "wins":
        wins,

        "losses":
        losses,

        "win_rate":
        round(
            win_rate,
            2
        ),

        "net_points":
        round(
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
# SCANNER
# =========================================================

def calculate_scanner(data):

    empty_result = {

        "signal": "WAIT",

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
        data
    )


    if data is None:
        return empty_result


    row = data.iloc[-1]


    signal = get_signal(
        row
    )


    price = float(
        row["Close"]
    )


    low = float(
        row["Low"]
    )


    high = float(
        row["High"]
    )


    stop_loss = None

    target_1 = None

    target_2 = None


    if signal == "CALL":

        risk = (
            price
            -
            low
        )


        if (
            risk > 0

            and

            risk <= MAX_STOP_LOSS_POINTS

        ):

            stop_loss = low

            target_1 = (

                price
                +
                risk * RR_1

            )

            target_2 = (

                price
                +
                risk * RR_2

            )

        else:

            signal = "WAIT"


    elif signal == "PUT":

        risk = (
            high
            -
            price
        )


        if (

            risk > 0

            and

            risk <= MAX_STOP_LOSS_POINTS

        ):

            stop_loss = high

            target_1 = (

                price
                -
                risk * RR_1

            )

            target_2 = (

                price
                -
                risk * RR_2

            )

        else:

            signal = "WAIT"


    return {

        "signal":
        signal,

        "price":
        round(
            price,
            2
        ),

        "ema9":
        round(
            float(row["EMA9"]),
            2
        ),

        "ema15":
        round(
            float(row["EMA15"]),
            2
        ),

        "vwap":
        round(
            float(row["VWAP"]),
            2
        ),

        "stop_loss":

        round(
            stop_loss,
            2
        )

        if stop_loss is not None

        else None,


        "target_1":

        round(
            target_1,
            2
        )

        if target_1 is not None

        else None,


        "target_2":

        round(
            target_2,
            2
        )

        if target_2 is not None

        else None,


        "time":
        str(
            data.index[-1]
        )
    }


# =========================================================
# CHART JSON
# =========================================================

def chart_json(
    data,
    trades,
    running_trade
):

    if data is None:
        return []


    if data.empty:
        return []


    data = calculate_all_signals(
        data
    )


    result = []


    # =====================================================
    # TRADE ENTRY MARKERS ONLY
    # =====================================================

    marker_times = {}


    for trade in trades:

        try:

            ts = int(

                pd.Timestamp(
                    trade["entry_time"]
                ).timestamp()

            )


            marker_times[ts] = (
                trade["type"]
            )


        except Exception:
            pass


    if running_trade is not None:

        try:

            ts = int(

                pd.Timestamp(
                    running_trade[
                        "entry_time"
                    ]
                ).timestamp()

            )


            marker_times[ts] = (
                running_trade[
                    "type"
                ]
            )


        except Exception:
            pass


    # =====================================================
    # DATA
    # =====================================================

    for timestamp, row in data.iterrows():

        try:

            ts = int(

                pd.Timestamp(
                    timestamp
                ).timestamp()

            )


            marker = marker_times.get(
                ts,
                ""
            )


            result.append({

                "time":
                ts,

                "open":
                round(
                    float(row["Open"]),
                    2
                ),

                "high":
                round(
                    float(row["High"]),
                    2
                ),

                "low":
                round(
                    float(row["Low"]),
                    2
                ),

                "close":
                round(
                    float(row["Close"]),
                    2
                ),

                "ema9":
                round(
                    float(row["EMA9"]),
                    2
                ),

                "ema15":
                round(
                    float(row["EMA15"]),
                    2
                ),

                "vwap":
                round(
                    float(row["VWAP"]),
                    2
                ),

                "marker":
                marker
            })


        except Exception as e:

            print(
                "CHART ERROR:",
                e
            )


    return result


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


<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>


<style>

* {

    box-sizing:
    border-box;

}


body {

    margin: 0;

    padding: 10px;

    background:
    #080c12;

    color:
    #ffffff;

    font-family:
    Arial,
    sans-serif;

}


h1 {

    font-size:
    18px;

    margin:
    8px 0 14px;

}


h2 {

    font-size:
    16px;

    margin:
    18px 0 10px;

}


.card {

    background:
    #111923;

    border:
    1px solid #263241;

    border-radius:
    12px;

    padding:
    12px;

    margin-bottom:
    12px;

}


.tf {

    display:
    flex;

    gap:
    6px;

    overflow-x:
    auto;

    padding-bottom:
    6px;

}


button {

    padding:
    8px 11px;

    border-radius:
    8px;

    border:
    1px solid #34465a;

    background:
    #172331;

    color:
    white;

    cursor:
    pointer;

    white-space:
    nowrap;

}


button.active {

    background:
    #2463eb;

}


.grid {

    display:
    grid;

    grid-template-columns:
    repeat(2, 1fr);

    gap:
    9px;

}


.box {

    background:
    #172331;

    padding:
    11px;

    border-radius:
    8px;

}


.label {

    font-size:
    11px;

    color:
    #aab7c4;

}


.value {

    font-size:
    16px;

    margin-top:
    5px;

    font-weight:
    bold;

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
    #60a5fa;

}


.small {

    font-size:
    11px;

    color:
    #aab7c4;

}


#chart {

    width:
    100%;

    height:
    430px;

}


.running-card {

    border:
    1px solid #2563eb;

}


.status-running {

    font-size:
    17px;

    font-weight:
    bold;

    color:
    #60a5fa;

    margin-bottom:
    12px;

}


.progress-wrap {

    margin-top:
    12px;

}


.progress-bar {

    width:
    100%;

    height:
    10px;

    background:
    #263241;

    border-radius:
    10px;

    overflow:
    hidden;

}


.progress-fill {

    height:
    100%;

    background:
    #2563eb;

    width:
    0%;

}


.trade-row {

    padding:
    10px 0;

    border-bottom:
    1px solid #263241;

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


<!-- SCANNER -->

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


<!-- RUNNING TRADE -->

<div
id="runningTradeCard"
class="card running-card"
style="display:none;">

<div
class="status-running">

🔵 TRADE RUNNING

</div>


<div class="grid">


<div class="box">

<div class="label">
Type
</div>

<div
id="runningType"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Live Points
</div>

<div
id="livePoints"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Entry
</div>

<div
id="runningEntry"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Current Price
</div>

<div
id="runningPrice"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Stop Loss
</div>

<div
id="runningSL"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Risk
</div>

<div
id="runningRisk"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 1
</div>

<div
id="runningT1"
class="value">
-
</div>

</div>


<div class="box">

<div class="label">
Target 2
</div>

<div
id="runningT2"
class="value">
-
</div>

</div>


</div>


<div class="progress-wrap">

<div class="label">
Target 1 Progress
</div>

<div class="progress-bar">

<div
id="progressFill"
class="progress-fill">
</div>

</div>


<div
id="progressText"
class="small"
style="margin-top:6px;">
-
</div>

</div>


</div>


<!-- CHART -->

<h2>
📊 Index Chart
</h2>


<div class="card">

<div id="chart"></div>

</div>


<!-- BACKTEST -->

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


<!-- RECENT CLOSED TRADES -->

<div class="card">

<h2>
Recent Closed Trades
</h2>

<div id="trades"></div>

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
            430,

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
        "VWAP"

    });


}


window.addEventListener(
    "resize",

    () => {

        if (
            chart
        ) {

            chart.applyOptions({

                width:
                document.getElementById(
                    "chart"
                ).clientWidth

            });

        }

    }

);


function formatNumber(
    value
) {

    if (

        value === null

        ||

        value === undefined

        ||

        value === ""

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


        updateRunningTrade(
            data.backtest.running_trade
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
    scanner.signal ||
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


function updateRunningTrade(
    trade
) {


    const card =
    document.getElementById(
        "runningTradeCard"
    );


    if (
        !trade
    ) {

        card.style.display =
        "none";

        return;

    }


    card.style.display =
    "block";


    const typeElement =
    document.getElementById(
        "runningType"
    );


    typeElement.textContent =
    trade.type;


    typeElement.className =

    "value "

    +

    (

        trade.type === "CALL"

        ?

        "call"

        :

        "put"

    );


    const liveElement =
    document.getElementById(
        "livePoints"
    );


    liveElement.textContent =

    formatNumber(
        trade.live_points
    )

    +

    " pts";


    liveElement.className =

    "value "

    +

    (

        trade.live_points >= 0

        ?

        "good"

        :

        "bad"

    );


    document.getElementById(
        "runningEntry"
    ).textContent =
    formatNumber(
        trade.entry
    );


    document.getElementById(
        "runningPrice"
    ).textContent =
    formatNumber(
        trade.current_price
    );


    document.getElementById(
        "runningSL"
    ).textContent =
    formatNumber(
        trade.stop_loss
    );


    document.getElementById(
        "runningRisk"
    ).textContent =

    formatNumber(
        trade.risk
    )

    +

    " pts";


    document.getElementById(
        "runningT1"
    ).textContent =
    formatNumber(
        trade.target_1
    );


    document.getElementById(
        "runningT2"
    ).textContent =
    formatNumber(
        trade.target_2
    );


    let progress = 0;


    if (
        trade.type === "CALL"
    ) {

        progress =

        (

            trade.current_price

            -

            trade.entry

        )

        /

        (

            trade.target_1

            -

            trade.entry

        )

        *
        100;

    }

    else {

        progress =

        (

            trade.entry

            -

            trade.current_price

        )

        /

        (

            trade.entry

            -

            trade.target_1

        )

        *
        100;

    }


    progress = Math.max(

        0,

        Math.min(
            100,
            progress
        )

    );


    document.getElementById(
        "progressFill"
    ).style.width =

    progress

    +

    "%";


    document.getElementById(
        "progressText"
    ).textContent =

    progress.toFixed(1)

    +

    "% to Target 1 • "

    +

    trade.minutes_running

    +

    " min running";

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


    if (
        candleSeries.setMarkers
    ) {

        candleSeries.setMarkers(
            markers
        );

    }


    chart.timeScale().fitContent();

}


function updateBacktest(
    backtest
) {


    document.getElementById(
        "totalTrades"
    ).textContent =
    backtest.total_trades;


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
    formatNumber(
        backtest.net_points
    );


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

            "</b> &nbsp; "

            +

            "Entry: "

            +

            trade.entry

            +

            " &nbsp; SL: "

            +

            trade.stop_loss

            +

            "<br>"

            +

            "Exit: "

            +

            trade.exit

            +

            " &nbsp; Points: "

            +

            trade.points

            +

            " &nbsp; "

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


    if (
        data is None

        or

        data.empty
    ):

        return jsonify({

            "error":
            "Market data not available"

        }), 500


    scanner = calculate_scanner(
        data
    )


    backtest = run_backtest(

        data,

        tf

    )


    chart = chart_json(

        data,

        backtest["trades"],

        backtest["running_trade"]

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
