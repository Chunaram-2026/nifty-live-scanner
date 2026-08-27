from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import os
import math

app = Flask(__name__)


# =========================================================
# SETTINGS
# =========================================================

MAX_TRADES_PER_DAY = 3

RISK_REWARD = 2.0

EMA_ANGLE_LOOKBACK = 3

EMA_MIN_ANGLE = 30


# =========================================================
# INDEX SYMBOLS
# =========================================================

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN"
}


# =========================================================
# TIMEFRAMES
# =========================================================

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

    # MultiIndex fix
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

    for col in required:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    ).fillna(0)

    data = data.dropna(
        subset=required
    )

    return data


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data(symbol, tf):

    interval, period, resample_rule = \
        timeframe_settings(tf)

    try:

        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        data = clean_columns(data)

        if data is None:
            return None

        # =============================================
        # RESAMPLE
        # =============================================

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

    volume = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    ).fillna(0)

    # =================================================
    # EMA 9
    # =================================================

    data["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    # =================================================
    # EMA 15
    # =================================================

    data["EMA15"] = close.ewm(
        span=15,
        adjust=False
    ).mean()

    # =================================================
    # TYPICAL PRICE
    # =================================================

    typical_price = (
        high +
        low +
        close
    ) / 3

    # =================================================
    # VWAP
    # Reset every trading day
    # =================================================

    try:

        data["_date"] = data.index.date

        cumulative_pv = (
            typical_price * volume
        ).groupby(
            data["_date"]
        ).cumsum()

        cumulative_volume = (
            volume.groupby(
                data["_date"]
            ).cumsum()
        )

        data["VWAP"] = np.where(

            cumulative_volume > 0,

            cumulative_pv /
            cumulative_volume,

            typical_price

        )

        data.drop(
            columns=["_date"],
            inplace=True
        )

    except Exception:

        data["VWAP"] = typical_price

    # =================================================
    # ATR / AVERAGE RANGE
    # =================================================

    data["RANGE"] = (
        high - low
    ).abs()

    data["AVG_RANGE"] = (
        data["RANGE"]
        .rolling(14)
        .mean()
    )

    data["AVG_RANGE"] = (
        data["AVG_RANGE"]
        .fillna(data["RANGE"])
    )

    return data


# =========================================================
# EMA ANGLE
#
# Chart screen का exact degree code से possible नहीं होता,
# इसलिए candle range के हिसाब से normalized angle लिया गया है.
# इससे लगभग 30° upward / downward trend filter मिलता है.
# =========================================================

def get_ema_angle(
    data,
    column,
    index
):

    if index < EMA_ANGLE_LOOKBACK:

        return 0.0

    try:

        current = float(
            data[column].iloc[index]
        )

        previous = float(
            data[column].iloc[
                index -
                EMA_ANGLE_LOOKBACK
            ]
        )

        avg_range = float(
            data["AVG_RANGE"]
            .iloc[index]
        )

        if avg_range <= 0:

            return 0.0

        slope = (

            current -
            previous

        ) / (

            avg_range *
            EMA_ANGLE_LOOKBACK

        )

        angle = math.degrees(
            math.atan(slope)
        )

        return round(
            angle,
            2
        )

    except Exception:

        return 0.0


# =========================================================
# CANDLE QUALITY
# =========================================================

def candle_details(row):

    try:

        open_price = float(
            row["Open"]
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

    except Exception:

        return None

    candle_range = high - low

    if candle_range <= 0:

        return None

    body = abs(
        close - open_price
    )

    upper_wick = high - max(
        open_price,
        close
    )

    lower_wick = min(
        open_price,
        close
    ) - low

    body_ratio = (
        body /
        candle_range
    )

    return {

        "open": open_price,

        "high": high,

        "low": low,

        "close": close,

        "range": candle_range,

        "body": body,

        "upper_wick": upper_wick,

        "lower_wick": lower_wick,

        "body_ratio": body_ratio,

        "bullish":
            close > open_price,

        "bearish":
            close < open_price

    }


# =========================================================
# STRONG BULLISH CANDLE
#
# Full body bullish
# OR Hammer
# =========================================================

def is_good_bullish_candle(row):

    c = candle_details(row)

    if c is None:

        return False

    # Strong bullish body

    strong_body = (

        c["bullish"]

        and

        c["body_ratio"] >= 0.55

    )

    # Hammer

    hammer = (

        c["bullish"]

        and

        c["lower_wick"] >=
        c["body"] * 1.5

        and

        c["upper_wick"] <=
        max(
            c["body"],
            c["range"] * 0.25
        )

    )

    return (

        strong_body
        or
        hammer

    )


# =========================================================
# STRONG BEARISH CANDLE
#
# Full body bearish
# OR Shooting Star
# =========================================================

def is_good_bearish_candle(row):

    c = candle_details(row)

    if c is None:

        return False

    # Strong bearish body

    strong_body = (

        c["bearish"]

        and

        c["body_ratio"] >= 0.55

    )

    # Shooting star / upper rejection

    shooting_star = (

        c["bearish"]

        and

        c["upper_wick"] >=
        c["body"] * 1.5

        and

        c["lower_wick"] <=
        max(
            c["body"],
            c["range"] * 0.25
        )

    )

    return (

        strong_body
        or
        shooting_star

    )


# =========================================================
# SIGNAL
#
# NO LIQUIDITY SWEEP
#
# CALL
#
# EMA9 > EMA15
# Both EMA upward near 30 degree
# VWAP below both EMA
# Good bullish candle
#
# PUT
#
# EMA9 < EMA15
# Both EMA downward near -30 degree
# VWAP above both EMA
# Good bearish candle
# =========================================================

def get_signal(
    data,
    index
):

    if data is None:

        return "WAIT"

    if index < max(
        20,
        EMA_ANGLE_LOOKBACK
    ):

        return "WAIT"

    try:

        row = data.iloc[index]

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

    # =================================================
    # EMA ANGLES
    # =================================================

    ema9_angle = get_ema_angle(

        data,
        "EMA9",
        index

    )

    ema15_angle = get_ema_angle(

        data,
        "EMA15",
        index

    )

    # =================================================
    # CALL CONDITIONS
    # =================================================

    bullish_ema_structure = (

        ema9 > ema15

    )

    bullish_angle = (

        ema9_angle >= EMA_MIN_ANGLE

        and

        ema15_angle >= (
            EMA_MIN_ANGLE * 0.70
        )

    )

    bullish_vwap = (

        vwap < ema9

        and

        vwap < ema15

    )

    bullish_candle = (
        is_good_bullish_candle(row)
    )

    if (

        bullish_ema_structure

        and

        bullish_angle

        and

        bullish_vwap

        and

        bullish_candle

    ):

        return "CALL"

    # =================================================
    # PUT CONDITIONS
    # =================================================

    bearish_ema_structure = (

        ema9 < ema15

    )

    bearish_angle = (

        ema9_angle <= -EMA_MIN_ANGLE

        and

        ema15_angle <= (
            -EMA_MIN_ANGLE * 0.70
        )

    )

    bearish_vwap = (

        vwap > ema9

        and

        vwap > ema15

    )

    bearish_candle = (
        is_good_bearish_candle(row)
    )

    if (

        bearish_ema_structure

        and

        bearish_angle

        and

        bearish_vwap

        and

        bearish_candle

    ):

        return "PUT"

    return "WAIT"


# =========================================================
# ADD SIGNAL MARKERS
# =========================================================

def add_signal_markers(data):

    if data is None:
        return None

    if data.empty:
        return data

    data = data.copy()

    markers = []

    previous_signal = "WAIT"

    for i in range(len(data)):

        signal = get_signal(
            data,
            i
        )

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

        markers.append(marker)

        previous_signal = signal

    data["MARKER"] = markers

    return data


# =========================================================
# CALCULATE ALL
# =========================================================

def calculate_all_signals(data):

    data = calculate_indicators(data)

    if data is None:

        return None

    data = add_signal_markers(data)

    return data


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(data):

    if data is None:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None,
            "time": None
        }

    if len(data) < 25:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None,
            "time": None
        }

    data = calculate_all_signals(data)

    if data is None:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None,
            "time": None
        }

    last_index = len(data) - 1

    row = data.iloc[-1]

    signal = get_signal(
        data,
        last_index
    )

    ema9_angle = get_ema_angle(
        data,
        "EMA9",
        last_index
    )

    ema15_angle = get_ema_angle(
        data,
        "EMA15",
        last_index
    )

    return {

        "signal": signal,

        "price": round(
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

        "ema9_angle": ema9_angle,

        "ema15_angle": ema15_angle,

        "time": str(
            data.index[-1]
        )

    }


# =========================================================
# CHART JSON
# =========================================================

def chart_json(data):

    if data is None:
        return []

    if data.empty:
        return []

    data = calculate_all_signals(data)

    if data is None:
        return []

    result = []

    for i, (
        timestamp,
        row
    ) in enumerate(
        data.iterrows()
    ):

        try:

            ts = int(
                pd.Timestamp(
                    timestamp
                ).timestamp()
            )

            signal = get_signal(
                data,
                i
            )

            marker = row.get(
                "MARKER",
                ""
            )

            result.append({

                "time": ts,

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

                "signal": signal,

                "marker": marker

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
# ENTRY = Signal candle CLOSE
#
# CALL:
# SL = Signal candle LOW
# TP = Entry + Risk × R:R
#
# PUT:
# SL = Signal candle HIGH
# TP = Entry - Risk × R:R
#
# MAXIMUM 3 TRADES PER DAY
# =========================================================

def run_backtest(data):

    empty_result = {

        "trades": [],

        "total_trades": 0,

        "wins": 0,

        "losses": 0,

        "win_rate": 0,

        "net_points": 0

    }

    if data is None:

        return empty_result

    if len(data) < 25:

        return empty_result

    data = calculate_all_signals(data)

    if data is None:

        return empty_result

    if len(data) < 25:

        return empty_result

    trades = []

    in_position = False

    current_trade = None

    trades_per_day = {}

    # =================================================
    # LOOP
    # =================================================

    i = 20

    while i < len(data):

        row = data.iloc[i]

        timestamp = data.index[i]

        trade_date = str(
            pd.Timestamp(
                timestamp
            ).date()
        )

        # =============================================
        # OPEN NEW TRADE
        # =============================================

        if not in_position:

            today_trades = trades_per_day.get(
                trade_date,
                0
            )

            # Maximum 3 trades/day

            if today_trades < MAX_TRADES_PER_DAY:

                signal = get_signal(
                    data,
                    i
                )

                if signal == "CALL":

                    entry = float(
                        row["Close"]
                    )

                    stop_loss = float(
                        row["Low"]
                    )

                    risk = (
                        entry -
                        stop_loss
                    )

                    if risk > 0:

                        target = (

                            entry +

                            risk *
                            RISK_REWARD

                        )

                        current_trade = {

                            "type": "CALL",

                            "entry": entry,

                            "stop_loss": stop_loss,

                            "target": target,

                            "entry_time": timestamp,

                            "entry_index": i

                        }

                        in_position = True

                        trades_per_day[
                            trade_date
                        ] = (

                            today_trades + 1

                        )

                elif signal == "PUT":

                    entry = float(
                        row["Close"]
                    )

                    stop_loss = float(
                        row["High"]
                    )

                    risk = (

                        stop_loss -
                        entry

                    )

                    if risk > 0:

                        target = (

                            entry -

                            risk *
                            RISK_REWARD

                        )

                        current_trade = {

                            "type": "PUT",

                            "entry": entry,

                            "stop_loss": stop_loss,

                            "target": target,

                            "entry_time": timestamp,

                            "entry_index": i

                        }

                        in_position = True

                        trades_per_day[
                            trade_date
                        ] = (

                            today_trades + 1

                        )

        # =============================================
        # MANAGE CALL
        # Start checking from next candle
        # =============================================

        elif (

            current_trade["type"] == "CALL"

            and

            i > current_trade["entry_index"]

        ):

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            entry = current_trade[
                "entry"
            ]

            stop_loss = current_trade[
                "stop_loss"
            ]

            target = current_trade[
                "target"
            ]

            # Conservative rule:
            # If both SL and TP hit in same candle,
            # SL is assumed first.

            if low <= stop_loss:

                exit_price = stop_loss

                points = (

                    exit_price -
                    entry

                )

                trades.append({

                    "type": "CALL",

                    "entry": round(
                        entry,
                        2
                    ),

                    "exit": round(
                        exit_price,
                        2
                    ),

                    "points": round(
                        points,
                        2
                    ),

                    "result": "LOSS",

                    "reason": "STOP LOSS",

                    "entry_time": str(
                        current_trade[
                            "entry_time"
                        ]
                    ),

                    "exit_time": str(
                        timestamp
                    )

                })

                in_position = False

                current_trade = None

            elif high >= target:

                exit_price = target

                points = (

                    exit_price -
                    entry

                )

                trades.append({

                    "type": "CALL",

                    "entry": round(
                        entry,
                        2
                    ),

                    "exit": round(
                        exit_price,
                        2
                    ),

                    "points": round(
                        points,
                        2
                    ),

                    "result": "WIN",

                    "reason": "TARGET",

                    "entry_time": str(
                        current_trade[
                            "entry_time"
                        ]
                    ),

                    "exit_time": str(
                        timestamp
                    )

                })

                in_position = False

                current_trade = None

        # =============================================
        # MANAGE PUT
        # =============================================

        elif (

            current_trade["type"] == "PUT"

            and

            i > current_trade["entry_index"]

        ):

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            entry = current_trade[
                "entry"
            ]

            stop_loss = current_trade[
                "stop_loss"
            ]

            target = current_trade[
                "target"
            ]

            # Conservative rule

            if high >= stop_loss:

                exit_price = stop_loss

                points = (

                    entry -
                    exit_price

                )

                trades.append({

                    "type": "PUT",

                    "entry": round(
                        entry,
                        2
                    ),

                    "exit": round(
                        exit_price,
                        2
                    ),

                    "points": round(
                        points,
                        2
                    ),

                    "result": "LOSS",

                    "reason": "STOP LOSS",

                    "entry_time": str(
                        current_trade[
                            "entry_time"
                        ]
                    ),

                    "exit_time": str(
                        timestamp
                    )

                })

                in_position = False

                current_trade = None

            elif low <= target:

                exit_price = target

                points = (

                    entry -
                    exit_price

                )

                trades.append({

                    "type": "PUT",

                    "entry": round(
                        entry,
                        2
                    ),

                    "exit": round(
                        exit_price,
                        2
                    ),

                    "points": round(
                        points,
                        2
                    ),

                    "result": "WIN",

                    "reason": "TARGET",

                    "entry_time": str(
                        current_trade[
                            "entry_time"
                        ]
                    ),

                    "exit_time": str(
                        timestamp
                    )

                })

                in_position = False

                current_trade = None

        i += 1

    # =================================================
    # CLOSE LAST OPEN TRADE
    # =================================================

    if (

        in_position

        and

        current_trade is not None

    ):

        last_row = data.iloc[-1]

        last_price = float(
            last_row["Close"]
        )

        last_time = data.index[-1]

        if current_trade["type"] == "CALL":

            points = (

                last_price -
                current_trade["entry"]

            )

        else:

            points = (

                current_trade["entry"] -
                last_price

            )

        result = (

            "WIN"

            if points > 0

            else "LOSS"

        )

        trades.append({

            "type":
                current_trade["type"],

            "entry": round(
                current_trade["entry"],
                2
            ),

            "exit": round(
                last_price,
                2
            ),

            "points": round(
                points,
                2
            ),

            "result": result,

            "reason": "MARKET CLOSE",

            "entry_time": str(
                current_trade[
                    "entry_time"
                ]
            ),

            "exit_time": str(
                last_time
            )

        })

    # =================================================
    # STATISTICS
    # =================================================

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

    net_points = sum(

        trade["points"]

        for trade in trades

    )

    if total_trades > 0:

        win_rate = (

            wins /
            total_trades

        ) * 100

    else:

        win_rate = 0

    return {

        "trades": trades,

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
            )

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
content="width=device-width,initial-scale=1">

<title>
Personal Scalping Scanner
</title>

<script
src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js">
</script>


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
    8px 0
    16px;

}

h2 {

    font-size: 18px;

    margin:
    20px 0
    10px;

}

.card {

    background: #111923;

    border:
    1px solid #263241;

    border-radius: 12px;

    padding: 14px;

    margin-bottom: 12px;

}

.section-title {

    font-size: 13px;

    color: #aeb9c7;

    margin-bottom: 8px;

}

.row {

    display: flex;

    gap: 8px;

    flex-wrap: wrap;

}

button {

    border:
    1px solid #354252;

    background: #182230;

    color: #ffffff;

    border-radius: 8px;

    padding:
    9px 13px;

    font-size: 14px;

    cursor: pointer;

}

button.active {

    background: #6d3fd1;

    border-color: #7d51e0;

}

.signal {

    font-size: 24px;

    font-weight: bold;

    margin-bottom: 10px;

}

.wait {

    color: #f4c542;

}

.call {

    color: #36d98c;

}

.put {

    color: #ff6b6b;

}

.data-line {

    margin:
    6px 0;

    font-size: 15px;

}

.small {

    color: #8d9aaa;

    font-size: 12px;

}

#chart {

    width: 100%;

    height: 430px;

}

.stats {

    display:
    grid;

    grid-template-columns:
    1fr 1fr;

    gap: 10px;

}

.stat {

    background: #17212d;

    border-radius: 8px;

    padding: 12px;

}

.stat-label {

    color: #9ba7b5;

    font-size: 12px;

}

.stat-value {

    font-size: 19px;

    margin-top: 6px;

}

.trade {

    border-bottom:
    1px solid #263241;

    padding:
    10px 0;

    font-size: 13px;

}

.win {

    color: #4ee39b;

}

.loss {

    color: #ff7070;

}

.legend {

    margin-top: 10px;

    font-size: 12px;

    color: #aeb9c7;

}

.ema9 {

    color: #f4c542;

}

.ema15 {

    color: #46bdf0;

}

.vwap {

    color: #bd7df5;

}

.info {

    line-height: 1.6;

    font-size: 13px;

    color: #b8c1cc;

}

</style>

</head>


<body>


<h1>
⚡ Personal Scalping Scanner
</h1>


<div class="card">

<div class="section-title">
TIMEFRAME
</div>

<div
class="row"
id="timeframes">

<button
class="tf-btn"
data-tf="1m">
1M
</button>

<button
class="tf-btn"
data-tf="2m">
2M
</button>

<button
class="tf-btn active"
data-tf="3m">
3M
</button>

<button
class="tf-btn"
data-tf="5m">
5M
</button>

<button
class="tf-btn"
data-tf="15m">
15M
</button>

<button
class="tf-btn"
data-tf="1h">
1H
</button>

<button
class="tf-btn"
data-tf="2h">
2H
</button>

<button
class="tf-btn"
data-tf="1d">
1D
</button>

<button
class="tf-btn"
data-tf="1wk">
1W
</button>

</div>

</div>


<div class="card">

<div class="section-title">
INDEX
</div>

<div
class="row"
id="indices">

<button
class="index-btn active"
data-index="NIFTY 50">
NIFTY 50
</button>

<button
class="index-btn"
data-index="BANK NIFTY">
BANK NIFTY
</button>

<button
class="index-btn"
data-index="SENSEX">
SENSEX
</button>

</div>

</div>


<div class="card">

<div class="section-title">
LIVE SIGNAL
</div>

<div
id="signal"
class="signal wait">
WAIT
</div>

<div
id="scanner">
Loading...
</div>

</div>


<h2>
📊 Index Chart
</h2>

<div class="card">

<div id="chart"></div>

<div class="legend">

<span class="ema9">
● EMA 9
</span>

&nbsp;&nbsp;

<span class="ema15">
● EMA 15
</span>

&nbsp;&nbsp;

<span class="vwap">
● VWAP
</span>

&nbsp;&nbsp;

<span>
▲ CALL
</span>

&nbsp;&nbsp;

<span>
▼ PUT
</span>

</div>

</div>


<h2>
📈 Backtest
</h2>

<div class="card">

<div class="stats">

<div class="stat">

<div class="stat-label">
Total Trades
</div>

<div
id="totalTrades"
class="stat-value">
0
</div>

</div>


<div class="stat">

<div class="stat-label">
Wins
</div>

<div
id="wins"
class="stat-value win">
0
</div>

</div>


<div class="stat">

<div class="stat-label">
Losses
</div>

<div
id="losses"
class="stat-value loss">
0
</div>

</div>


<div class="stat">

<div class="stat-label">
Win Rate
</div>

<div
id="winRate"
class="stat-value">
0%
</div>

</div>


<div class="stat">

<div class="stat-label">
Net Points
</div>

<div
id="netPoints"
class="stat-value">
0
</div>

</div>

</div>


<h2>
Recent Trades
</h2>

<div id="trades">
Loading...
</div>

</div>


<div class="card">

<div class="info">

<b>Strategy Rules</b>

<br>

CALL:
EMA 9 ऊपर EMA 15,
दोनों upward लगभग 30°,
VWAP दोनों EMA के नीचे,
strong bullish candle / hammer.

<br>

PUT:
EMA 9 नीचे EMA 15,
दोनों downward लगभग 30°,
VWAP दोनों EMA के ऊपर,
strong bearish candle / shooting star.

<br>

Stop Loss:
Signal candle का Low / High.

<br>

Risk Reward:
1:2

<br>

Maximum:
3 trades per day.

</div>

</div>


<script>


let selectedTF = "3m";

let selectedIndex = "NIFTY 50";

let chart = null;

let candleSeries = null;

let ema9Series = null;

let ema15Series = null;

let vwapSeries = null;


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

            height: 430,

            layout: {

                background: {
                    type: "solid",
                    color: "#111923"
                },

                textColor: "#c9d1d9"

            },

            grid: {

                vertLines: {
                    color: "#1f2a36"
                },

                horzLines: {
                    color: "#1f2a36"
                }

            },

            rightPriceScale: {

                borderColor:
                "#263241"

            },

            timeScale: {

                borderColor:
                "#263241",

                timeVisible: true

            }

        }
    );


    candleSeries =
    chart.addCandlestickSeries({

        upColor: "#36d98c",

        downColor: "#ff6b6b",

        borderVisible: false,

        wickUpColor: "#36d98c",

        wickDownColor: "#ff6b6b"

    });


    ema9Series =
    chart.addLineSeries({

        color: "#f4c542",

        lineWidth: 2,

        title: "EMA 9"

    });


    ema15Series =
    chart.addLineSeries({

        color: "#46bdf0",

        lineWidth: 2,

        title: "EMA 15"

    });


    vwapSeries =
    chart.addLineSeries({

        color: "#bd7df5",

        lineWidth: 2,

        title: "VWAP"

    });

}


window.addEventListener(
    "resize",
    function() {

        if (chart) {

            const container =
            document.getElementById(
                "chart"
            );

            chart.applyOptions({

                width:
                container.clientWidth

            });

        }

    }
);


async function loadData() {

    try {

        const response =
        await fetch(

            "/api/data?index=" +

            encodeURIComponent(
                selectedIndex
            )

            +

            "&tf=" +

            encodeURIComponent(
                selectedTF
            )

        );


        const data =
        await response.json();


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

    catch(error) {

        console.log(
            error
        );

        document.getElementById(
            "scanner"
        ).innerHTML =
        "Error loading data";

    }

}


function updateScanner(data) {

    const signalElement =
    document.getElementById(
        "signal"
    );


    signalElement.innerText =
    data.signal;


    signalElement.className =
    "signal";


    if (
        data.signal === "CALL"
    ) {

        signalElement.classList.add(
            "call"
        );

    }

    else if (
        data.signal === "PUT"
    ) {

        signalElement.classList.add(
            "put"
        );

    }

    else {

        signalElement.classList.add(
            "wait"
        );

    }


    document.getElementById(
        "scanner"
    ).innerHTML =

        '<div class="data-line">' +

        '<b>Index:</b> ' +

        selectedIndex +

        '</div>' +

        '<div class="data-line">' +

        '<b>Timeframe:</b> ' +

        selectedTF +

        '</div>' +

        '<div class="data-line">' +

        '<b>Price:</b> ' +

        data.price +

        '</div>' +

        '<div class="data-line">' +

        '<b>EMA 9:</b> ' +

        data.ema9 +

        '</div>' +

        '<div class="data-line">' +

        '<b>EMA 15:</b> ' +

        data.ema15 +

        '</div>' +

        '<div class="data-line">' +

        '<b>VWAP:</b> ' +

        data.vwap +

        '</div>' +

        '<div class="small">' +

        data.time +

        '</div>';

}


function updateChart(data) {

    if (!chart) {

        createChart();

    }


    const candles =

    data.map(function(item) {

        return {

            time: item.time,

            open: item.open,

            high: item.high,

            low: item.low,

            close: item.close

        };

    });


    const ema9 =

    data.map(function(item) {

        return {

            time: item.time,

            value: item.ema9

        };

    });


    const ema15 =

    data.map(function(item) {

        return {

            time: item.time,

            value: item.ema15

        };

    });


    const vwap =

    data.map(function(item) {

        return {

            time: item.time,

            value: item.vwap

        };

    });


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


    data.forEach(
        function(item) {

            if (
                item.marker === "CALL"
            ) {

                markers.push({

                    time: item.time,

                    position:
                    "belowBar",

                    color:
                    "#36d98c",

                    shape:
                    "arrowUp",

                    text:
                    "CALL"

                });

            }


            if (
                item.marker === "PUT"
            ) {

                markers.push({

                    time: item.time,

                    position:
                    "aboveBar",

                    color:
                    "#ff6b6b",

                    shape:
                    "arrowDown",

                    text:
                    "PUT"

                });

            }

        }
    );


    try {

        candleSeries.setMarkers(
            markers
        );

    }

    catch(error) {

        console.log(
            "Marker error:",
            error
        );

    }


    chart.timeScale().fitContent();

}


function updateBacktest(data) {

    document.getElementById(
        "totalTrades"
    ).innerText =
    data.total_trades;


    document.getElementById(
        "wins"
    ).innerText =
    data.wins;


    document.getElementById(
        "losses"
    ).innerText =
    data.losses;


    document.getElementById(
        "winRate"
    ).innerText =

    data.win_rate + "%";


    document.getElementById(
        "netPoints"
    ).innerText =
    data.net_points;


    const tradesElement =
    document.getElementById(
        "trades"
    );


    if (

        !data.trades

        ||

        data.trades.length === 0

    ) {

        tradesElement.innerHTML =
        "No trades found";

        return;

    }


    let html = "";


    data.trades
    .slice(-30)
    .reverse()
    .forEach(
        function(trade) {

            const resultClass =

            trade.result === "WIN"

            ?

            "win"

            :

            "loss";


            html +=

            '<div class="trade">' +

            '<b>' +

            trade.type +

            '</b>' +

            ' &nbsp; Entry: ' +

            trade.entry +

            ' &nbsp; Exit: ' +

            trade.exit +

            ' &nbsp; Points: ' +

            '<span class="' +

            resultClass +

            '">' +

            trade.points +

            '</span>' +

            ' &nbsp; ' +

            '<span class="' +

            resultClass +

            '">' +

            trade.result +

            '</span>' +

            '<br>' +

            '<span class="small">' +

            trade.reason +

            '</span>' +

            '</div>';

        }
    );


    tradesElement.innerHTML =
    html;

}


document
.querySelectorAll(
    ".tf-btn"
)
.forEach(
    function(button) {

        button.addEventListener(
            "click",
            function() {

                document
                .querySelectorAll(
                    ".tf-btn"
                )
                .forEach(
                    function(btn) {

                        btn.classList.remove(
                            "active"
                        );

                    }
                );


                button.classList.add(
                    "active"
                );


                selectedTF =
                button.dataset.tf;


                loadData();

            }
        );

    }
);


document
.querySelectorAll(
    ".index-btn"
)
.forEach(
    function(button) {

        button.addEventListener(
            "click",
            function() {

                document
                .querySelectorAll(
                    ".index-btn"
                )
                .forEach(
                    function(btn) {

                        btn.classList.remove(
                            "active"
                        );

                    }
                );


                button.classList.add(
                    "active"
                );


                selectedIndex =
                button.dataset.index;


                loadData();

            }
        );

    }
);


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
# API DATA
# =========================================================

@app.route("/api/data")
def api_data():

    index_name = request.args.get(
        "index",
        "NIFTY 50"
    )

    tf = request.args.get(
        "tf",
        "3m"
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


    scanner =
    calculate_scanner(
        data
    )


    chart =
    chart_json(
        data
    )


    backtest =
    run_backtest(
        data
    )


    return jsonify({

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
        "ok",

        "scanner":
        "running",

        "chart":
        "running",

        "backtest":
        "running"

    })


# =========================================================
# RUN
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

        debug=False

    )
