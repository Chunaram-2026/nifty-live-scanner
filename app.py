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
    "1wk",
]


# Maximum trades per day
MAX_TRADES_PER_DAY = 3


# =========================================================
# EMA TREND SETTINGS
# =========================================================

EMA_SLOPE_LOOKBACK = 5

# पहले 0.20 था, जो काफी strict था
# इसे practical रखा गया है
MIN_EMA_SLOPE = 0.02


# =========================================================
# CANDLE QUALITY
# =========================================================

# Candle body कम से कम range का 35%
MIN_BODY_PERCENT = 0.35


# Rejection wick ratio
WICK_RATIO = 1.5


# Stop-loss buffer
SL_BUFFER_POINTS = 0.0


# Risk Reward
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

    # Yahoo Finance में direct 3m reliable नहीं है
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


    # Fix MultiIndex columns
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


    data = data.dropna(
        subset=required
    )


    for col in required + ["Volume"]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )


    data = data.dropna(
        subset=required
    )


    data = data.sort_index()


    return data


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data(symbol, tf):

    interval, period, resample_rule = timeframe_settings(tf)


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

        print("DOWNLOAD ERROR:", e)

        traceback.print_exc()

        return None


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(data):

    if data is None or data.empty:

        return None


    data = data.copy()


    open_price = pd.to_numeric(
        data["Open"],
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


    close = pd.to_numeric(
        data["Close"],
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
    # SOURCE:
    #
    # (OPEN + HIGH + LOW + CLOSE) / 4
    # =====================================================

    vwap_price = (

        open_price +

        high +

        low +

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

        vwap_price *

        volume

    ).groupby(
        dates
    ).cumsum()


    cumulative_volume = (

        volume

    ).groupby(
        dates
    ).cumsum()


    # If volume is unavailable,
    # fallback to OHLC4 price
    data["VWAP"] = np.where(

        cumulative_volume > 0,

        cumulative_pv /
        cumulative_volume,

        vwap_price

    )


    # =====================================================
    # EMA SLOPE
    # =====================================================

    data["EMA9_SLOPE"] = (

        data["EMA9"] -

        data["EMA9"].shift(
            EMA_SLOPE_LOOKBACK
        )

    )


    data["EMA15_SLOPE"] = (

        data["EMA15"] -

        data["EMA15"].shift(
            EMA_SLOPE_LOOKBACK
        )

    )


    return data


# =========================================================
# CANDLE INFORMATION
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

    body = abs(c - o)


    if candle_range <= 0:

        return {
            "valid": False
        }


    body_ratio = body / candle_range


    upper_wick = (

        h -

        max(o, c)

    )


    lower_wick = (

        min(o, c) -

        l

    )


    bullish = c > o

    bearish = c < o


    # =====================================================
    # STRONG BODY
    # =====================================================

    strong_body = (

        body_ratio >=
        MIN_BODY_PERCENT

    )


    # =====================================================
    # BULLISH REJECTION / HAMMER
    # =====================================================

    bullish_hammer = (

        bullish

        and

        lower_wick >=
        body * WICK_RATIO

        and

        upper_wick <=
        candle_range * 0.40

    )


    # =====================================================
    # BEARISH REJECTION
    # =====================================================

    bearish_hammer = (

        bearish

        and

        upper_wick >=
        body * WICK_RATIO

        and

        lower_wick <=
        candle_range * 0.40

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


    # Normalize slope by price
    ema9_slope_pct = (

        ema9_slope /

        close

    ) * 100


    ema15_slope_pct = (

        ema15_slope /

        close

    ) * 100


    bullish = (

        ema9 > ema15

        and

        ema9_slope_pct >=
        MIN_EMA_SLOPE

        and

        ema15_slope_pct > 0

    )


    bearish = (

        ema9 < ema15

        and

        ema9_slope_pct <=
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


    values = [

        price,

        ema9,

        ema15,

        vwap

    ]


    if any(
        pd.isna(x)
        for x in values
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

    )


    if (

        bearish_structure

        and

        candle["good_bearish"]

    ):

        return "PUT"


    return "WAIT"


# =========================================================
# ADD SIGNAL MARKERS
#
# लगातार candles पर same marker नहीं
# =========================================================

def add_signal_markers(data):

    if data is None or data.empty:

        return data


    data = data.copy()


    markers = []


    previous_signal = "WAIT"


    for _, row in data.iterrows():

        signal = get_signal(
            row
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


        markers.append(
            marker
        )


        previous_signal = signal


    data["MARKER"] = markers


    return data


# =========================================================
# CALCULATE ALL
# =========================================================

def calculate_all_signals(data):

    if data is None or data.empty:

        return None


    data = calculate_indicators(
        data
    )


    if data is None or data.empty:

        return None


    data = add_signal_markers(
        data
    )


    return data


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(data):

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


    if data is None or len(data) < 20:

        return empty_result


    data = calculate_all_signals(
        data
    )


    if data is None or data.empty:

        return empty_result


    # Latest candle
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


    # =====================================================
    # CALL
    # =====================================================

    if signal == "CALL":

        stop_loss = (

            low -

            SL_BUFFER_POINTS

        )


        risk = (

            price -

            stop_loss

        )


        if risk > 0:

            target_1 = (

                price +

                risk * RR_1

            )


            target_2 = (

                price +

                risk * RR_2

            )


    # =====================================================
    # PUT
    # =====================================================

    elif signal == "PUT":

        stop_loss = (

            high +

            SL_BUFFER_POINTS

        )


        risk = (

            stop_loss -

            price

        )


        if risk > 0:

            target_1 = (

                price -

                risk * RR_1

            )


            target_2 = (

                price -

                risk * RR_2

            )


    return {

        "signal":
        signal,

        "price":
        round(price, 2),

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

        round(stop_loss, 2)

        if stop_loss is not None

        else None,


        "target_1":

        round(target_1, 2)

        if target_1 is not None

        else None,


        "target_2":

        round(target_2, 2)

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

def chart_json(data):

    if data is None or data.empty:

        return []


    data = calculate_all_signals(
        data
    )


    if data is None or data.empty:

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
                "CHART ROW ERROR:",
                e
            )


    return result


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(data):

    empty = {

        "trades": [],

        "total_trades": 0,

        "wins": 0,

        "losses": 0,

        "win_rate": 0,

        "net_points": 0,

        "target_1_hits": 0,

        "target_2_hits": 0

    }


    if data is None or len(data) < 30:

        return empty


    data = calculate_all_signals(
        data
    )


    if data is None or len(data) < 30:

        return empty


    trades = []

    open_trade = None

    daily_trade_count = {}


    # =====================================================
    # START BACKTEST
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
        # FIRST:
        # CHECK EXISTING TRADE
        # =================================================

        if open_trade is not None:


            trade_type = open_trade["type"]

            entry = open_trade["entry"]

            sl = open_trade["stop_loss"]

            target1 = open_trade["target_1"]

            target2 = open_trade["target_2"]


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
                # SL priority if same candle hits both

                if low <= sl:

                    exit_price = sl

                    exit_reason = "STOP LOSS"


                elif high >= target2:

                    exit_price = target2

                    exit_reason = "TARGET 1:3"


                elif high >= target1:

                    exit_price = target1

                    exit_reason = "TARGET 1:2"


            # =============================================
            # PUT EXIT
            # =============================================

            elif trade_type == "PUT":


                if high >= sl:

                    exit_price = sl

                    exit_reason = "STOP LOSS"


                elif low <= target2:

                    exit_price = target2

                    exit_reason = "TARGET 1:3"


                elif low <= target1:

                    exit_price = target1

                    exit_reason = "TARGET 1:2"


            # =============================================
            # CLOSE TRADE
            # =============================================

            if exit_price is not None:


                if trade_type == "CALL":

                    points = (

                        exit_price -

                        entry

                    )


                else:

                    points = (

                        entry -

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
                    str(timestamp)

                })


                open_trade = None


        # =================================================
        # IF TRADE STILL OPEN
        # DON'T TAKE NEW TRADE
        # =================================================

        if open_trade is not None:

            continue


        # =================================================
        # MAX TRADES PER DAY
        # =================================================

        today_count = daily_trade_count.get(

            date_key,

            0

        )


        if today_count >= MAX_TRADES_PER_DAY:

            continue


        # =================================================
        # GET SIGNAL
        # =================================================

        signal = get_signal(
            row
        )


        if signal == "WAIT":

            continue


        entry = float(
            row["Close"]
        )


        low = float(
            row["Low"]
        )


        high = float(
            row["High"]
        )


        # =================================================
        # CALL ENTRY
        # =================================================

        if signal == "CALL":


            stop_loss = (

                low -

                SL_BUFFER_POINTS

            )


            risk = (

                entry -

                stop_loss

            )


            if risk <= 0:

                continue


            target_1 = (

                entry +

                risk * RR_1

            )


            target_2 = (

                entry +

                risk * RR_2

            )


            open_trade = {

                "type":
                "CALL",


                "entry":
                entry,


                "stop_loss":
                stop_loss,


                "target_1":
                target_1,


                "target_2":
                target_2,


                "entry_time":
                str(timestamp)

            }


            daily_trade_count[
                date_key
            ] = today_count + 1


        # =================================================
        # PUT ENTRY
        # =================================================

        elif signal == "PUT":


            stop_loss = (

                high +

                SL_BUFFER_POINTS

            )


            risk = (

                stop_loss -

                entry

            )


            if risk <= 0:

                continue


            target_1 = (

                entry -

                risk * RR_1

            )


            target_2 = (

                entry -

                risk * RR_2

            )


            open_trade = {

                "type":
                "PUT",


                "entry":
                entry,


                "stop_loss":
                stop_loss,


                "target_1":
                target_1,


                "target_2":
                target_2,


                "entry_time":
                str(timestamp)

            }


            daily_trade_count[
                date_key
            ] = today_count + 1


    # =====================================================
    # CLOSE LAST OPEN TRADE
    # =====================================================

    if open_trade is not None:


        last_row = data.iloc[-1]


        last_price = float(
            last_row["Close"]
        )


        if open_trade["type"] == "CALL":

            points = (

                last_price -

                open_trade["entry"]

            )


        else:

            points = (

                open_trade["entry"] -

                last_price

            )


        trades.append({

            "type":
            open_trade["type"],


            "entry":
            round(
                open_trade["entry"],
                2
            ),


            "exit":
            round(
                last_price,
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

            "WIN"

            if points > 0

            else "LOSS",


            "exit_reason":
            "END OF DATA",


            "entry_time":
            open_trade["entry_time"],


            "exit_time":
            str(
                data.index[-1]
            )

        })


    # =====================================================
    # BACKTEST STATISTICS
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

        if trade["exit_reason"] == "TARGET 1:2"

    )


    target_2_hits = sum(

        1

        for trade in trades

        if trade["exit_reason"] == "TARGET 1:3"

    )


    net_points = sum(

        trade["points"]

        for trade in trades

    )


    win_rate = 0


    if total_trades > 0:

        win_rate = (

            wins /

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


<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js">
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

    background: #2463eb;

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

    color: #aab7c4;

}


.value {

    font-size: 18px;

    margin-top: 5px;

    font-weight: bold;

}


.call {

    color: #4ade80;

}


.put {

    color: #fb7185;

}


.wait {

    color: #facc15;

}


.good {

    color: #4ade80;

}


.bad {

    color: #fb7185;

}


.small {

    font-size: 12px;

    color: #aab7c4;

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
class="value wait">

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
VWAP
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

<div id="chart"></div>

</div>


<h2>
📈 Backtest
</h2>


<div class="card">


<div class="grid">


<div class="box">

<div class="label">
Total Trades
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
Recent Trades
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
            500,


            layout: {

                background: {
                    type: "solid",
                    color: "#111923"
                },

                textColor:
                "#d1d4dc"

            },


            grid: {

                vertLines: {
                    color: "#202b38"
                },

                horzLines: {
                    color: "#202b38"
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


function formatNumber(value) {


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


        const signalElement =
        document.getElementById(
            "signal"
        );


        signalElement.textContent =
        "Loading...";


        signalElement.className =
        "value wait";


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


        if (
            data.error
        ) {

            signalElement.textContent =
            "ERROR";

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

    catch (error) {


        console.error(
            error
        );


        const signalElement =
        document.getElementById(
            "signal"
        );


        signalElement.textContent =
        "ERROR";


        signalElement.className =
        "value put";


    }


}


function updateScanner(scanner) {


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
    "value " +

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


function updateChart(chartData) {


    if (
        !chart
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


    candleSeries.setMarkers(
        markers
    );


    chart.timeScale().fitContent();


}


function updateBacktest(backtest) {


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
    backtest.win_rate +
    "%";


    const netElement =
    document.getElementById(
        "netPoints"
    );


    netElement.textContent =
    backtest.net_points;


    netElement.className =
    "value " +

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
        "<div class='small'>No trades found.</div>";


        return;

    }


    trades.forEach(
        trade => {


            const div =
            document.createElement(
                "div"
            );


            div.style.padding =
            "10px 0";


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


    if data is None or data.empty:


        return jsonify({

            "error":
            "Market data not available"

        }), 500


    scanner = calculate_scanner(
        data
    )


    chart = chart_json(
        data
    )


    backtest = run_backtest(
        data
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
# HEALTH CHECK
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
