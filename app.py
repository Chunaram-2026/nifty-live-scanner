from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# =========================================================
# INDEX SYMBOLS
# =========================================================

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN"
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
# TIMEFRAME SETTINGS
# =========================================================

def timeframe_settings(tf):

    # Yahoo Finance limitations को ध्यान में रखकर
    if tf == "1m":
        return "1m", "7d", None

    if tf == "2m":
        return "2m", "60d", None

    if tf == "3m":
        # Yahoo में 3m direct नहीं है
        # 1m data से 3m बनाया जाएगा
        return "1m", "7d", "3min"

    if tf == "5m":
        return "5m", "60d", None

    if tf == "15m":
        return "15m", "60d", None

    if tf == "1h":
        return "1h", "730d", None

    if tf == "2h":
        # 1h से 2h बनाया जाएगा
        return "1h", "730d", "2h"

    if tf == "1d":
        return "1d", "5y", None

    if tf == "1wk":
        return "1wk", "10y", None

    return "5m", "60d", None


# =========================================================
# CLEAN YFINANCE DATA
# =========================================================

def clean_columns(data):

    if data is None:
        return None

    if data.empty:
        return None

    # yfinance MultiIndex handling
    if isinstance(data.columns, pd.MultiIndex):

        # पहले level में Open, High, Low, Close...
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:

        if col not in data.columns:
            return None

    if "Volume" not in data.columns:
        data["Volume"] = 0

    data = data.dropna(subset=required)

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
            threads=False
        )

        data = clean_columns(data)

        if data is None:
            return None

        # ---------------------------------------------
        # 3 MINUTE / 2 HOUR RESAMPLING
        # ---------------------------------------------

        if resample_rule:

            # timezone aware index को रहने दें
            data = data.resample(resample_rule).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            })

            data = data.dropna(subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ])

        return data

    except Exception as e:

        print("DOWNLOAD ERROR:", e)

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
    # TYPICAL PRICE
    # =====================================================

    typical_price = (
        high + low + close
    ) / 3

    # =====================================================
    # VWAP
    #
    # VWAP हर trading day पर reset होगा
    # =====================================================

    data["_date"] = data.index.date

    cumulative_pv = (
        typical_price * volume
    ).groupby(data["_date"]).cumsum()

    cumulative_volume = (
        volume.groupby(data["_date"]).cumsum()
    )

    data["VWAP"] = np.where(
        cumulative_volume > 0,
        cumulative_pv / cumulative_volume,
        typical_price
    )

    data.drop(
        columns=["_date"],
        inplace=True
    )

    return data


# =========================================================
# SIGNAL
# =========================================================

def get_signal(row):

    price = row["Close"]
    ema9 = row["EMA9"]
    ema15 = row["EMA15"]
    vwap = row["VWAP"]

    if pd.isna(price):
        return "NO DATA"

    if pd.isna(ema9) or pd.isna(ema15):
        return "WAIT"

    if pd.isna(vwap):
        return "WAIT"

    # CALL BIAS
    if (
        price > vwap
        and ema9 > ema15
    ):
        return "CALL"

    # PUT BIAS
    if (
        price < vwap
        and ema9 < ema15
    ):
        return "PUT"

    return "WAIT"


# =========================================================
# SIGNAL MARKERS
#
# केवल signal बदलने पर marker लगाया जाएगा
# =========================================================

def add_signal_markers(data):

    data = data.copy()

    signals = []

    previous = "WAIT"

    for _, row in data.iterrows():

        current = get_signal(row)

        marker = ""

        if current == "CALL" and previous != "CALL":
            marker = "CALL"

        elif current == "PUT" and previous != "PUT":
            marker = "PUT"

        signals.append(marker)

        if current in ["CALL", "PUT"]:
            previous = current

        elif current == "WAIT":
            previous = "WAIT"

    data["MARKER"] = signals

    return data


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(data):

    if data is None or len(data) < 20:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    data = calculate_indicators(data)

    if data is None or data.empty:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    row = data.iloc[-1]

    signal = get_signal(row)

    return {
        "signal": (
            "CALL BIAS"
            if signal == "CALL"
            else
            "PUT BIAS"
            if signal == "PUT"
            else
            "WAIT"
        ),

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
        )
    }


# =========================================================
# CHART DATA
# =========================================================

def chart_json(data):

    if data is None or data.empty:
        return []

    result = []

    for timestamp, row in data.iterrows():

        try:

            # Unix timestamp
            ts = int(
                pd.Timestamp(timestamp).timestamp()
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

                "signal": get_signal(row),

                "marker": row.get(
                    "MARKER",
                    ""
                )
            })

        except Exception:
            continue

    return result


# =========================================================
# BACKTEST
#
# Logic:
#
# CALL:
# Price > VWAP
# EMA9 > EMA15
#
# PUT:
# Price < VWAP
# EMA9 < EMA15
#
# Entry अगली candle के Open पर
# Exit opposite signal आने पर
# =========================================================

def run_backtest(data):

    if data is None or data.empty:
        return {
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0
        }

    data = calculate_indicators(data)

    if data is None or len(data) < 20:

        return {
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0
        }

    signals = []

    for _, row in data.iterrows():
        signals.append(
            get_signal(row)
        )

    data = data.copy()

    data["Signal"] = signals

    trades = []

    position = None

    entry_price = None

    entry_time = None

    position_type = None

    for i in range(1, len(data)):

        current_signal = data["Signal"].iloc[i]

        current_open = float(
            data["Open"].iloc[i]
        )

        current_close = float(
            data["Close"].iloc[i]
        )

        current_time = data.index[i]

        # =============================================
        # ENTRY
        # =============================================

        if position is None:

            if current_signal == "CALL":

                position = "CALL"

                position_type = "CALL"

                entry_price = current_open

                entry_time = current_time

                continue

            if current_signal == "PUT":

                position = "PUT"

                position_type = "PUT"

                entry_price = current_open

                entry_time = current_time

                continue

        # =============================================
        # CALL EXIT
        # =============================================

        if position == "CALL":

            if current_signal == "PUT":

                exit_price = current_open

                points = (
                    exit_price - entry_price
                )

                trades.append({
                    "type": "CALL",
                    "entry": round(
                        entry_price,
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
                    "result": (
                        "WIN"
                        if points > 0
                        else "LOSS"
                    ),
                    "entry_time": str(
                        entry_time
                    ),
                    "exit_time": str(
                        current_time
                    )
                })

                position = "PUT"

                position_type = "PUT"

                entry_price = current_open

                entry_time = current_time

        # =============================================
        # PUT EXIT
        # =============================================

        elif position == "PUT":

            if current_signal == "CALL":

                exit_price = current_open

                points = (
                    entry_price - exit_price
                )

                trades.append({
                    "type": "PUT",
                    "entry": round(
                        entry_price,
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
                    "result": (
                        "WIN"
                        if points > 0
                        else "LOSS"
                    ),
                    "entry_time": str(
                        entry_time
                    ),
                    "exit_time": str(
                        current_time
                    )
                })

                position = "CALL"

                position_type = "CALL"

                entry_price = current_open

                entry_time = current_time

    # =============================================
    # LAST OPEN POSITION CLOSE
    # =============================================

    if position is not None and entry_price is not None:

        last_price = float(
            data["Close"].iloc[-1]
        )

        last_time = data.index[-1]

        if position == "CALL":

            points = (
                last_price - entry_price
            )

        else:

            points = (
                entry_price - last_price
            )

        trades.append({
            "type": position_type,
            "entry": round(
                entry_price,
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
            "result": (
                "WIN"
                if points > 0
                else "LOSS"
            ),
            "entry_time": str(
                entry_time
            ),
            "exit_time": str(
                last_time
            )
        })

    total_trades = len(trades)

    wins = sum(
        1
        for x in trades
        if x["result"] == "WIN"
    )

    losses = sum(
        1
        for x in trades
        if x["result"] == "LOSS"
    )

    net_points = sum(
        x["points"]
        for x in trades
    )

    if total_trades > 0:

        win_rate = (
            wins / total_trades
        ) * 100

    else:

        win_rate = 0

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
        )
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    html = r"""
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>Personal Scalping Scanner</title>

<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<style>

body {

    background:#080c12;

    color:white;

    font-family:Arial,sans-serif;

    padding:12px;

    margin:0;
}

h1 {

    font-size:22px;

    margin:10px 0 15px;
}

h2 {

    font-size:19px;

    margin-top:25px;
}

.card {

    background:#111923;

    border:1px solid #263241;

    border-radius:12px;

    padding:15px;

    margin-bottom:12px;
}

.tf {

    display:flex;

    gap:6px;

    overflow:auto;

    margin-bottom:12px;

    padding-bottom:5px;
}

button {

    padding:9px 12px;

    border-radius:8px;

    border:1px solid #39485b;

    background:#17212d;

    color:white;

    white-space:nowrap;
}

button.active {

    background:#6d28d9;

    border-color:#8b5cf6;
}

.value {

    margin:6px 0;
}

.wait {

    color:#ffd45a;
}

.call {

    color:#54dc9a;
}

.put {

    color:#ff6673;
}

#chart {

    width:100%;

    height:500px;

    border-radius:10px;

    overflow:hidden;

    background:#0d141d;
}

.stats {

    display:grid;

    grid-template-columns:
        repeat(2,1fr);

    gap:8px;
}

.stat {

    background:#17212d;

    padding:12px;

    border-radius:8px;
}

.stat-title {

    font-size:12px;

    color:#9ca3af;
}

.stat-value {

    font-size:18px;

    margin-top:5px;
}

.trade {

    border-bottom:
        1px solid #263241;

    padding:8px 0;

    font-size:13px;
}

</style>

</head>

<body>

<h1>⚡ Personal Scalping Scanner</h1>


<!-- =====================================================
     SCANNER TIMEFRAME
     ===================================================== -->

<div class="tf">

<button onclick="loadScanner('1m')">1M</button>

<button onclick="loadScanner('2m')">2M</button>

<button onclick="loadScanner('3m')">3M</button>

<button onclick="loadScanner('5m')">5M</button>

<button onclick="loadScanner('15m')">15M</button>

<button onclick="loadScanner('1h')">1H</button>

<button onclick="loadScanner('2h')">2H</button>

<button onclick="loadScanner('1d')">1D</button>

<button onclick="loadScanner('1wk')">1W</button>

</div>


<div id="result">

Loading...

</div>


<!-- =====================================================
     CHART
     ===================================================== -->

<h2>📊 Index Chart</h2>

<div class="tf">

<button
id="btnNifty"
onclick="loadChart('^NSEI','NIFTY 50')">

NIFTY 50

</button>

<button
id="btnBank"
onclick="loadChart('^NSEBANK','BANK NIFTY')">

BANK NIFTY

</button>

<button
id="btnSensex"
onclick="loadChart('^BSESN','SENSEX')">

SENSEX

</button>

</div>


<div class="tf">

<button onclick="changeChartTF('1m')">
1M
</button>

<button onclick="changeChartTF('2m')">
2M
</button>

<button onclick="changeChartTF('3m')">
3M
</button>

<button onclick="changeChartTF('5m')">
5M
</button>

<button onclick="changeChartTF('15m')">
15M
</button>

<button onclick="changeChartTF('1h')">
1H
</button>

<button onclick="changeChartTF('2h')">
2H
</button>

<button onclick="changeChartTF('1d')">
1D
</button>

<button onclick="changeChartTF('1wk')">
1W
</button>

</div>


<div id="chart"></div>


<!-- =====================================================
     BACKTEST
     ===================================================== -->

<h2>🧪 Backtest</h2>

<div class="card">

<button onclick="runBacktest()">

Run Backtest

</button>

<div
id="backtestResult"
style="margin-top:15px;">

Backtest नहीं चला है।

</div>

</div>


<script>


// =======================================================
// GLOBAL VARIABLES
// =======================================================

let currentSymbol = "^NSEI";

let currentIndex = "NIFTY 50";

let currentTF = "5m";

let chart = null;

let candleSeries = null;

let ema9Series = null;

let ema15Series = null;

let vwapSeries = null;


// =======================================================
// SCANNER
// =======================================================

function loadScanner(tf) {

    document.getElementById("result").innerHTML =
        "Loading " + tf + "...";


    fetch(
        "/api/scan?tf=" +
        encodeURICom
