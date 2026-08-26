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
# CLEAN YFINANCE DATA
# =========================================================

def clean_columns(data):

    if data is None:
        return None

    if data.empty:
        return None

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

        # =================================================
        # RESAMPLE 3 MIN / 2 HOUR
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
        high +
        low +
        close
    ) / 3

    # =====================================================
    # VWAP
    # Reset every trading day
    # =====================================================

    data["_date"] = data.index.date

    cumulative_pv = (
        typical_price * volume
    ).groupby(
        data["_date"]
    ).cumsum()

    cumulative_volume = (
        volume
        .groupby(data["_date"])
        .cumsum()
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

    return data


# =========================================================
# SIGNAL
#
# CALL:
# Price > VWAP
# EMA9 > EMA15
# Current LOW sweeps previous LOW
# Current CLOSE returns above previous LOW
#
# PUT:
# Price < VWAP
# EMA9 < EMA15
# Current HIGH sweeps previous HIGH
# Current CLOSE returns below previous HIGH
# =========================================================

def get_signal(
    row,
    previous=None
):

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

    if pd.isna(price):
        return "NO DATA"

    if (
        pd.isna(ema9)
        or pd.isna(ema15)
    ):
        return "WAIT"

    if pd.isna(vwap):
        return "WAIT"

    if previous is None:
        return "WAIT"

    try:

        prev_high = float(
            previous["High"]
        )

        prev_low = float(
            previous["Low"]
        )

    except Exception:

        return "WAIT"

    if (
        pd.isna(prev_high)
        or pd.isna(prev_low)
    ):
        return "WAIT"

    # =====================================================
    # BUY / CALL LIQUIDITY SWEEP
    # =====================================================

    buy_sweep = (
        float(row["Low"]) < prev_low
        and price > prev_low
    )

    if (
        price > vwap
        and ema9 > ema15
        and buy_sweep
    ):
        return "CALL"

    # =====================================================
    # SELL / PUT LIQUIDITY SWEEP
    # =====================================================

    sell_sweep = (
        float(row["High"]) > prev_high
        and price < prev_high
    )

    if (
        price < vwap
        and ema9 < ema15
        and sell_sweep
    ):
        return "PUT"

    return "WAIT"


# =========================================================
# ADD SIGNAL MARKERS
# =========================================================

def add_signal_markers(data):

    if data is None or data.empty:
        return data

    data = data.copy()

    signals = []

    previous_signal = "WAIT"

    previous_row = None

    for _, row in data.iterrows():

        current = get_signal(
            row,
            previous_row
        )

        marker = ""

        # New CALL only
        if (
            current == "CALL"
            and previous_signal != "CALL"
        ):
            marker = "CALL"

        # New PUT only
        elif (
            current == "PUT"
            and previous_signal != "PUT"
        ):
            marker = "PUT"

        signals.append(marker)

        # Update signal state
        if current in [
            "CALL",
            "PUT"
        ]:

            previous_signal = current

        elif current == "WAIT":

            previous_signal = "WAIT"

        previous_row = row

    data["MARKER"] = signals

    return data


# =========================================================
# CALCULATE ALL SIGNALS
# =========================================================

def calculate_all_signals(data):

    if data is None or data.empty:
        return None

    data = calculate_indicators(
        data
    )

    if data is None or data.empty:
        return None

    return add_signal_markers(
        data
    )


# =========================================================
# SCANNER
# =========================================================

def calculate_scanner(data):

    if (
        data is None
        or len(data) < 20
    ):

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    data = calculate_all_signals(
        data
    )

    if (
        data is None
        or data.empty
    ):

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    row = data.iloc[-1]

    previous = (
        data.iloc[-2]
        if len(data) >= 2
        else None
    )

    signal = get_signal(
        row,
        previous
    )

    if signal == "CALL":
        bias = "CALL"

    elif signal == "PUT":
        bias = "PUT"

    else:
        bias = "WAIT"

    return {

        "signal": bias,

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

        "time": str(
            data.index[-1]
        )
    }


# =========================================================
# CHART JSON
# =========================================================

def chart_json(data):

    if (
        data is None
        or data.empty
    ):
        return []

    data = calculate_all_signals(
        data
    )

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

            previous = (
                data.iloc[i - 1]
                if i > 0
                else None
            )

            signal = get_signal(
                row,
                previous
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
                "CHART ROW ERROR:",
                e
            )

            continue

    return result


# =========================================================
# BACKTEST
#
# Signal candle पर signal बनता है.
# Entry अगली candle के OPEN पर.
# Exit opposite signal की अगली candle OPEN पर.
# =========================================================

def run_backtest(data):

    if (
        data is None
        or data.empty
    ):

        return {
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0
        }

    data = calculate_all_signals(
        data
    )

    if (
        data is None
        or len(data) < 20
    ):

        return {
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0
        }

    signals = []

    previous = None

    for _, row in data.iterrows():

        signal = get_signal(
            row,
            previous
        )

        signals.append(
            signal
        )

        previous = row

    data = data.copy()

    data["Signal"] = signals

    trades = []

    position = None

    entry_price = None

    entry_time = None

    position_type = None

    # =====================================================
    # SIGNAL CANDLE -> NEXT CANDLE OPEN ENTRY
    # =====================================================

    for i in range(
        0,
        len(data) - 1
    ):

        signal = data[
            "Signal"
        ].iloc[i]

        next_open = float(
            data["Open"].iloc[i + 1]
        )

        next_time = data.index[i + 1]

        # =================================================
        # NO POSITION
        # =================================================

        if position is None:

            if signal == "CALL":

                position = "CALL"

                position_type = "CALL"

                entry_price = next_open

                entry_time = next_time

                continue

            if signal == "PUT":

                position = "PUT"

                position_type = "PUT"

                entry_price = next_open

                entry_time = next_time

                continue

        # =================================================
        # CALL -> PUT
        # =================================================

        if position == "CALL":

            if signal == "PUT":

                exit_price = next_open

                points = (
                    exit_price -
                    entry_price
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
                        next_time
                    )
                })

                # Reverse position
                position = "PUT"

                position_type = "PUT"

                entry_price = next_open

                entry_time = next_time

        # =================================================
        # PUT -> CALL
        # =================================================

        elif position == "PUT":

            if signal == "CALL":

                exit_price = next_open

                points = (
                    entry_price -
                    exit_price
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
                        next_time
                    )
                })

                # Reverse position
                position = "CALL"

                position_type = "CALL"

                entry_price = next_open

                entry_time = next_time

    # =====================================================
    # CLOSE LAST OPEN POSITION
    # =====================================================

    if (
        position is not None
        and entry_price is not None
    ):

        last_price = float(
            data["Close"].iloc[-1]
        )

        last_time = data.index[-1]

        if position == "CALL":

            points = (
                last_price -
                entry_price
            )

        else:

            points = (
                entry_price -
                last_price
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

    # =====================================================
    # STATISTICS
    # =====================================================

    total_trades = len(
        trades
    )

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

    html = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>
Personal Scalping Scanner
</title>

<script src=
"https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js">
</script>

<style>

* {
    box-sizing:border-box;
}

body {

    background:#080c12;

    color:#ffffff;

    font-family:
    Arial,
    sans-serif;

    margin:0;

    padding:12px;
}

h1 {

    font-size:22px;

    margin:
    8px 0 15px;
}

h2 {

    font-size:18px;

    margin:
    20px 0 10px;
}

.card {

    background:#111923;

    border:
    1px solid #263241;

    border-radius:12px;

    padding:14px;

    margin-bottom:12px;
}

.tf {

    display:flex;

    gap:6px;

    overflow-x:auto;

    margin-bottom:12px;

    padding-bottom:5px;
}

button {

    padding:
    9px 12px;

    border-radius:8px;

    border:
    1px solid #39485b;

    background:#17212d;

    color:#ffffff;

    white-space:nowrap;

    cursor:pointer;
}

button.active {

    background:#6d28d9;

    border-color:#8b5cf6;
}

.index-buttons {

    display:flex;

    gap:7px;

    overflow-x:auto;

    padding-bottom:5px;
}

.result {

    font-size:16px;

    line-height:1.7;
}

.signal {

    font-size:25px;

    font-weight:bold;

    margin-bottom:8px;
}

.call {

    color:#54dc9a;
}

.put {

    color:#ff6673;
}

.wait {

    color:#ffd45a;
}

.value {

    margin:
    3px 0;
}

.chart-card {

    padding:8px;
}

#chart {

    width:100%;

    height:520px;

    border-radius:10px;

    overflow:hidden;

    background:#0d141d;
}

.chart-info {

    display:flex;

    flex-wrap:wrap;

    gap:12px;

    padding:
    8px 4px;

    font-size:12px;

    color:#cbd5e1;
}

.ema9-label {

    color:#facc15;
}

.ema15-label {

    color:#38bdf8;
}

.vwap-label {

    color:#c084fc;
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

    padding:9px 0;

    font-size:12px;

    line-height:1.6;
}

.win {

    color:#54dc9a;
}

.loss {

    color:#ff6673;
}

.small {

    font-size:12px;

    color:#9ca3af;
}

.loading {

    text-align:center;

    padding:20px;

    color:#9ca3af;
}

.error {

    color:#ff6673;

    padding:10px;
}

</style>

</head>

<body>

<h1>
⚡ Personal Scalping Scanner
</h1>


<!-- =====================================================
     TIMEFRAME
     ===================================================== -->

<div class="card">

<div class="small">
TIMEFRAME
</div>

<div class="tf">

<button
id="tf-1m"
onclick="selectTF('1m')">
1M
</button>

<button
id="tf-2m"
onclick="selectTF('2m')">
2M
</button>

<button
id="tf-3m"
onclick="selectTF('3m')">
3M
</button>

<button
id="tf-5m"
onclick="selectTF('5m')">
5M
</button>

<button
id="tf-15m"
onclick="selectTF('15m')">
15M
</button>

<button
id="tf-1h"
onclick="selectTF('1h')">
1H
</button>

<button
id="tf-2h"
onclick="selectTF('2h')">
2H
</button>

<button
id="tf-1d"
onclick="selectTF('1d')">
1D
</button>

<button
id="tf-1wk"
onclick="selectTF('1wk')">
1W
</button>

</div>

</div>


<!-- =====================================================
     INDEX
     ===================================================== -->

<div class="card">

<div class="small">
INDEX
</div>

<div class="index-buttons">

<button
id="index-NIFTY 50"
onclick="selectIndex('NIFTY 50')">
NIFTY 50
</button>

<button
id="index-BANK NIFTY"
onclick="selectIndex('BANK NIFTY')">
BANK NIFTY
</button>

<button
id="index-SENSEX"
onclick="selectIndex('SENSEX')">
SENSEX
</button>

</div>

</div>


<!-- =====================================================
     SCANNER RESULT
     ===================================================== -->

<div class="card">

<div class="small">
LIVE SIGNAL
</div>

<div id="result">

<div class="loading">
Loading...
</div>

</div>

</div>


<!-- =====================================================
     CHART
     ===================================================== -->

<h2>
📊 Index Chart
</h2>

<div class="card chart-card">

<div id="chart"></div>

<div class="chart-info">

<span class="ema9-label">
● EMA 9
</span>

<span class="ema15-label">
● EMA 15
</span>

<span class="vwap-label">
● VWAP
</span>

<span>
▲ CALL
</span>

<span>
▼ PUT
</span>

</div>

</div>


<!-- =====================================================
     BACKTEST
     ===================================================== -->

<h2>
📈 Backtest
</h2>

<div class="card">

<div class="stats">

<div class="stat">

<div class="stat-title">
Total Trades
</div>

<div
class="stat-value"
id="totalTrades">
-
</div>

</div>

<div class="stat">

<div class="stat-title">
Wins
</div>

<div
class="stat-value"
id="wins">
-
</div>

</div>

<div class="stat">

<div class="stat-title">
Losses
</div>

<div
class="stat-value"
id="losses">
-
</div>

</div>

<div class="stat">

<div class="stat-title">
Win Rate
</div>

<div
class="stat-value"
id="winRate">
-
</div>

</div>

<div class="stat">

<div class="stat-title">
Net Points
</div>

<div
class="stat-value"
id="netPoints">
-
</div>

</div>

</div>

</div>


<div class="card">

<div class="small">
RECENT TRADES
</div>

<div id="trades">

-
    
</div>

</div>


<script>

let currentIndex = "NIFTY 50";

let currentTF = "5m";

let chart = null;

let candleSeries = null;

let ema9Series = null;

let ema15Series = null;

let vwapSeries = null;

let refreshTimer = null;


/* =====================================================
   FORMAT NUMBER
   ===================================================== */

function formatNumber(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "-";
    }

    return Number(value)
        .toLocaleString(
            "en-IN",
            {
                maximumFractionDigits:2
            }
        );
}


/* =====================================================
   SET ACTIVE BUTTON
   ===================================================== */

function updateButtons() {

    document
        .querySelectorAll(".tf button")
        .forEach(
            b => b.classList.remove("active")
        );

    const tfButton =
        document.getElementById(
            "tf-" + currentTF
        );

    if (tfButton) {
        tfButton.classList.add("active");
    }


    document
        .querySelectorAll(".index-buttons button")
        .forEach(
            b => b.classList.remove("active")
        );

    const indexButton =
        document.getElementById(
            "index-" + currentIndex
        );

    if (indexButton) {
        indexButton.classList.add("active");
    }

}


/* =====================================================
   SELECT TIMEFRAME
   ===================================================== */

function selectTF(tf) {

    currentTF = tf;

    updateButtons();

    loadAll();

}


/* =====================================================
   SELECT INDEX
   ===================================================== */

function selectIndex(index) {

    currentIndex = index;

    updateButtons();

    loadAll();

}


/* =====================================================
   CREATE CHART
   ===================================================== */

function createChart() {

    const container =
        document.getElementById(
            "chart"
        );

    if (chart) {

        chart.remove();

        chart = null;
    }

    chart =
        LightweightCharts.createChart(
            container,
            {

                width:
                    container.clientWidth,

                height:520,

                layout: {

                    background: {
                        color:"#0d141d"
                    },

                    textColor:"#cbd5e1"

                },

                grid: {

                    vertLines: {
                        color:"#1b2633"
                    },

                    horzLines: {
                        color:"#1b2633"
                    }

                },

                rightPriceScale: {

                    borderColor:
                        "#263241"

                },

                timeScale: {

                    borderColor:
                        "#263241",

                    timeVisible:true,

                    secondsVisible:false

                }

            }
        );


    candleSeries =
        chart.addCandlestickSeries({

            upColor:"#22c55e",

            downColor:"#ef4444",

            borderUpColor:"#22c55e",

            borderDownColor:"#ef4444",

            wickUpColor:"#22c55e",

            wickDownColor:"#ef4444"

        });


    ema9Series =
        chart.addLineSeries({

            lineWidth:2,

            color:"#facc15"

        });


    ema15Series =
        chart.addLineSeries({

            lineWidth:2,

            color:"#38bdf8"

        });


    vwapSeries =
        chart.addLineSeries({

            lineWidth:2,

            color:"#c084fc"

        });


    window.addEventListener(
        "resize",
        resizeChart
    );

}


/* =====================================================
   RESIZE CHART
   ===================================================== */

function resizeChart() {

    const container =
        document.getElementById(
            "chart"
        );

    if (
        chart &&
        container
    ) {

        chart.applyOptions({

            width:
                container.clientWidth

        });

    }

}


/* =====================================================
   LOAD CHART
   ===================================================== */

async function loadChart() {

    try {

        const url =
            "/api/chart?index="
            + encodeURIComponent(
                currentIndex
            )
            + "&tf="
            + encodeURIComponent(
                currentTF
            );

        const response =
            await fetch(url);

        const data =
            await response.json();

        if (
            !Array.isArray(data)
            || data.length === 0
        ) {

            return;

        }

        if (!chart) {

            createChart();

        }


        const candles = [];

        const ema9 = [];

        const ema15 = [];

        const vwap = [];

        const markers = [];


        data.forEach(
            row => {

                candles.push({

                    time:row.time,

                    open:row.open,

                    high:row.high,

                    low:row.low,

                    close:row.close

                });


                ema9.push({

                    time:row.time,

                    value:row.ema9

                });


                ema15.push({

                    time:row.time,

                    value:row.ema15

                });


                vwap.push({

                    time:row.time,

                    value:row.vwap

                });


                if (
                    row.marker === "CALL"
                ) {

                    markers.push({

                        time:row.time,

                        position:"belowBar",

                        color:"#22c55e",

                        shape:"arrowUp",

                        text:"CALL"

                    });

                }


                if (
                    row.marker === "PUT"
                ) {

                    markers.push({

                        time:row.time,

                        position:"aboveBar",

                        color:"#ef4444",

                        shape:"arrowDown",

                        text:"PUT"

                    });

                }

            }
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


        if (
            candleSeries.setMarkers
        ) {

            candleSeries.setMarkers(
                markers
            );

        }


        chart.timeScale()
            .fitContent();

    }

    catch(error) {

        console.log(
            "Chart error:",
            error
        );

    }

}


/* =====================================================
   LOAD SCANNER
   ===================================================== */

async function loadScanner() {

    const result =
        document.getElementById(
            "result"
        );

    try {

        const url =
            "/api/scanner?index="
            + encodeURIComponent(
                currentIndex
            )
            + "&tf="
            + encodeURIComponent(
                currentTF
            );

        const response =
            await fetch(url);

        const data =
            await response.json();


        if (data.error) {

            result.innerHTML =
                "<div class='error'>"
                + data.error
                + "</div>";

            return;

        }


        let signalClass =
            "wait";

        let signalText =
            data.signal;


        if (
            data.signal === "CALL"
        ) {

            signalClass = "call";

            signalText =
                "🟢 CALL";

        }

        else if (
            data.signal === "PUT"
        ) {

            signalClass = "put";

            signalText =
                "🔴 PUT";

        }

        else {

            signalText =
                "🟡 WAIT";

        }


        result.innerHTML = `

            <div class="signal ${signalClass}">
                ${signalText}
            </div>

            <div class="value">
                <b>Index:</b>
                ${currentIndex}
            </div>

            <div class="value">
                <b>Timeframe:</b>
                ${currentTF}
            </div>

            <div class="value">
                <b>Price:</b>
                ${formatNumber(data.price)}
            </div>

            <div class="value">
                <b>EMA 9:</b>
                ${formatNumber(data.ema9)}
            </div>

            <div class="value">
                <b>EMA 15:</b>
                ${formatNumber(data.ema15)}
            </div>

            <div class="value">
                <b>VWAP:</b>
                ${formatNumber(data.vwap)}
            </div>

            <div class="small">
                ${data.time || ""}
            </div>
        `;

    }

    catch(error) {

        result.innerHTML =
            "<div class='error'>"
            + "Scanner error"
            + "</div>";

        console.log(error);

    }

}


/* =====================================================
   LOAD BACKTEST
   ===================================================== */

async function loadBacktest() {

    try {

        const url =
            "/api/backtest?index="
            + encodeURIComponent(
                currentIndex
            )
            + "&tf="
            + encodeURIComponent(
                currentTF
            );

        const response =
            await fetch(url);

        const data =
            await response.json();


        document.getElementById(
            "totalTrades"
        ).textContent =
            data.total_trades ?? "-";


        document.getElementById(
            "wins"
        ).textContent =
            data.wins ?? "-";


        document.getElementById(
            "losses"
        ).textContent =
            data.losses ?? "-";


        document.getElementById(
            "winRate"
        ).textContent =
            data.win_rate !== undefined
            ? data.win_rate + "%"
            : "-";


        document.getElementById(
            "netPoints"
        ).textContent =
            data.net_points ?? "-";


        const trades =
            document.getElementById(
                "trades"
            );


        if (
            !data.trades ||
            data.trades.length === 0
        ) {

            trades.innerHTML =
                "<div class='small'>"
                + "No trades"
                + "</div>";

            return;

        }


        const recent =
            data.trades
                .slice(-20)
                .reverse();


        trades.innerHTML =
            recent.map(
                t => {

                    const cls =
                        t.result === "WIN"
                        ? "win"
                        : "loss";

                    return `

                    <div class="trade">

                        <b>${t.type}</b>

                        &nbsp; |

                        Entry:
                        ${t.entry}

                        &nbsp; |

                        Exit:
                        ${t.exit}

                        &nbsp; |

                        Points:
                        <span class="${cls}">
                            ${t.points}
                        </span>

                        &nbsp; |

                        <span class="${cls}">
                            ${t.result}
                        </span>

                        <br>

                        <span class="small">
                            ${t.entry_time}
                            →
                            ${t.exit_time}
                        </span>

                    </div>

                    `;

                }
            )
            .join("");

    }

    catch(error) {

        console.log(
            "Backtest error:",
            error
        );

    }

}


/* =====================================================
   LOAD EVERYTHING
   ===================================================== */

async function loadAll() {

    await loadScanner();

    await loadChart();

    await loadBacktest();

}


/* =====================================================
   AUTO REFRESH
   ===================================================== */

function startAutoRefresh() {

    if (refreshTimer) {

        clearInterval(
            refreshTimer
        );

    }

    refreshTimer =
        setInterval(
            loadAll,
            30000
        );

}


/* =====================================================
   INITIAL LOAD
   ===================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function() {

        updateButtons();

        createChart();

        loadAll();

        startAutoRefresh();

    }
);

</script>

</body>

</html>
"""

    return html


# =========================================================
# API - SCANNER
# =========================================================

@app.route(
    "/api/scanner"
)
def api_scanner():

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

    result = calculate_scanner(
        data
    )

    return jsonify(result)


# =========================================================
# API - CHART
# =========================================================

@app.route(
    "/api/chart"
)
def api_chart():

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

    result = chart_json(
        data
    )

    return jsonify(result)


# =========================================================
# API - BACKTEST
# =========================================================

@app.route(
    "/api/backtest"
)
def api_backtest():

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

    result = run_backtest(
        data
    )

    return jsonify(result)


# =========================================================
# API - HEALTH
# =========================================================

@app.route(
    "/api/health"
)
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
