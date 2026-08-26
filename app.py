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

    try:

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

    except Exception:

        data["VWAP"] = typical_price

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

def get_signal(row, previous=None):

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

        current_high = float(
            row["High"]
        )

        current_low = float(
            row["Low"]
        )

    except Exception:

        return "WAIT"

    if any(
        pd.isna(x)
        for x in [
            price,
            ema9,
            ema15,
            vwap,
            current_high,
            current_low
        ]
    ):

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
    # CALL LIQUIDITY SWEEP
    # =====================================================

    buy_sweep = (
        current_low < prev_low
        and price > prev_low
    )

    if (
        price > vwap
        and ema9 > ema15
        and buy_sweep
    ):

        return "CALL"

    # =====================================================
    # PUT LIQUIDITY SWEEP
    # =====================================================

    sell_sweep = (
        current_high > prev_high
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

    markers = []

    previous_signal = "WAIT"

    previous_row = None

    for _, row in data.iterrows():

        current = get_signal(
            row,
            previous_row
        )

        signals.append(current)

        marker = ""

        if (
            current == "CALL"
            and previous_signal != "CALL"
        ):

            marker = "CALL"

        elif (
            current == "PUT"
            and previous_signal != "PUT"
        ):

            marker = "PUT"

        markers.append(marker)

        if current in [
            "CALL",
            "PUT"
        ]:

            previous_signal = current

        elif current == "WAIT":

            previous_signal = "WAIT"

        previous_row = row

    data["Signal"] = signals

    data["MARKER"] = markers

    return data


# =========================================================
# ALL INDICATORS + SIGNALS
# =========================================================

def prepare_data(data):

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

    if (
        data is None
        or len(data) < 20
    ):

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None,
            "time": None
        }

    data = prepare_data(data)

    if data is None or data.empty:

        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None,
            "time": None
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

    if data is None or data.empty:

        return []

    data = prepare_data(data)

    if data is None or data.empty:

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

                "signal": str(
                    row.get(
                        "Signal",
                        "WAIT"
                    )
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
#
# अभी SL / TARGET शामिल नहीं हैं.
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

    data = prepare_data(data)

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

    trades = []

    position = None

    entry_price = None

    entry_time = None

    position_type = None

    # =====================================================
    # SIGNAL CANDLE
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
        # ENTRY
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

                position = "CALL"

                position_type = "CALL"

                entry_price = next_open

                entry_time = next_time

    # =====================================================
    # CLOSE LAST POSITION
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

    total_trades = len(trades)

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

<!--
IMPORTANT:
Use a fixed compatible Lightweight Charts version.
-->

<script src=
"https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js">
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

.value {

    margin:
        6px 0;
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

.error {

    color:#ff6673;

    padding:10px;

    background:#241015;

    border-radius:8px;
}

.chart-wrapper {

    width:100%;

    height:520px;

    background:#0d141d;

    border:
        1px solid #263241;

    border-radius:10px;

    overflow:hidden;
}

#chart {

    width:100%;

    height:100%;
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

.green {

    color:#54dc9a;
}

.red {

    color:#ff6673;
}

.yellow {

    color:#ffd45a;
}

.small {

    font-size:12px;

    color:#9ca3af;
}

</style>

</head>

<body>

<h1>
⚡ Personal Scalping Scanner
</h1>


<!-- =====================================================
     TIMEFRAME BUTTONS
     ===================================================== -->

<div class="tf">

<button
data-tf="1m"
onclick="selectTimeframe('1m')">
1M
</button>

<button
data-tf="2m"
onclick="selectTimeframe('2m')">
2M
</button>

<button
data-tf="3m"
onclick="selectTimeframe('3m')">
3M
</button>

<button
data-tf="5m"
onclick="selectTimeframe('5m')">
5M
</button>

<button
data-tf="15m"
onclick="selectTimeframe('15m')">
15M
</button>

<button
data-tf="1h"
onclick="selectTimeframe('1h')">
1H
</button>

<button
data-tf="2h"
onclick="selectTimeframe('2h')">
2H
</button>

<button
data-tf="1d"
onclick="selectTimeframe('1d')">
1D
</button>

<button
data-tf="1wk"
onclick="selectTimeframe('1wk')">
1W
</button>

</div>


<!-- =====================================================
     INDEX BUTTONS
     ===================================================== -->

<div class="tf">

<button
data-index="NIFTY 50"
onclick="selectIndex('NIFTY 50')">
NIFTY 50
</button>

<button
data-index="BANK NIFTY"
onclick="selectIndex('BANK NIFTY')">
BANK NIFTY
</button>

<button
data-index="SENSEX"
onclick="selectIndex('SENSEX')">
SENSEX
</button>

</div>


<!-- =====================================================
     SCANNER RESULT
     ===================================================== -->

<div id="result">

Loading...

</div>


<!-- =====================================================
     CHART
     ===================================================== -->

<h2>
📊 Index Chart
</h2>

<div class="chart-wrapper">

<div id="chart"></div>

</div>


<!-- =====================================================
     BACKTEST
     ===================================================== -->

<h2>
📈 Backtest
</h2>

<div
class="card"
id="backtest">
Loading...
</div>


<script>

/* ========================================================
   GLOBAL VARIABLES
======================================================== */

let currentIndex = "NIFTY 50";

let currentTf = "5m";

let chart = null;

let candleSeries = null;

let ema9Series = null;

let ema15Series = null;

let vwapSeries = null;


/* ========================================================
   INDEX SYMBOLS
======================================================== */

const INDEX_SYMBOLS = {

    "NIFTY 50":
        "^NSEI",

    "BANK NIFTY":
        "^NSEBANK",

    "SENSEX":
        "^BSESN"

};


/* ========================================================
   SELECT TIMEFRAME
======================================================== */

function selectTimeframe(tf) {

    currentTf = tf;

    document
        .querySelectorAll(
            "[data-tf]"
        )
        .forEach(button => {

            button.classList.remove(
                "active"
            );

        });

    const button =
        document.querySelector(
            `[data-tf="${tf}"]`
        );

    if (button) {

        button.classList.add(
            "active"
        );

    }

    loadScanner();

    loadChart();

    loadBacktest();

}


/* ========================================================
   SELECT INDEX
======================================================== */

function selectIndex(indexName) {

    currentIndex = indexName;

    document
        .querySelectorAll(
            "[data-index]"
        )
        .forEach(button => {

            button.classList.remove(
                "active"
            );

        });

    const button =
        document.querySelector(
            `[data-index="${indexName}"]`
        );

    if (button) {

        button.classList.add(
            "active"
        );

    }

    loadScanner();

    loadChart();

    loadBacktest();

}


/* ========================================================
   LOAD SCANNER
======================================================== */

async function loadScanner() {

    const result =
        document.getElementById(
            "result"
        );

    result.innerHTML =
        "Loading " +
        currentTf +
        " data...";

    try {

        const response =
            await fetch(
                "/api/scan?tf=" +
                encodeURIComponent(
                    currentTf
                )
            );

        if (!response.ok) {

            throw new Error(
                "Scanner HTTP error"
            );

        }

        const data =
            await response.json();

        const x =
            data[currentIndex];

        if (!x) {

            result.innerHTML =
                '<div class="error">' +
                "No data available" +
                "</div>";

            return;

        }

        let cls = "wait";

        if (
            x.signal &&
            x.signal.includes("CALL")
        ) {

            cls = "call";

        }

        else if (
            x.signal &&
            x.signal.includes("PUT")
        ) {

            cls = "put";

        }

        result.innerHTML = `

        <div class="card">

            <h2>
                ${currentIndex}
            </h2>

            <div class="value">
                Price:
                ${x.price ?? "-"}
            </div>

            <div class="value">
                EMA 9:
                ${x.ema9 ?? "-"}
            </div>

            <div class="value">
                EMA 15:
                ${x.ema15 ?? "-"}
            </div>

            <div class="value">
                VWAP:
                ${x.vwap ?? "-"}
            </div>

            <div class="small">
                Time:
                ${x.time ?? "-"}
            </div>

            <h3 class="${cls}">
                ${x.signal ?? "WAIT"}
            </h3>

        </div>

        `;

    }

    catch (error) {

        console.error(
            error
        );

        result.innerHTML =
            '<div class="error">' +
            "Scanner error: " +
            error.message +
            "</div>";

    }

}


/* ========================================================
   CREATE CHART
======================================================== */

function createChart() {

    const container =
        document.getElementById(
            "chart"
        );

    if (!container) {

        return;

    }

    if (chart) {

        try {

            chart.remove();

        }

        catch (e) {

            console.log(e);

        }

        chart = null;

    }

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

                        color:
                            "#0d141d"

                    },

                    textColor:
                        "#d1d5db"

                },

                grid: {

                    vertLines: {

                        color:
                            "#182330"

                    },

                    horzLines: {

                        color:
                            "#182330"

                    }

                },

                crosshair: {

                    mode:
                        LightweightCharts.CrosshairMode.Normal

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


    /* ====================================================
       CANDLE SERIES
    ==================================================== */

    candleSeries =
        chart.addCandlestickSeries({

            upColor:
                "#26a69a",

            downColor:
                "#ef5350",

            borderUpColor:
                "#26a69a",

            borderDownColor:
                "#ef5350",

            wickUpColor:
                "#26a69a",

            wickDownColor:
                "#ef5350"

        });


    /* ====================================================
       EMA 9
    ==================================================== */

    ema9Series =
        chart.addLineSeries({

            color:
                "#ffd21f",

            lineWidth:
                2,

            priceLineVisible:
                false,

            lastValueVisible:
                true

        });


    /* ====================================================
       EMA 15
    ==================================================== */

    ema15Series =
        chart.addLineSeries({

            color:
                "#38bdf8",

            lineWidth:
                2,

            priceLineVisible:
                false,

            lastValueVisible:
                true

        });


    /* ====================================================
       VWAP
    ==================================================== */

    vwapSeries =
        chart.addLineSeries({

            color:
                "#a855f7",

            lineWidth:
                2,

            priceLineVisible:
                false,

            lastValueVisible:
                true

        });

}


/* ========================================================
   LOAD CHART DATA
======================================================== */

async function loadChart() {

    try {

        const symbol =
            INDEX_SYMBOLS[
                currentIndex
            ];

        if (!symbol) {

            return;

        }

        const response =
            await fetch(
                "/api/chart?symbol=" +
                encodeURIComponent(
                    symbol
                ) +
                "&tf=" +
                encodeURIComponent(
                    currentTf
                )
            );

        if (!response.ok) {

            throw new Error(
                "Chart HTTP error"
            );

        }

        const data =
            await response.json();

        if (
            !Array.isArray(data)
            || data.length === 0
        ) {

            console.log(
                "No chart data"
            );

            return;

        }

        createChart();

        const candles = [];

        const ema9 = [];

        const ema15 = [];

        const vwap = [];

        const markers = [];


        /* =================================================
           CONVERT DATA
        ================================================= */

        data.forEach(row => {

            const time =
                Number(row.time);

            const open =
                Number(row.open);

            const high =
                Number(row.high);

            const low =
                Number(row.low);

            const close =
                Number(row.close);

            if (
                !Number.isFinite(time)
                || !Number.isFinite(open)
                || !Number.isFinite(high)
                || !Number.isFinite(low)
                || !Number.isFinite(close)
            ) {

                return;

            }


            candles.push({

                time:
                    time,

                open:
                    open,

                high:
                    high,

                low:
                    low,

                close:
                    close

            });


            const e9 =
                Number(row.ema9);

            if (
                Number.isFinite(e9)
            ) {

                ema9.push({

                    time:
                        time,

                    value:
                        e9

                });

            }


            const e15 =
                Number(row.ema15);

            if (
                Number.isFinite(e15)
            ) {

                ema15.push({

                    time:
                        time,

                    value:
                        e15

                });

            }


            const vw =
                Number(row.vwap);

            if (
                Number.isFinite(vw)
            ) {

                vwap.push({

                    time:
                        time,

                    value:
                        vw

                });

            }


            /* =============================================
               CALL MARKER
            ============================================= */

            if (
                row.marker ===
                "CALL"
            ) {

                markers.push({

                    time:
                        time,

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


            /* =============================================
               PUT MARKER
            ============================================= */

            else if (
                row.marker ===
                "PUT"
            ) {

                markers.push({

                    time:
                        time,

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

        });


        /* =================================================
           SET SERIES
        ================================================= */

        if (
            candles.length > 0
        ) {

            candleSeries.setData(
                candles
            );

        }

        if (
            ema9.length > 0
        ) {

            ema9Series.setData(
                ema9
            );

        }

        if (
            ema15.length > 0
        ) {

            ema15Series.setData(
                ema15
            );

        }

        if (
            vwap.length > 0
        ) {

            vwapSeries.setData(
                vwap
            );

        }


        /* =================================================
           SET MARKERS
        ================================================= */

        if (
            markers.length > 0
        ) {

            markers.sort(
                (a, b) =>
                    a.time - b.time
            );

            candleSeries.setMarkers(
                markers
            );

        }


        /* =================================================
           FIT CONTENT
        ================================================= */

        chart
            .timeScale()
            .fitContent();

    }

    catch (error) {

        console.error(
            "CHART ERROR:",
            error
        );

    }

}


/* ========================================================
   LOAD BACKTEST
======================================================== */

async function loadBacktest() {

    const box =
        document.getElementById(
            "backtest"
        );

    box.innerHTML =
        "Loading backtest...";

    try {

        const symbol =
            INDEX_SYMBOLS[
                currentIndex
            ];

        const response =
            await fetch(
                "/api/backtest?symbol=" +
                encodeURIComponent(
                    symbol
                ) +
                "&tf=" +
                encodeURIComponent(
                    currentTf
                )
            );

        if (!response.ok) {

            throw new Error(
                "Backtest HTTP error"
            );

        }

        const data =
            await response.json();


        let winClass =
            "yellow";

        if (
            data.win_rate >= 50
        ) {

            winClass =
                "green";

        }

        else if (
            data.win_rate < 40
        ) {

            winClass =
                "red";

        }


        let pointsClass =
            "yellow";

        if (
            data.net_points > 0
        ) {

            pointsClass =
                "green";

        }

        else if (
            data.net_points < 0
        ) {

            pointsClass =
                "red";

        }


        let html = `

        <div class="stats">

            <div class="stat">

                <div class="stat-title">
                    Total Trades
                </div>

                <div class="stat-value">
                    ${data.total_trades}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    Wins
                </div>

                <div class="stat-value green">
                    ${data.wins}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    Losses
                </div>

                <div class="stat-value red">
                    ${data.losses}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    Win Rate
                </div>

                <div class="stat-value ${winClass}">
                    ${data.win_rate}%
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    Net Points
                </div>

                <div class="stat-value ${pointsClass}">
                    ${data.net_points}
                </div>

            </div>

        </div>

        <h3>
            Recent Trades
        </h3>

        `;


        const trades =
            Array.isArray(
                data.trades
            )
            ? data.trades
            : [];


        if (
            trades.length === 0
        ) {

            html +=
                '<div class="small">' +
                "No trades found" +
                "</div>";

        }

        else {

            const recent =
                trades.slice(
                    -20
                ).reverse();


            recent.forEach(
                trade => {

                    const cls =
                        trade.result ===
                        "WIN"
                        ? "green"
                        : "red";


                    html += `

                    <div class="trade">

                        <b>
                            ${trade.type}
                        </b>

                        &nbsp;

                        Entry:
                        ${trade.entry}

                        &nbsp;

                        Exit:
                        ${trade.exit}

                        &nbsp;

                        Points:
                        <span class="${cls}">
                            ${trade.points}
                        </span>

                        &nbsp;

                        <span class="${cls}">
                            ${trade.result}
                        </span>

                    </div>

                    `;

                }
            );

        }


        box.innerHTML =
            html;

    }

    catch (error) {

        console.error(
            error
        );

        box.innerHTML =
            '<div class="error">' +
            "Backtest error: " +
            error.message +
            "</div>";

    }

}


/* ========================================================
   RESIZE CHART
======================================================== */

window.addEventListener(
    "resize",
    () => {

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
                    container.clientWidth,

                height:
                    container.clientHeight

            });

        }

    }
);


/* ========================================================
   INITIAL LOAD
======================================================== */

window.addEventListener(
    "load",
    () => {

        document
            .querySelector(
                '[data-tf="5m"]'
            )
            .classList.add(
                "active"
            );

        document
            .querySelector(
                '[data-index="NIFTY 50"]'
            )
            .classList.add(
                "active"
            );

        loadScanner();

        loadChart();

        loadBacktest();

    }
);

</script>

</body>

</html>
"""

    return html


# =========================================================
# API: SCANNER
# =========================================================

@app.route("/api/scan")
def api_scan():

    timeframe = request.args.get(
        "tf",
        "5m"
    )

    if timeframe not in TIMEFRAMES:

        timeframe = "5m"

    result = {}

    for name, symbol in INDICES.items():

        data = download_data(
            symbol,
            timeframe
        )

        result[name] = calculate_scanner(
            data
        )

    return jsonify(result)


# =========================================================
# API: CHART
# =========================================================

@app.route("/api/chart")
def api_chart():

    symbol = request.args.get(
        "symbol",
        "^NSEI"
    )

    timeframe = request.args.get(
        "tf",
        "5m"
    )

    if timeframe not in TIMEFRAMES:

        timeframe = "5m"

    if symbol not in INDICES.values():

        symbol = "^NSEI"

    data = download_data(
        symbol,
        timeframe
    )

    result = chart_json(
        data
    )

    return jsonify(result)


# =========================================================
# API: BACKTEST
# =========================================================

@app.route("/api/backtest")
def api_backtest():

    symbol = request.args.get(
        "symbol",
        "^NSEI"
    )

    timeframe = request.args.get(
        "tf",
        "5m"
    )

    if timeframe not in TIMEFRAMES:

        timeframe = "5m"

    if symbol not in INDICES.values():

        symbol = "^NSEI"

    data = download_data(
        symbol,
        timeframe
    )

    result = run_backtest(
        data
    )

    return jsonify(result)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
