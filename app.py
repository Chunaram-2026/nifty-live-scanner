from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import os

app = Flask(__name__)

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN"
}

TIMEFRAMES = [
    "1m", "2m", "3m", "5m",
    "15m", "1h", "2h", "1d", "1wk"
]


def clean_columns(data):
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in data.columns:
            return None

    if "Volume" not in data.columns:
        data["Volume"] = 0

    return data.dropna(subset=required)


def download_data(symbol, interval, period):
    try:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        return clean_columns(data)

    except Exception:
        return None


def resample_data(data, rule):
    if data is None or data.empty:
        return None

    try:
        result = data.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })

        return result.dropna(subset=["Open", "High", "Low", "Close"])

    except Exception:
        return None


def get_data(symbol, timeframe):

    # 1 minute data is used as the base for
    # 1m, 2m, 3m and 5m.
    if timeframe in ["1m", "2m", "3m", "5m"]:

        data = download_data(
            symbol,
            "1m",
            "7d"
        )

        if data is None or data.empty:
            return None

        if timeframe == "1m":
            return data

        if timeframe == "2m":
            return resample_data(data, "2min")

        if timeframe == "3m":
            return resample_data(data, "3min")

        if timeframe == "5m":
            return resample_data(data, "5min")


    # 15 minute
    if timeframe == "15m":

        return download_data(
            symbol,
            "15m",
            "60d"
        )


    # 1 hour
    if timeframe == "1h":

        return download_data(
            symbol,
            "1h",
            "730d"
        )


    # 2 hour is created from 1 hour candles
    if timeframe == "2h":

        data = download_data(
            symbol,
            "1h",
            "730d"
        )

        return resample_data(data, "2h")


    # Daily
    if timeframe == "1d":

        return download_data(
            symbol,
            "1d",
            "5y"
        )


    # Weekly
    if timeframe == "1wk":

        return download_data(
            symbol,
            "1wk",
            "10y"
        )


    return None


def calculate_signal(data):

    if data is None or data.empty:
        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    if len(data) < 20:
        return {
            "signal": "WAIT - LOW DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    try:

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

        valid = (
            close.notna()
            & high.notna()
            & low.notna()
        )

        close = close[valid]
        high = high[valid]
        low = low[valid]
        volume = volume[valid]

        if len(close) < 20:
            return {
                "signal": "NO DATA",
                "price": None,
                "ema9": None,
                "ema15": None,
                "vwap": None
            }

        ema9 = close.ewm(
            span=9,
            adjust=False
        ).mean()

        ema15 = close.ewm(
            span=15,
            adjust=False
        ).mean()

        typical_price = (
            high + low + close
        ) / 3

        # VWAP
        cumulative_volume = volume.cumsum()

        cumulative_value = (
            typical_price * volume
        ).cumsum()

        if cumulative_volume.iloc[-1] > 0:

            vwap = (
                cumulative_value
                / cumulative_volume
            )

        else:
            # Some index feeds provide zero volume.
            # Use typical price as fallback.
            vwap = typical_price

        price = float(close.iloc[-1])
        e9 = float(ema9.iloc[-1])
        e15 = float(ema15.iloc[-1])
        vw = float(vwap.iloc[-1])

        if price > vw and e9 > e15:

            signal = "CALL BIAS"

        elif price < vw and e9 < e15:

            signal = "PUT BIAS"

        else:

            signal = "WAIT"

        return {
            "signal": signal,
            "price": round(price, 2),
            "ema9": round(e9, 2),
            "ema15": round(e15, 2),
            "vwap": round(vw, 2)
        }

    except Exception as e:

        return {
            "signal": "DATA ERROR",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }


@app.route("/")
def home():

    html = """
    <!DOCTYPE html>

    <html>

    <head>

    <meta name="viewport"
          content="width=device-width,initial-scale=1">

    <title>Personal Scalping Scanner</title>

    <style>

    body {
        background:#080c12;
        color:white;
        font-family:Arial,sans-serif;
        padding:15px;
        margin:0;
    }

    h1 {
        font-size:22px;
        margin-bottom:15px;
    }

    .tf {
        display:flex;
        gap:6px;
        overflow-x:auto;
        margin-bottom:15px;
        padding-bottom:5px;
    }

    button {
        padding:9px 13px;
        border-radius:8px;
        border:1px solid #39485b;
        background:#17212d;
        color:white;
        white-space:nowrap;
    }

    button.active {
        background:#5b16a8;
    }

    .card {
        background:#111923;
        border:1px solid #263241;
        border-radius:12px;
        padding:15px;
        margin-bottom:12px;
    }

    .value {
        margin:7px 0;
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
    }

    </style>

    </head>

    <body>

    <h1>⚡ Personal Scalping Scanner</h1>

    <div class="tf">

        <button id="b1m"
                onclick="loadData('1m')">
            1M
        </button>

        <button id="b2m"
                onclick="loadData('2m')">
            2M
        </button>

        <button id="b3m"
                onclick="loadData('3m')">
            3M
        </button>

        <button id="b5m"
                onclick="loadData('5m')">
            5M
        </button>

        <button id="b15m"
                onclick="loadData('15m')">
            15M
        </button>

        <button id="b1h"
                onclick="loadData('1h')">
            1H
        </button>

        <button id="b2h"
                onclick="loadData('2h')">
            2H
        </button>

        <button id="b1d"
                onclick="loadData('1d')">
            1D
        </button>

        <button id="b1wk"
                onclick="loadData('1wk')">
            1W
        </button>

    </div>

    <div id="result">
        Loading 5M data...
    </div>


    <script>

    function loadData(tf) {

        document.getElementById("result").innerHTML =
            "Loading " + tf + " data...";

        document.querySelectorAll("button")
            .forEach(function(btn) {
                btn.classList.remove("active");
            });

        let selected =
            document.getElementById("b" + tf);

        if (selected) {
            selected.classList.add("active");
        }

        fetch("/api/scan?tf=" + tf)

        .then(function(response) {

            if (!response.ok) {
                throw new Error("Server error");
            }

            return response.json();

        })

        .then(function(data) {

            let html = "";

            for (const index in data) {

                const x = data[index];

                let cls = "wait";

                if (x.signal &&
                    x.signal.includes("CALL")) {
                    cls = "call";
                }

                if (x.signal &&
                    x.signal.includes("PUT")) {
                    cls = "put";
                }

                if (x.signal &&
                    x.signal.includes("ERROR")) {
                    cls = "error";
                }

                html += `

                <div class="card">

                    <h2>${index}</h2>

                    <div class="value">
                    Price: ${x.price ?? "-"}
                    </div>

                    <div class="value">
                    EMA 9: ${x.ema9 ?? "-"}
                    </div>

                    <div class="value">
                    EMA 15: ${x.ema15 ?? "-"}
                    </div>

                    <div class="value">
                    VWAP: ${x.vwap ?? "-"}
                    </div>

                    <h3 class="${cls}">
                    ${x.signal}
                    </h3>

                </div>

                `;
            }

            document.getElementById("result")
                .innerHTML = html;

        })

        .catch(function(error) {

            document.getElementById("result")
                .innerHTML =
                '<div class="card error">' +
                'DATA CONNECTION ERROR' +
                '</div>';

        });

    }

    loadData("5m");

    </script>

    </body>

    </html>
    """

    return html


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

        data = get_data(
            symbol,
            timeframe
        )

        result[name] = calculate_signal(data)

    return jsonify(result)


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
