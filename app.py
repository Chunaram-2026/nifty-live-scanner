from flask import Flask, jsonify, request
import yfinance as yf

app = Flask(__name__)

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN"
}

TIMEFRAMES = ["1m", "2m", "3m", "5m", "15m", "1h", "2h", "1d", "1wk"]


def get_data(symbol, timeframe):
    data = yf.download(
        symbol,
        period="7d",
        interval=timeframe,
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        return None

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


def calculate_signal(data):
    if data is None or len(data) < 20:
        return {
            "signal": "NO DATA",
            "price": None,
            "ema9": None,
            "ema15": None,
            "vwap": None
        }

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema15 = close.ewm(span=15, adjust=False).mean()

    typical_price = (high + low + close) / 3

    volume_sum = volume.cumsum()

    vwap = (
        typical_price * volume
    ).cumsum() / volume_sum

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


@app.route("/")
def home():

    html = """
    <html>
    <head>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Personal Scalping Scanner</title>

    <style>
    body {
        background:#080c12;
        color:white;
        font-family:Arial;
        padding:15px;
    }

    .card {
        background:#111923;
        border:1px solid #263241;
        border-radius:12px;
        padding:15px;
        margin-bottom:12px;
    }

    h1 {
        font-size:22px;
    }

    h2 {
        margin-bottom:5px;
    }

    .tf {
        display:flex;
        gap:6px;
        overflow:auto;
        margin-bottom:15px;
    }

    button {
        padding:9px 13px;
        border-radius:8px;
        border:1px solid #39485b;
        background:#17212d;
        color:white;
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
    </style>
    </head>

    <body>

    <h1>⚡ Personal Scalping Scanner</h1>

    <div class="tf">
        <button onclick="loadData('1m')">1M</button>
        <button onclick="loadData('2m')">2M</button>
        <button onclick="loadData('3m')">3M</button>
        <button onclick="loadData('5m')">5M</button>
        <button onclick="loadData('15m')">15M</button>
        <button onclick="loadData('1h')">1H</button>
<button onclick="loadData('2h')">2H</button>
<button onclick="loadData('1d')">1D</button>
<button onclick="loadData('1wk')">1W</button>
    </div>

    <div id="result">5M data loading...</div>

    <script>

    function loadData(tf) {

        document.getElementById("result").innerHTML =
        "Loading " + tf + " data...";

        fetch("/api/scan?tf=" + tf)
        .then(response => response.json())
        .then(data => {

            let html = "";

            for (const index in data) {

                const x = data[index];

                let cls = "wait";

                if (x.signal.includes("CALL"))
                    cls = "call";

                if (x.signal.includes("PUT"))
                    cls = "put";

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

            document.getElementById("result").innerHTML = html;
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

    timeframe = request.args.get("tf", "5m")

    if timeframe not in TIMEFRAMES:
        timeframe = "5m"

    result = {}

    for name, symbol in INDICES.items():

        data = get_data(symbol, timeframe)

        result[name] = calculate_signal(data)

    return jsonify(result)


if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
