import os
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Load environment variables from .env file
load_dotenv()

# Import Orchestrator & Chatbot Modules
try:
    from orchestrator import get_investment_recommendation
    from orchestrator.chatbot import get_chat_response
except ImportError as e:
    print(f"⚠️ Import Warning: Orchestrator modules missing or incomplete: {e}")
    get_investment_recommendation = None
    get_chat_response = None

app = Flask(__name__)
DEFAULT_TICKER = "NVDA"

# Global Ticker Mappings for Top Indices Bar
MACRO_TICKERS = [
    {"name": "NIFTY 50", "symbol": "^NSEI"},
    {"name": "BANK NIFTY", "symbol": "^NSEBANK"},
    {"name": "GOLD (XAU/USD)", "symbol": "GC=F"},
    {"name": "NASDAQ", "symbol": "^IXIC"},
]


def fetch_macro_overview():
    """Fetches real-time spot prices and percentage changes for global indices using batch queries."""
    overview = []
    symbols_list = [item["symbol"] for item in MACRO_TICKERS]

    try:
        tickers = yf.Tickers(" ".join(symbols_list))
        for item in MACRO_TICKERS:
            name = item["name"]
            symbol = item["symbol"]
            try:
                t = tickers.tickers[symbol]
                fast = t.fast_info
                last_price = getattr(fast, "last_price", None) or getattr(
                    fast, "previous_close", 0.0
                )
                prev_close = (
                    getattr(fast, "previous_close", None) or last_price
                )
                chg_pct = 0.0

                if prev_close and prev_close > 0:
                    chg_pct = ((last_price - prev_close) / prev_close) * 100

                overview.append(
                    {
                        "name": name,
                        "symbol": symbol,
                        "price": round(float(last_price), 2),
                        "change_pct": round(float(chg_pct), 2),
                    }
                )
            except Exception:
                overview.append(
                    {"name": name, "symbol": symbol, "price": 0.0, "change_pct": 0.0}
                )
    except Exception as e:
        print(f"⚠️ Macro overview fetch failed: {e}")
        overview = [
            {"name": item["name"], "symbol": item["symbol"], "price": 0.0, "change_pct": 0.0}
            for item in MACRO_TICKERS
        ]

    return overview


@app.route("/", methods=["GET", "POST"])
def index():
    """Main Web Dashboard Route."""
    ticker = DEFAULT_TICKER
    if request.method == "POST":
        ticker = request.form.get("symbol", "").strip().upper() or DEFAULT_TICKER
    elif request.method == "GET" and request.args.get("symbol"):
        ticker = request.args.get("symbol", "").strip().upper()

    macro_data = fetch_macro_overview()
    trade_analysis = None
    stock_news = []
    error_message = None

    if get_investment_recommendation:
        try:
            # 1. Run Quantitative + Fundamental + Risk Engine
            trade_analysis = get_investment_recommendation(ticker)

            # 2. Fetch Live Stock-Specific News Feed
            t_obj = yf.Ticker(ticker)
            raw_news = t_obj.news or []

            for item in raw_news[:5]:
                content = item.get("content", {})
                title = content.get("title") or item.get("title", "Market Update")
                publisher = (
                    content.get("provider", {}).get("displayName")
                    or item.get("publisher", "Financial News")
                )
                link = (
                    content.get("canonicalUrl", {}).get("url")
                    or item.get("link", "#")
                )
                summary = content.get("summary") or item.get("summary", "")

                stock_news.append(
                    {
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "summary": summary[:180] + "..." if len(summary) > 180 else summary,
                    }
                )
        except Exception as e:
            print(f"❌ Error analyzing ticker '{ticker}': {e}")
            error_message = (
                f"Could not complete analysis for '{ticker}'. Please verify the symbol."
            )
    else:
        error_message = "Orchestrator module is offline or missing."

    return render_template(
        "index.html",
        ticker=ticker,
        macro_data=macro_data,
        trade_analysis=trade_analysis,
        stock_news=stock_news,
        error_message=error_message,
    )


@app.route("/api/search_stocks", methods=["GET"])
def api_search_stocks():
    """TradingView-Style Universal Dynamic Search Endpoint.

    Searches US Stocks, Indian Stocks (NSE/BSE), Cryptos, Commodities, and Forex.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    results = []
    try:
        # Live Yahoo Finance Search Query
        search_engine = yf.Search(query, max_results=8)
        quotes = getattr(search_engine, "quotes", [])

        for quote in quotes:
            symbol = quote.get("symbol", "")
            shortname = (
                quote.get("shortname") or quote.get("longname") or symbol
            )
            exchange = quote.get("exchange", "")
            quote_type = quote.get("quoteType", "EQUITY")

            results.append(
                {
                    "symbol": symbol,
                    "name": shortname,
                    "exchange": exchange,
                    "type": quote_type,
                }
            )
    except Exception as e:
        print(f"Search API Exception: {e}")

    # Fallback Curated List if API returns empty
    if not results:
        FALLBACK_LIST = [
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE", "type": "EQUITY"},
            {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "exchange": "NSE", "type": "EQUITY"},
            {"symbol": "BTC-USD", "name": "Bitcoin USD", "exchange": "CCC", "type": "CRYPTO"},
            {"symbol": "GC=F", "name": "Gold Futures", "exchange": "NYM", "type": "FUTURE"},
        ]
        q_upper = query.upper()
        results = [
            s
            for s in FALLBACK_LIST
            if q_upper in s["symbol"] or q_upper in s["name"].upper()
        ]

    return jsonify(results[:8])


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """JSON API Endpoint for the AI Assistant Widget."""
    if not get_chat_response:
        return jsonify({"success": False, "response": "Chatbot offline."}), 500

    data = request.get_json() or {}
    symbol = data.get("symbol", DEFAULT_TICKER).strip().upper()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_session")

    if not message:
        return jsonify({"success": False, "response": "Please enter a message."}), 400

    try:
        reply = get_chat_response(symbol=symbol, message=message, session_id=session_id)
        return jsonify({"success": True, "response": reply, "session_id": session_id})
    except Exception as e:
        return jsonify({"success": False, "response": f"Error processing query: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 Agentic Quant & Financial Intelligence Workstation")
    print(f"🔑 Groq API Key Status: {'Active' if os.getenv('GROQ_API_KEY') else 'Missing (.env)'}")
    print("=======================================================\n")
    app.run(debug=True, host="0.0.0.0", port=5000)