import os
import sys
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
import yfinance as yf

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

try:
    from fundamental_worker.rag_search import fetch_and_prioritize_global_news
except ImportError as e:
    print(f"⚠️ Import Warning (rag_search): {e}")
    fetch_and_prioritize_global_news = lambda: []

try:
    from orchestrator.chatbot import get_chat_response
except ImportError as e:
    print(f"⚠️ Import Warning (chatbot): {e}")
    get_chat_response = None

try:
    from orchestrator.decision_agent import get_investment_recommendation
except ImportError as e:
    print(f"⚠️ Import Warning (decision_agent): {e}")
    get_investment_recommendation = None

# AI Algo Trading Bot Integration
try:
    from trading_bot import (
        register_user,
        authenticate_user,
        get_user_bot,
        get_enriched_bot_state,
        reset_user_bot,
        update_bot_config,
        execute_algo_cycle,
        execute_manual_paper_trade,
        close_manual_position,
        get_all_live_prices,
        search_tradingview_instruments,
        get_instrument_live_quote,
        is_indian_market_open,
        is_us_market_open,
    )
    HAS_TRADING_BOT = True
except Exception as e:
    print(f"⚠️ Trading Bot Import Warning: {e}")
    HAS_TRADING_BOT = False
    is_indian_market_open = lambda: False
    is_us_market_open = lambda: True
    get_instrument_live_quote = lambda s, n="": {}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "quickquantpro_secret_key_2026")

MACRO_TICKERS = [
    {"name": "NIFTY 50", "symbol": "^NSEI", "market": "INDIAN"},
    {"name": "BANK NIFTY", "symbol": "^NSEBANK", "market": "INDIAN"},
    {"name": "GOLD (XAU/USD)", "symbol": "GC=F", "market": "US"},
    {"name": "NASDAQ", "symbol": "^IXIC", "market": "US"},
]


DEFAULT_MACRO_DATA = [
    {"name": "NIFTY 50", "symbol": "^NSEI", "price": 25050.25, "change_points": 142.30, "change_pct": 0.57, "is_live": False, "market": "INDIAN"},
    {"name": "BANK NIFTY", "symbol": "^NSEBANK", "price": 54120.80, "change_points": 310.50, "change_pct": 0.58, "is_live": False, "market": "INDIAN"},
    {"name": "GOLD (XAU/USD)", "symbol": "GC=F", "price": 2658.40, "change_points": 12.80, "change_pct": 0.48, "is_live": True, "market": "US"},
    {"name": "NASDAQ", "symbol": "^IXIC", "price": 18120.10, "change_points": -45.20, "change_pct": -0.25, "is_live": True, "market": "US"},
]

_MACRO_CACHE = {"data": list(DEFAULT_MACRO_DATA), "ts": 0}

def _update_macro_in_background():
    """Background worker to fetch macro data without blocking the HTTP request thread."""
    overview = []
    try:
        for item in MACRO_TICKERS:
            is_live = is_indian_market_open() if item["market"] == "INDIAN" else is_us_market_open()
            price = None
            try:
                t = yf.Ticker(item["symbol"])
                fast = getattr(t, "fast_info", None)
                if fast and hasattr(fast, "last_price") and fast.last_price:
                    price = float(fast.last_price)
                    prev = float(getattr(fast, "previous_close", price) or price)
                    pts = price - prev
                    pct = (pts / prev * 100) if prev else 0.0
                    overview.append({
                        "name": item["name"],
                        "symbol": item["symbol"],
                        "price": round(price, 2),
                        "change_points": round(pts, 2),
                        "change_pct": round(pct, 2),
                        "is_live": is_live,
                        "market": item["market"]
                    })
            except Exception:
                pass
            if not price:
                fallback = next((d for d in DEFAULT_MACRO_DATA if d["symbol"] == item["symbol"]), None)
                if fallback:
                    cp = dict(fallback)
                    cp["is_live"] = is_live
                    overview.append(cp)
        if overview:
            _MACRO_CACHE["data"] = overview
            _MACRO_CACHE["ts"] = time.time()
    except Exception:
        pass


def fetch_macro_overview():
    """Instantly returns cached/fallback macro data. Refreshes asynchronously."""
    now = time.time()
    # If cache is older than 60s, trigger a background refresh thread
    if (now - _MACRO_CACHE["ts"]) > 60:
        _MACRO_CACHE["ts"] = now  # Debounce updates
        import threading
        t = threading.Thread(target=_update_macro_in_background, daemon=True)
        t.start()
    return _MACRO_CACHE["data"] or DEFAULT_MACRO_DATA


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = None
    if request.method == "POST":
        ticker = request.form.get("symbol", "").strip().upper()
    elif request.method == "GET" and request.args.get("symbol"):
        ticker = request.args.get("symbol", "").strip().upper()

    macro_data = fetch_macro_overview()
    global_news = fetch_and_prioritize_global_news() if not ticker else []
    trade_analysis = None
    stock_news = []
    error_message = None

    if ticker and get_investment_recommendation:
        try:
            trade_analysis = get_investment_recommendation(ticker)
            stock_news = trade_analysis.get(
                "fundamental_summary", {}
            ).get("stock_news", [])
        except Exception as e:
            error_message = f"Could not complete analysis for '{ticker}'."

    return render_template(
        "index.html",
        ticker=ticker,
        macro_data=macro_data,
        global_news=global_news,
        trade_analysis=trade_analysis,
        stock_news=stock_news,
        error_message=error_message,
    )


@app.route("/api/search_stocks", methods=["GET"])
def api_search_stocks():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = []
    try:
        for quote in yf.Search(query, max_results=8).quotes:
            results.append(
                {
                    "symbol": quote.get("symbol", ""),
                    "name": quote.get("shortname")
                    or quote.get("longname")
                    or quote.get("symbol", ""),
                    "exchange": quote.get("exchange", ""),
                }
            )
    except Exception:
        pass
    return jsonify(results[:8])


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global get_chat_response
    try:
        data = request.get_json() or {}
        user_msg = data.get("message", "").strip()
        ticker_symbol = data.get("symbol", "").strip().upper()
        session_id = data.get("session_id", "default_session")

        if not user_msg:
            return jsonify(
                {"success": True, "response": "Please enter a message."}
            )

        print(f"\n[API /api/chat] Incoming query: '{user_msg}' | Symbol: '{ticker_symbol}'")

        # Dynamic fallback import if not loaded at startup
        if get_chat_response is None:
            try:
                from orchestrator.chatbot import get_chat_response as _lazy_chat
                get_chat_response = _lazy_chat
            except Exception as e:
                print(f"❌ Failed to load chatbot dynamically: {e}")

        if get_chat_response:
            reply = get_chat_response(
                symbol=ticker_symbol, message=user_msg, session_id=session_id
            )
            print(f"[API /api/chat] Responded successfully ({len(reply)} chars)")
            return jsonify({"success": True, "response": reply})
        else:
            return jsonify(
                {"success": False, "response": "⚠️ Chatbot module is temporarily unavailable."}
            )
    except Exception as e:
        print(f"❌ [API /api/chat Exception] {e}")
        return jsonify({"success": False, "response": f"🤖 Error: {str(e)}"})


# ===========================================================================
# AI Algo Trading Bot API Routes
# ===========================================================================

@app.route("/api/bot/auth", methods=["POST"])
def api_bot_auth():
    if not HAS_TRADING_BOT:
        return jsonify({"success": False, "message": "Trading bot module offline."}), 500

    data = request.get_json() or {}
    action = data.get("action", "login")  # 'login' or 'register'
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required."})

    if action == "register":
        res = register_user(username, password)
        if res.get("success"):
            session["user_id"] = res["user_id"]
            session["username"] = res["username"]
        return jsonify(res)
    else:
        res = authenticate_user(username, password)
        if res.get("success"):
            session["user_id"] = res["user_id"]
            session["username"] = res["username"]
        return jsonify(res)


@app.route("/api/bot/logout", methods=["POST"])
def api_bot_logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return jsonify({"success": True, "message": "Logged out successfully."})


@app.route("/api/bot/state", methods=["GET"])
def api_bot_state():
    if not HAS_TRADING_BOT:
        return jsonify({"success": False, "message": "Trading bot module offline."}), 500

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False, "message": "Please log in to access your personal AI Trading Bot."})

    bot_data = get_enriched_bot_state(user_id)
    live_prices = get_all_live_prices()
    return jsonify({
        "authenticated": True,
        "username": session.get("username", "Trader"),
        "bot": bot_data,
        "market_data": live_prices
    })


@app.route("/api/bot/toggle", methods=["POST"])
def api_bot_toggle():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    bot_data = get_user_bot(user_id)
    new_status = "PAUSED" if bot_data["status"] == "RUNNING" else "RUNNING"
    update_bot_config(user_id, status=new_status)
    updated = get_enriched_bot_state(user_id)
    return jsonify({"success": True, "bot": updated})


@app.route("/api/bot/config", methods=["POST"])
def api_bot_config():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    data = request.get_json() or {}
    mode = data.get("mode")
    risk_pct = float(data.get("risk_pct", 1.0)) if "risk_pct" in data else None
    rr_ratio = float(data.get("rr_ratio", 2.0)) if "rr_ratio" in data else None

    update_bot_config(user_id, mode=mode, risk_pct=risk_pct, rr_ratio=rr_ratio)
    updated = get_enriched_bot_state(user_id)
    return jsonify({"success": True, "bot": updated})


@app.route("/api/bot/reset", methods=["POST"])
def api_bot_reset():
    """Resets user's account balance back to 100,000.00 points at any instance."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    reset_user_bot(user_id)
    updated = get_enriched_bot_state(user_id)
    return jsonify({"success": True, "bot": updated, "message": "Account balance reset to 100,000.00."})


@app.route("/api/bot/trigger", methods=["POST"])
def api_bot_trigger():
    """Executes an algorithmic pass and evaluates market conditions."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    updated = execute_algo_cycle(user_id)
    live_prices = get_all_live_prices()
    return jsonify({"success": True, "bot": updated, "market_data": live_prices})


@app.route("/api/bot/manual_trade", methods=["POST"])
def api_bot_manual_trade():
    """Executes manual paper trade entered by trader, updating 100k balance."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip()
    asset_name = data.get("name", symbol)
    direction = data.get("direction", "BUY").upper()
    entry_price = float(data.get("price", 100.0))
    lots = int(data.get("lots", 1))
    lot_size = int(data.get("lot_size", 1))
    quantity = lots * lot_size
    stop_loss = float(data.get("stop_loss", 0.0))
    take_profit = float(data.get("take_profit", 0.0))
    reason = data.get("reason", f"Manual paper execution: {direction} {lots} lot(s) ({quantity} qty) of {asset_name} @ {entry_price}")

    res = execute_manual_paper_trade(
        user_id=user_id,
        symbol=symbol,
        asset_name=asset_name,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reason=reason
    )
    return jsonify(res)


@app.route("/api/bot/close_trade", methods=["POST"])
def api_bot_close_trade():
    """Closes an active position manually."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Not authenticated."}), 401

    data = request.get_json() or {}
    trade_id = data.get("trade_id")
    if not trade_id:
        return jsonify({"success": False, "message": "trade_id required."})

    res = close_manual_position(user_id, int(trade_id))
    return jsonify(res)


@app.route("/api/bot/search_instruments", methods=["GET"])
def api_bot_search_instruments():
    """TradingView-style multi-asset search (Stocks, Options CE/PE, Forex, Gold, Futures)."""
    q = request.args.get("q", "").strip()
    results = search_tradingview_instruments(q)
    return jsonify(results)


@app.route("/api/bot/market_data", methods=["GET"])
def api_bot_market_data():
    prices = get_all_live_prices()
    return jsonify(prices)


@app.route("/api/macro_overview", methods=["GET"])
def api_macro_overview():
    """Returns live global macro overview with Live/Closed status for auto-updating landing ribbon."""
    return jsonify(fetch_macro_overview())


@app.route("/api/bot/quote", methods=["GET"])
def api_bot_quote():
    """Returns real-time spot price, contract LTP, % change, and Live/Closed status for any instrument."""
    symbol = request.args.get("symbol", "").strip()
    name = request.args.get("name", "").strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    quote = get_instrument_live_quote(symbol, name)
    return jsonify(quote)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)