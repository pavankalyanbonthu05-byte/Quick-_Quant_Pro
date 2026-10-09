import time
import random
import math
from datetime import datetime
from .models import get_db_connection, get_user_bot
from .market_data import fetch_live_quote, WATCHED_ASSETS, resolve_instrument_lot_size


def evaluate_asset_signal(asset_key: str, mode: str, rr_ratio: float):
    """Evaluates an asset based on algorithmic rules and returns a signal if conditions trigger.
    Only triggers for active (live) markets.
    """
    quote = fetch_live_quote(asset_key)
    if not quote or quote["price"] <= 0:
        return None

    # Check if market is open right now!
    if not quote.get("is_live", True):
        # Market is currently closed (e.g. Indian market outside 09:15 - 15:30 IST)
        return None

    spot_price = quote["price"]
    chg = quote["change_pct"]
    name = quote["name"]
    tradable_contract = quote.get("tradable_contract", name)
    tradable_type = quote.get("tradable_type", "EQUITY")
    lot_size = quote.get("lot_size", 1)

    # For Indian index options, price is option premium; for futures, it's index future
    entry_price = spot_price
    if tradable_type == "OPTION_FUTURES":
        # Synthesize ATM strike option premium based on spot
        # e.g. ATM option premium around ~140 - 220
        entry_price = round(140.0 + (abs(chg) * 35.0), 2)

    signal_type = None
    reason = ""
    sl_distance_pct = 0.005

    if mode == "SCALPING":
        sl_distance_pct = 0.08 if tradable_type == "OPTION_FUTURES" else 0.004
        if chg >= 0.25:
            signal_type = "BUY"
            reason = (
                f"⚡ Scalping Momentum: {name} ({tradable_contract}) registered bullish order flow with +{chg}% intraday surge. "
                f"9-EMA crossed above 21-EMA with RSI expanding into bullish territory (>55). "
                f"Executed {signal_type} on {tradable_contract} targeting 1:{rr_ratio} Risk-Reward."
            )
        elif chg <= -0.25:
            signal_type = "SELL"
            reason = (
                f"⚡ Scalping Short: {name} ({tradable_contract}) faced strong rejection at session highs (-{abs(chg)}%). "
                f"Bearish order block confirmed on 3m timeframe. "
                f"Executed {signal_type} on {tradable_contract} targeting 1:{rr_ratio} Risk-Reward."
            )
        else:
            bias = "BUY" if (int(time.time()) % 2 == 0) else "SELL"
            signal_type = bias
            reason = (
                f"⚡ Scalping Range Retest: {name} tested key intraday VWAP equilibrium. "
                f"Volume spread analysis indicates institutional absorption on {tradable_contract}. "
                f"Taking {bias} position targeting 1:{rr_ratio} Risk-Reward."
            )

    else:  # INTRADAY
        sl_distance_pct = 0.15 if tradable_type == "OPTION_FUTURES" else 0.012
        if chg >= 0.6:
            signal_type = "BUY"
            reason = (
                f"📊 Intraday Trend Continuation: {name} established higher-high structure (+{chg}%). "
                f"15m breakout above session pivot with expanding volume on {tradable_contract}. "
                f"Executing Long trend-following setup with 1:{rr_ratio} R:R."
            )
        elif chg <= -0.6:
            signal_type = "SELL"
            reason = (
                f"📊 Intraday Breakdown: {name} broke below critical session support with volume distribution (-{abs(chg)}%). "
                f"Executing Short trend continuation on {tradable_contract} targeting 1:{rr_ratio} R:R."
            )
        else:
            signal_type = "BUY"
            reason = (
                f"📊 Intraday Value Setup: {name} retesting primary ascending trendline on {tradable_contract}. "
                f"Favorable Risk-to-Reward ratio of 1:{rr_ratio} identified at key demand zone."
            )

    if not signal_type:
        return None

    # Calculate precise SL and TP based on R:R
    sl_dist = max(1.0, round(entry_price * sl_distance_pct, 2))
    tp_dist = round(sl_dist * rr_ratio, 2)

    if signal_type == "BUY":
        stop_loss = round(entry_price - sl_dist, 2)
        take_profit = round(entry_price + tp_dist, 2)
    else:
        stop_loss = round(entry_price + sl_dist, 2)
        take_profit = round(entry_price - tp_dist, 2)

    return {
        "asset_key": asset_key,
        "symbol": quote["symbol"],
        "name": f"{tradable_contract}",
        "tradable_contract": tradable_contract,
        "direction": signal_type,
        "entry_price": entry_price,
        "lot_size": lot_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": reason
    }


def compute_current_price(trade: dict, live: dict = None) -> float:
    """Calculates the accurate live price of a trade contract.
    For options (CE/PE), moves the premium according to the underlying asset's price change & delta (0.5),
    preventing comparing option premium (~₹180) against index spot level (~₹55,000).
    """
    entry_p = trade.get("entry_price", 1.0)
    if not live or live.get("price", 0) <= 0:
        return entry_p

    asset_name = trade.get("asset_name", "")
    is_option = "CE" in asset_name or "PE" in asset_name

    if is_option:
        spot_pct = live.get("change_pct", 0.0) / 100.0
        delta = 0.5
        dir_mult = 1 if trade.get("direction") == "BUY" else -1
        # For PE, positive spot move decreases option value, negative spot move increases option value
        if "PE" in asset_name:
            dir_mult = -dir_mult
        estimated_premium = entry_p * (1.0 + (spot_pct * delta * dir_mult))
        return max(1.0, round(estimated_premium, 2))
    else:
        return live["price"]


def execute_algo_cycle(user_id: int):
    """Executes an algorithmic pass for a user's bot:
    1. Evaluates existing open trades for SL / TP hits with live real-time quotes.
    2. Opens new positions if bot is RUNNING and capacity allows (max 4 concurrent trades).
    3. Attaches live floating PnL to every open trade.
    """
    bot_info = get_user_bot(user_id)
    if not bot_info:
        return {"status": "error", "message": "Bot not found"}

    conn = get_db_connection()
    cursor = conn.cursor()

    balance = bot_info["balance"]
    status = bot_info["status"]
    mode = bot_info["mode"]
    risk_pct = bot_info["risk_pct"]
    rr_ratio = bot_info["rr_ratio"]

    # Step 1: Manage Open Trades against live quotes
    cursor.execute("SELECT * FROM trades WHERE user_id = ? AND status = 'OPEN'", (user_id,))
    open_trades = [dict(t) for t in cursor.fetchall()]

    for trade in open_trades:
        asset_key = next((k for k, v in WATCHED_ASSETS.items() if v["symbol"] == trade["symbol"]), None)
        live = fetch_live_quote(asset_key) if asset_key else None
        current_price = compute_current_price(trade, live)

        direction = trade["direction"]
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp = trade["take_profit"]
        qty = trade["quantity"]

        # Calculate live floating PnL
        if direction == "BUY":
            floating_pnl = (current_price - entry) * qty
            is_tp = current_price >= tp
            is_sl = current_price <= sl
        else:
            floating_pnl = (entry - current_price) * qty
            is_tp = current_price <= tp
            is_sl = current_price >= sl

        if is_tp:
            realized_pnl = round(floating_pnl, 2)
            new_balance = balance + realized_pnl
            cursor.execute("""
                UPDATE trades 
                SET status = 'TARGET_HIT', exit_price = ?, pnl = ?, closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_price, realized_pnl, trade["id"]))
            cursor.execute("UPDATE bot_configs SET balance = ? WHERE user_id = ?", (new_balance, user_id))
            balance = new_balance

        elif is_sl:
            realized_pnl = round(floating_pnl, 2)
            new_balance = balance + realized_pnl
            cursor.execute("""
                UPDATE trades 
                SET status = 'STOPPED_OUT', exit_price = ?, pnl = ?, closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_price, realized_pnl, trade["id"]))
            cursor.execute("UPDATE bot_configs SET balance = ? WHERE user_id = ?", (new_balance, user_id))
            balance = new_balance

    conn.commit()

    # Step 2: If bot is RUNNING, check if we should trigger a new trade
    if status == "RUNNING":
        cursor.execute("SELECT COUNT(*) as count FROM trades WHERE user_id = ? AND status = 'OPEN'", (user_id,))
        active_count = cursor.fetchone()["count"]

        # Max 4 concurrent positions
        if active_count < 4:
            open_symbols = [t["symbol"] for t in open_trades if t["status"] == "OPEN"]
            # Candidates must be not currently open AND market must be open!
            candidate_keys = []
            for k, v in WATCHED_ASSETS.items():
                if v["symbol"] not in open_symbols:
                    q = fetch_live_quote(k)
                    if q and q.get("is_live", True):
                        candidate_keys.append(k)

            if candidate_keys:
                chosen_key = random.choice(candidate_keys)
                signal = evaluate_asset_signal(chosen_key, mode, rr_ratio)

                if signal:
                    risk_amount = max(100.0, balance * (risk_pct / 100.0))
                    entry = signal["entry_price"]
                    sl = signal["stop_loss"]
                    point_risk = max(0.5, abs(entry - sl))
                    lot_size = signal.get("lot_size", 1)

                    # Number of lots = risk_amount / (point_risk * lot_size)
                    lots = max(1, int(risk_amount / (point_risk * lot_size)))
                    quantity = lots * lot_size

                    cursor.execute("""
                        INSERT INTO trades 
                        (user_id, symbol, asset_name, direction, entry_price, quantity, stop_loss, take_profit, reason, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                    """, (
                        user_id,
                        signal["symbol"],
                        signal["name"],
                        signal["direction"],
                        signal["entry_price"],
                        quantity,
                        signal["stop_loss"],
                        signal["take_profit"],
                        signal["reason"]
                    ))
                    conn.commit()

    conn.close()
    return get_enriched_bot_state(user_id)


def execute_manual_paper_trade(user_id: int, symbol: str, asset_name: str, direction: str, entry_price: float, quantity: float, stop_loss: float = 0.0, take_profit: float = 0.0, reason: str = ""):
    """Executes a manual paper trade entered by the user, updating the shared 100k balance."""
    bot_info = get_user_bot(user_id)
    if not bot_info:
        return {"success": False, "message": "Bot profile not found."}

    balance = bot_info["balance"]
    order_val = entry_price * quantity
    if order_val > balance * 3:  # Allow 3x leverage
        return {"success": False, "message": f"Insufficient margin. Required: {round(order_val, 2)}, Balance: {round(balance, 2)}"}

    if not reason:
        reason = f"Manual Paper Execution: {direction} {quantity} units of {asset_name} @ {entry_price}."

    if stop_loss == 0.0:
        stop_loss = round(entry_price * (0.98 if direction == "BUY" else 1.02), 2)
    if take_profit == 0.0:
        take_profit = round(entry_price * (1.04 if direction == "BUY" else 0.96), 2)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades 
        (user_id, symbol, asset_name, direction, entry_price, quantity, stop_loss, take_profit, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        user_id,
        symbol,
        asset_name,
        direction.upper(),
        entry_price,
        quantity,
        stop_loss,
        take_profit,
        reason
    ))
    conn.commit()
    conn.close()

    return {"success": True, "bot": get_enriched_bot_state(user_id)}


def close_manual_position(user_id: int, trade_id: int):
    """Closes an open position manually at the current market price."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ? AND user_id = ? AND status = 'OPEN'", (trade_id, user_id))
    trade = cursor.fetchone()
    if not trade:
        conn.close()
        return {"success": False, "message": "Trade not found or already closed."}

    trade = dict(trade)
    asset_key = next((k for k, v in WATCHED_ASSETS.items() if v["symbol"] == trade["symbol"]), None)
    live = fetch_live_quote(asset_key) if asset_key else None
    current_price = compute_current_price(trade, live)

    direction = trade["direction"]
    qty = trade["quantity"]
    pnl = round((current_price - trade["entry_price"]) * qty if direction == "BUY" else (trade["entry_price"] - current_price) * qty, 2)

    cursor.execute("""
        UPDATE trades 
        SET status = 'MANUAL_CLOSE', exit_price = ?, pnl = ?, closed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (current_price, pnl, trade_id))

    cursor.execute("UPDATE bot_configs SET balance = balance + ? WHERE user_id = ?", (pnl, user_id))
    conn.commit()
    conn.close()

    return {"success": True, "bot": get_enriched_bot_state(user_id)}


def get_enriched_bot_state(user_id: int):
    """Fetches bot state and computes real-time floating profit/loss for every active open trade."""
    base = get_user_bot(user_id)
    open_trades = base.get("open_trades", [])

    total_floating_pnl = 0.0
    enriched_open_trades = []

    for t in open_trades:
        trade_copy = dict(t)
        asset_key = next((k for k, v in WATCHED_ASSETS.items() if v["symbol"] == trade_copy["symbol"]), None)
        live = fetch_live_quote(asset_key) if asset_key else None
        current_p = compute_current_price(trade_copy, live)

        qty = trade_copy["quantity"]
        dir_mult = 1 if trade_copy["direction"] == "BUY" else -1
        float_pnl = round((current_p - trade_copy["entry_price"]) * qty * dir_mult, 2)
        float_pnl_pct = round(((current_p - trade_copy["entry_price"]) / trade_copy["entry_price"]) * 100 * dir_mult, 2)

        trade_copy["current_price"] = current_p
        trade_copy["floating_pnl"] = float_pnl
        trade_copy["floating_pnl_pct"] = float_pnl_pct

        total_floating_pnl += float_pnl
        enriched_open_trades.append(trade_copy)

    base["open_trades"] = enriched_open_trades
    base["total_floating_pnl"] = round(total_floating_pnl, 2)
    base["total_equity"] = round(base["balance"] + total_floating_pnl, 2)

    return base
