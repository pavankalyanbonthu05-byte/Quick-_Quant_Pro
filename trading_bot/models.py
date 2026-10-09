import os
import sqlite3
import hashlib
import time
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bot_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        balance REAL DEFAULT 100000.0,
        initial_capital REAL DEFAULT 100000.0,
        status TEXT DEFAULT 'PAUSED', -- 'RUNNING' or 'PAUSED'
        mode TEXT DEFAULT 'SCALPING', -- 'SCALPING' or 'INTRADAY'
        risk_pct REAL DEFAULT 1.0,     -- risk 1% per trade
        rr_ratio REAL DEFAULT 2.0,     -- 1:2 Risk to Reward
        assets_active TEXT DEFAULT 'GOLD,NASDAQ,NIFTY,RELIANCE',
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        asset_name TEXT NOT NULL,
        direction TEXT NOT NULL, -- 'BUY' or 'SELL'
        entry_price REAL NOT NULL,
        quantity REAL NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        exit_price REAL,
        pnl REAL DEFAULT 0.0,
        status TEXT DEFAULT 'OPEN', -- 'OPEN', 'TARGET_HIT', 'STOPPED_OUT', 'MANUAL_CLOSE'
        reason TEXT NOT NULL,       -- Detailed reasoning why algo took this trade
        opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()


def register_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        return {"success": False, "message": "Username and password required."}

    pwd_hash = hash_password(password)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
        user_id = cursor.lastrowid

        # Initialize personal bot with 100k balance
        cursor.execute("""
            INSERT INTO bot_configs (user_id, balance, initial_capital, status, mode, risk_pct, rr_ratio)
            VALUES (?, 100000.0, 100000.0, 'PAUSED', 'SCALPING', 1.0, 2.0)
        """, (user_id,))

        conn.commit()
        return {"success": True, "user_id": user_id, "username": username}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Username already exists. Please login."}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    username = username.strip()
    pwd_hash = hash_password(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"success": True, "user_id": row["id"], "username": row["username"]}
    return {"success": False, "message": "Invalid credentials."}


def get_user_bot(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bot_configs WHERE user_id = ?", (user_id,))
    config = cursor.fetchone()

    if not config:
        # Create default if missing
        cursor.execute("""
            INSERT INTO bot_configs (user_id, balance, initial_capital, status, mode, risk_pct, rr_ratio)
            VALUES (?, 100000.0, 100000.0, 'PAUSED', 'SCALPING', 1.0, 2.0)
        """, (user_id,))
        conn.commit()
        cursor.execute("SELECT * FROM bot_configs WHERE user_id = ?", (user_id,))
        config = cursor.fetchone()

    cursor.execute("SELECT * FROM trades WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user_id,))
    trades = [dict(t) for t in cursor.fetchall()]

    # Calculate live open positions P&L and total closed P&L
    closed_pnl = sum([t["pnl"] for t in trades if t["status"] not in ("OPEN", "PENDING")])
    open_trades = [t for t in trades if t["status"] in ("OPEN", "PENDING")]

    conn.close()
    return {
        "user_id": user_id,
        "balance": round(config["balance"], 2),
        "initial_capital": round(config["initial_capital"], 2),
        "status": config["status"],
        "mode": config["mode"],
        "risk_pct": config["risk_pct"],
        "rr_ratio": config["rr_ratio"],
        "assets_active": config["assets_active"],
        "closed_pnl": round(closed_pnl, 2),
        "open_trades": open_trades,
        "all_trades": trades
    }


def reset_user_bot(user_id: int):
    """Resets the user's paper balance back to 100,000.00 and clears/closes active positions."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Reset balance and initial capital to 100,000.00
    cursor.execute("""
        UPDATE bot_configs 
        SET balance = 100000.0, initial_capital = 100000.0, last_updated = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (user_id,))

    # Close any open trades with 0 PnL
    cursor.execute("""
        UPDATE trades 
        SET status = 'MANUAL_CLOSE', closed_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND status = 'OPEN'
    """, (user_id,))

    conn.commit()
    conn.close()
    return get_user_bot(user_id)


def update_bot_config(user_id: int, status: str = None, mode: str = None, risk_pct: float = None, rr_ratio: float = None):
    conn = get_db_connection()
    cursor = conn.cursor()

    updates = []
    params = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if mode is not None:
        updates.append("mode = ?")
        params.append(mode)
    if risk_pct is not None:
        updates.append("risk_pct = ?")
        params.append(risk_pct)
    if rr_ratio is not None:
        updates.append("rr_ratio = ?")
        params.append(rr_ratio)

    if updates:
        updates.append("last_updated = CURRENT_TIMESTAMP")
        query = f"UPDATE bot_configs SET {', '.join(updates)} WHERE user_id = ?"
        params.append(user_id)
        cursor.execute(query, tuple(params))
        conn.commit()

    conn.close()
    return get_user_bot(user_id)


# Auto-initialize database on load
init_db()

