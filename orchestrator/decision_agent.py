import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fundamental_worker.rag_search import run_fundamental_agent
from quant_worker.predictor import run_quant_analysis

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


def calculate_trade_plan(quant_data: dict, fund_data: dict) -> dict:
    spot = quant_data.get("current_price", 100.0)
    forecast_return = quant_data.get("forecasted_return_pct", 0.0)
    rsi = quant_data.get("technical_indicators", {}).get("rsi_14", 50.0)
    volatility = quant_data.get("technical_indicators", {}).get(
        "volatility_20d", 0.02
    )

    if forecast_return >= 1.5 and rsi < 70:
        signal = "BUY"
        win_prob = min(
            88.0, 65.0 + (forecast_return * 2.5) + ((70 - rsi) * 0.2)
        )
    elif forecast_return <= -1.5 and rsi > 30:
        signal = "SELL"
        win_prob = min(
            85.0, 60.0 + (abs(forecast_return) * 2.5) + ((rsi - 30) * 0.2)
        )
    else:
        signal = "NO CALL"
        win_prob = 50.0

    atr_approx = spot * volatility * 1.5
    stop_loss = (
        round(spot - (atr_approx * 1.2), 2)
        if signal == "BUY"
        else round(spot + (atr_approx * 1.2), 2)
    )

    targets = []
    if signal in ["BUY", "SELL"]:
        direction = 1 if signal == "BUY" else -1
        t_configs = [
            (1, "High (80-90%)", round(atr_approx * 1.0, 2)),
            (2, "Medium (65-75%)", round(atr_approx * 2.0, 2)),
            (3, "Moderate (50-60%)", round(atr_approx * 3.5, 2)),
            (4, "High Risk (35-45%)", round(atr_approx * 5.0, 2)),
        ]
        for num, tier, pts in t_configs:
            targets.append(
                {
                    "target_number": num,
                    "probability_tier": tier,
                    "target_price": round(spot + (direction * pts), 2),
                    "points_gain": pts,
                    "return_pct": round((pts / spot) * 100, 2),
                }
            )

    return {
        "signal": signal,
        "win_probability_pct": round(win_prob, 1),
        "stop_loss": stop_loss if signal != "NO CALL" else None,
        "risk_reward_ratio": (
            round(
                targets[0]["points_gain"] / (abs(spot - stop_loss) + 1e-9), 2
            )
            if targets
            else 1.0
        ),
        "breakeven_range": (
            round(spot * 0.005, 2) if signal == "NO CALL" else None
        ),
        "targets": targets,
        "rationale": "",
    }


def get_investment_recommendation(symbol: str) -> dict:
    quant_data = run_quant_analysis(symbol)
    fund_data = run_fundamental_agent(symbol)
    trade_plan = calculate_trade_plan(quant_data, fund_data)

    api_key = os.getenv("GROQ_API_KEY","")
    prompt = (
        f"You are a quantitative trade strategist. Analyze {symbol}: "
        f"Spot {quant_data.get('current_price')}, Forecast Return {quant_data.get('forecasted_return_pct')}%, "
        f"RSI {quant_data.get('technical_indicators', {}).get('rsi_14')}. "
        f"Trade Signal: {trade_plan['signal']}. Write 2-3 concise sentences explaining the rationale behind this call. "
        f"Do not use dollar ($) signs."
    )

    if HAS_GROQ and api_key:
        try:
            client = Groq(api_key=api_key, timeout=3.5)
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            trade_plan["rationale"] = resp.choices[0].message.content.strip()
        except Exception as e:
            trade_plan["rationale"] = (
                f"Trade call based on 30D forecast ({quant_data.get('forecasted_return_pct')}%) "
                f"and RSI ({quant_data.get('technical_indicators', {}).get('rsi_14')})."
            )
    else:
        trade_plan["rationale"] = (
            f"Trade call based on 30D forecast ({quant_data.get('forecasted_return_pct')}%) "
            f"and RSI ({quant_data.get('technical_indicators', {}).get('rsi_14')})."
        )

    return {
        "symbol": symbol,
        "quant_summary": quant_data,
        "fundamental_summary": fund_data,
        "trade_plan": trade_plan,
    }


if __name__ == "__main__":
    # Quick execution test
    res = get_investment_recommendation("AAPL")
    print(res["trade_plan"])