import os
from .risk_calculator import calculate_segmented_trade_plan
from fundamental_worker import analyze_fundamentals
from quant_worker import predict_stock_trend

# Optional Groq client import
try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


def generate_llm_trade_rationale(
    symbol: str,
    current_price: float,
    trade_plan: dict,
    quant_signal: dict,
    fundamental_signal: dict,
) -> str:
    """Uses Groq Llama-3.3-70B to synthesize a professional trade rationale from

    Python metrics.
    """
    api_key = os.getenv("GROQ_API_KEY")

    # Fallback if no API key or groq library is not installed
    if not HAS_GROQ or not api_key:
        first_target = (
            trade_plan["targets"][0]["target_price"]
            if trade_plan["targets"]
            else "N/A"
        )
        rev_growth = fundamental_signal.get("financials", {}).get(
            "revenue_yoy_pct", 0
        )
        return (
            f"Automated Analysis for {symbol}: Trade signal is"
            f" {trade_plan['signal']} with a winning probability of"
            f" {trade_plan['win_probability_pct']}%. Target 1 is set at"
            f" ${first_target} with YoY Revenue growth at {rev_growth}%."
        )

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
You are an institutional quant risk analyst. Synthesize a concise 2-3 sentence trade rationale based on these exact python-calculated metrics:
- Ticker Symbol: {symbol}
- Current Spot Price: ${current_price}
- Calculated Signal: {trade_plan['signal']}
- Win Probability: {trade_plan['win_probability_pct']}%
- Segmented Targets: {trade_plan['targets']}
- Stop Loss: ${trade_plan.get('stop_loss')}
- Breakeven Range: {trade_plan.get('breakeven_range')}
- Technical RSI (14): {quant_signal.get('technical_indicators', {}).get('rsi_14')}
- YoY Revenue Growth: {fundamental_signal.get('financials', {}).get('revenue_yoy_pct')}%
- Business Profile: {fundamental_signal.get('company_profile', {}).get('business_summary')}

Explain concisely why this setup (targets & stop loss or breakeven) is justified based on the technicals and fundamentals.
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM API Call Exception: {e}")
        return trade_plan.get("rationale", "Rationale generation unavailable.")


def get_investment_recommendation(symbol: str) -> dict:
    """Main Orchestrator Entrypoint: Combines Worker 1 (Quant) and Worker 2

    (Fundamentals), calculates segmented targets/stop loss, and passes
    everything to the LLM to generate the final trade rationale.
    """
    symbol = symbol.strip().upper()

    # 1. Fetch Worker 1 (Quant LSTM Signal)
    quant_signal = predict_stock_trend(symbol, days=30)
    current_price = quant_signal.get("current_price", 0.0)
    forecast_trajectory = quant_signal.get("forecast_trajectory", [])
    tech = quant_signal.get("technical_indicators", {})
    volatility = tech.get("volatility_20d", 0.015)

    # 2. Fetch Worker 2 (Fundamental & RAG Signal)
    fundamental_signal = analyze_fundamentals(symbol)
    financials = fundamental_signal.get("financials", {})
    rev_growth = financials.get("revenue_yoy_pct", 0.0)

    # 3. Evaluate Win Probability Score
    rsi = tech.get("rsi_14", 50.0)
    ret_pct = quant_signal.get("forecasted_return_pct", 0.0)

    base_confidence = 50.0
    if ret_pct > 3.0:
        base_confidence += 20.0
    elif ret_pct < -3.0:
        base_confidence -= 20.0

    if rev_growth > 5.0:
        base_confidence += 15.0
    elif rev_growth < -5.0:
        base_confidence -= 15.0

    if 30 <= rsi <= 65:
        base_confidence += 10.0

    win_probability = min(92.0, max(35.0, round(base_confidence, 1)))

    # 4. Calculate Segmented Targets & Stop Loss (Python Math Engine)
    trade_plan = calculate_segmented_trade_plan(
        current_price=current_price,
        forecast_trajectory=forecast_trajectory,
        volatility=volatility,
        win_probability=win_probability,
    )

    # 5. Connect LLM for Rationale Synthesis
    llm_rationale = generate_llm_trade_rationale(
        symbol=symbol,
        current_price=current_price,
        trade_plan=trade_plan,
        quant_signal=quant_signal,
        fundamental_signal=fundamental_signal,
    )

    # Attach LLM rationale
    trade_plan["rationale"] = llm_rationale

    return {
        "symbol": symbol,
        "quant_summary": quant_signal,
        "fundamental_summary": fundamental_signal,
        "trade_plan": trade_plan,
    }