import os
import re
import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ---------------------------------------------------------------------------
# Institutional system prompt for Gemini-grade structured financial insights
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Robo — an elite quantitative financial intelligence assistant embedded \
inside an institutional trading platform. You deliver responses that are \
beautifully structured, insightful, and polished — exactly like Google Gemini \
or Bloomberg Intelligence.

Formatting Guidelines:
1. Start with a clear, bold **Sentiment Verdict** or headline answer (e.g. **Status: Neutral to Mildly Bearish (Short-Term)**).
2. Structure your analysis using clean markdown sections with ### headers:
   - ### 1. Sentiment Verdict
   - ### 2. Key Catalysts & Technical Signals (with bullet points for catalysts, technicals, flow)
   - ### 3. Strategic Outlook / Key Levels (support, resistance, actionable setups)
3. Use **bolding** strategically on key price levels, metrics, percentages, and signals.
4. Use markdown tables where appropriate to summarize metrics, key levels, or scenarios.
5. Provide actionable insights for traders/investors (Key Support, Key Resistance, Strategy).
6. Maintain an institutional, analytical tone — objective, crisp, and direct.
7. Do NOT use dollar ($) signs for Indian equities (use INR, ₹, or plain numbers).
8. Never output raw code or JSON unless explicitly requested.\
"""

PRIMARY_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "openai/gpt-oss-120b"


def _fetch_market_context(symbol: str) -> str:
    """Fetch live spot price and recent performance for context injection."""
    if not symbol:
        return "Scope: Global Market Query"

    try:
        import urllib.request, json
        c_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        c_req = urllib.request.Request(c_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(c_req, timeout=1.5) as c_res:
            meta = json.loads(c_res.read().decode("utf-8")).get("chart", {}).get("result", [{}])[0].get("meta", {})
            last_price = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose", last_price) or last_price

            if last_price:
                last_price = float(last_price)
                prev_close = float(prev_close) if prev_close else last_price
                chg_pct = round(((last_price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
                direction = "+" if chg_pct >= 0 else ""
                currency = "INR" if (".NS" in symbol or ".BO" in symbol) else ""
                return (
                    f"Selected Ticker: {symbol} | Spot Price: {round(float(last_price), 2)} {currency} | "
                    f"Day Change: {direction}{chg_pct}%"
                )
    except Exception as e:
        print(f"⚠️ [Robo] Context warning for {symbol}: {e}")

    return f"Selected Ticker: {symbol}"


def get_chat_response(
    symbol: str = "", message: str = "", session_id: str = "default"
) -> str:
    """Core AI chat function — returns polished, markdown-formatted financial intelligence."""
    if not message:
        return "Hello! How can I assist you with stock or market analysis today?"

    api_key = os.getenv(
        "GROQ_API_KEY",
        ""
    )
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing from configuration."

    if not HAS_GROQ:
        return "⚠️ Groq SDK is not installed (`pip install groq`)."

    context_text = _fetch_market_context(symbol)
    print(f"\n🤖 [Robo] Query: '{message[:80]}' | Context: {context_text}")

    try:
        client = Groq(api_key=api_key, timeout=5.0)
        user_content = f"Market Context: {context_text}\n\nUser Question: {message}"

        # Attempt with Primary model (qwen/qwen3.8-27b), fallback to gpt-oss-120b
        completion = None
        for model_choice in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                print(f"⚡ [Robo] Generating response via {model_choice}...")
                completion = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                if completion and completion.choices:
                    break
            except Exception as model_err:
                print(f"⚠️ [Robo] Error with {model_choice}: {model_err}. Trying fallback...")

        if not completion or not completion.choices:
            return "🤖 I am currently experiencing elevated load. Please try again in a few moments."

        raw_reply = completion.choices[0].message.content or ""
        # Clean out any <think> tags if present
        clean_reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()
        final_reply = clean_reply if clean_reply else raw_reply.strip()

        print(f"✅ [Robo] Response generated ({len(final_reply)} chars).")
        return final_reply

    except Exception as e:
        print(f"❌ [Robo Error] {e}")
        return f"🤖 I encountered an issue connecting to the AI model: {str(e)}"


if __name__ == "__main__":
    # Test execution
    res = get_chat_response(
        symbol="RELIANCE.NS", message="is reliance tending to sell side sentiment"
    )
    print("\n" + res)