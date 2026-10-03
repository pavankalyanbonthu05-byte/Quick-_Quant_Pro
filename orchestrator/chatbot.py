import os
from typing import Dict, List

from .decision_agent import get_investment_recommendation

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Session Memory Store (In-memory storage for conversation history)
# Format: { "session_id": [ {"role": "user"/"assistant", "content": "..."}, ... ] }
CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


def get_chat_response(
    symbol: str, message: str, session_id: str = "default_session"
) -> str:
    """Financial AI Assistant Chat Handler with Conversational Memory &

    Multi-Agent Context Grounding.
    """
    symbol = symbol.strip().upper()
    api_key = os.getenv("GROQ_API_KEY")

    # 1. Initialize or fetch session history
    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = []
    session_history = CHAT_SESSIONS[session_id]

    # 2. Fetch live stock analysis payload from Orchestrator
    try:
        analysis = get_investment_recommendation(symbol)
        trade_plan = analysis.get("trade_plan", {})
        quant = analysis.get("quant_summary", {})
        fundamental = analysis.get("fundamental_summary", {})
        company_profile = (
            fundamental.get("company_profile", {}).get("company_origin", "")
        )
    except Exception as e:
        print(f"Error gathering stock context for chatbot: {e}")
        analysis, trade_plan, quant, fundamental, company_profile = (
            {},
            {},
            {},
            {},
            "",
        )

    # 3. System Prompt Grounding the AI in calculated facts
    system_prompt = f"""
You are an expert Financial AI Analyst on the Agentic Quant & Financial Intelligence Platform. You are answering questions about the stock: {symbol}.

STRICT DATA CONTEXT:
- Spot Price: ${quant.get('current_price', 'N/A')}
- Signal: {trade_plan.get('signal', 'N/A')} (Win Probability: {trade_plan.get('win_probability_pct', 'N/A')}%)
- Stop Loss: ${trade_plan.get('stop_loss', 'N/A')}
- Segmented Targets: {trade_plan.get('targets', [])}
- Breakeven Range: {trade_plan.get('breakeven_range', 'N/A')}
- Technical Indicators (RSI, MAs): {quant.get('technical_indicators', {})}
- Financials (YoY Growth, Net Debt, PE): {fundamental.get('financials', {})}
- Shareholding (FII, DII, Promoter): {fundamental.get('shareholding', {})}
- RAG Company Context: {company_profile[:400]}...

RULES:
1. Answer concisely, professionally, and ground all claims in the data provided above.
2. Maintain conversation continuity using prior messages in the chat history.
3. Do not invent ungrounded financial figures.
"""

    # Fallback if Groq is unavailable
    if not HAS_GROQ or not api_key:
        first_target = (
            trade_plan.get("targets", [{}])[0].get("target_price", "N/A")
            if trade_plan.get("targets")
            else "N/A"
        )
        fallback_msg = (
            f"Bot Response ({symbol}): Signal is {trade_plan.get('signal')} with"
            f" Target 1 at ${first_target}. (Configure GROQ_API_KEY in .env for"
            " interactive AI chat)."
        )
        session_history.append({"role": "user", "content": message})
        session_history.append({"role": "assistant", "content": fallback_msg})
        return fallback_msg

    try:
        client = Groq(api_key=api_key)

        # Build prompt stack: System Prompt + Rolling Conversation History (Last 10 turns) + New Message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session_history[-10:])
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        reply = response.choices[0].message.content.strip()

        # Update session memory
        session_history.append({"role": "user", "content": message})
        session_history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print(f"Chatbot API Error: {e}")
        return f"I encountered an error processing your query for {symbol}: {str(e)}"