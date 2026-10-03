import json
from fundamental_worker import analyze_fundamentals

if __name__ == "__main__":
    symbol = "NVDA"
    print(
        f"\n🚀 Running Worker 2 (LangGraph Fundamental & RAG Agent) for"
        f" {symbol}...\n"
    )

    results = analyze_fundamentals(symbol)

    print("--- LangGraph Output State Payload ---")
    print(json.dumps(results, indent=2))