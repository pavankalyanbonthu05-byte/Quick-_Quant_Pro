import json
from quant_worker import predict_stock_trend

if __name__ == "__main__":
    test_symbol = "NVDA"
    print(f"🚀 Executing Worker 1 (Quant LSTM Tool) for {test_symbol}...\n")

    result = predict_stock_trend(test_symbol, days=30)

    print("--- Worker 1 JSON Output Payload ---")
    print(json.dumps(result, indent=2))