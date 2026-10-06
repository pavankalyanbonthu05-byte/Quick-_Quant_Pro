# Quick Quant Pro 📈🤖
### Quant & Financial Intelligence Platform

**Quick Quant Pro** is an autonomous financial intelligence and algorithmic paper trading platform. It unites neural quant forecasting, fundamental analysis, agentic vector RAG retrieval, and an autonomous AI trading bot supporting US and Indian markets.

---

## ⚡ Key Features

- **Autonomous AI Algo Trading Bot**:
  - Personal user accounts starting with **100,000 paper capital** (resettable anytime).
  - Scalping & intraday strategies with configurable risk-to-reward ratios and risk percentages.
  - Multi-asset support: **XAU/USD Gold, Mini/Micro NASDAQ Futures, NIFTY 50 & BANKNIFTY Futures/Options (CE/PE)** with exchange-standard lot sizing.
  - Detailed trade log capturing entry, target, stop loss, and the exact quant rationale behind every order.

- **Neural Quant Engine**:
  - 30-day neural forecast trajectory seamlessly connected to historical spot prices.
  - 20 MA and 200 MA trend lines.
  - Interactive dual-view chart: toggle between **Line** and **Candlestick** views.
  - Sub-5-second auto-updating spot price with real-time **🟢 LIVE** / **🔴 CLOSED** market indicators.

- **TradingView-Style Technical & Fundamental Analytics**:
  - **RSI (14)** oscillator sub-chart with overbought (70) and oversold (30) threshold bands.
  - **Shareholding Pattern**: Interactive donut chart displaying **FII**, **DII**, and **Retail / Promoter** stakes.
  - **Performance Horizon**: Trailing returns across 1W, 1M, 3M, 6M, and 1Y.
  - **Securities Results & Valuation**: P/E, EPS, Operating Margin, Beta, 52-Week High & Low.
  - **YoY & QoQ Comparison**: Annual and quarterly revenue and net income growth analysis.
  - **Stock History Ledger**: 10-day historical table with OHLCV data.

- **Agentic RAG & LLM Orchestrator**:
  - Persistent **ChromaDB** vector storage with auto-ingestion for zero-latency semantic recall on repeated searches.
  - **Groq LLM (Llama 3.3 70B)** trade decision engine calculating probability-tiered profit targets (T1–T4) and risk-reward levels.
  - Interactive embedded **AI Robo Advisor** chatbot.

- **Real-Time Market Data Feeds**:
  - **Indian Markets (NSE/BSE)**: Integrated with **Angel One SmartAPI** + TOTP authentication.
  - **US & Global Benchmarks**: Real-time Yahoo Finance market feeds.

---

## 📁 Project Architecture

```
lstm_model/
├── app.py                      # Flask backend API & web server
├── templates/
│   └── index.html              # Modern TailwindCSS & Plotly dark-themed dashboard
├── quant_worker/
│   ├── predictor.py            # Neural quant engine, moving averages, RSI & performance metrics
│   └── lstm_model.py           # Deep learning model definition
├── fundamental_worker/
│   ├── rag_search.py           # ChromaDB vector RAG, LLM news ranking, shareholders & filings
│   └── vector_store.py         # ChromaDB persistence & fallback pipeline
├── orchestrator/
│   ├── decision_agent.py       # Groq trade planner, targets T1-T4 & stop-loss engine
│   └── chatbot.py              # Financial assistant chat agent
├── trading_bot/
│   ├── engine.py               # Autonomous algo trading execution engine
│   ├── market_data.py          # Multi-asset live quote provider (Angel One + YFinance)
│   └── models.py               # Paper trading account, user auth & trade records
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition (Hugging Face Spaces ready)
└── .env                        # Private API credentials
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Enter Directory
```bash
git clone <your-repo-url>
cd "lstm_model"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Groq AI Orchestrator
GROQ_API_KEY=your_groq_api_key_here

# Angel One SmartAPI (Optional for Indian Real-Time Market Data)
ANGEL_API_KEY=your_smartapi_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PWD=your_account_password
ANGEL_TOTP_KEY=your_totp_secret_key
```

### 5. Run the Application
```bash
python app.py
```
Open your browser at **`http://localhost:5000`**.

---

## ☁️ Deployment

### Hugging Face Spaces (Recommended — Free 16GB RAM)
1. Create a **New Space** on [Hugging Face](https://huggingface.co/) and select **Docker** SDK.
2. Push the repository files.
3. In **Settings → Variables and Secrets**, add your `.env` variables (`GROQ_API_KEY`, `ANGEL_API_KEY`, etc.).
4. The space automatically builds and serves on port `7860`.

### Cloudflare Tunnel (Instant Free Public URL from Local Machine)
```bash
python app.py
# In another terminal:
cloudflared tunnel --url http://localhost:5000
```

---

## 🔒 Security
- All private API keys and credentials are kept strictly in `.env` and excluded from version control via `.gitignore` / `.dockerignore`.
- Paper trading runs in a sandboxed, risk-free environment.

