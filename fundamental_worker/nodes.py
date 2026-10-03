import yfinance as yf
from .state import FundamentalState
from .vector_store import query_or_index_company_history


def extract_financials_node(state: FundamentalState) -> dict:
    """Node 1: Extract core financial metrics (YoY Revenue, Profit Margins, Net Debt, PE, PB)."""
    symbol = state.get("symbol", "").strip().upper()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # Revenue Growth & Margins
        rev_growth = info.get("revenueGrowth", 0.0)
        rev_growth_pct = round(rev_growth * 100, 2) if rev_growth else 0.0

        profit_margins = info.get("profitMargins", 0.0)
        profit_margin_pct = (
            round(profit_margins * 100, 2) if profit_margins else 0.0
        )

        # Debt & Cash
        total_debt = info.get("totalDebt", 0.0) or 0.0
        total_cash = info.get("totalCash", 0.0) or 0.0
        net_debt = max(0.0, total_debt - total_cash)

        pe_ratio = round(info.get("trailingPE", 0.0) or 0.0, 2)
        pb_ratio = round(info.get("priceToBook", 0.0) or 0.0, 2)

        financials = {
            "revenue_yoy_pct": rev_growth_pct,
            "net_profit_margin_pct": profit_margin_pct,
            "net_debt": round(net_debt, 2),
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
        }
        return {"financials": financials}
    except Exception as e:
        print(f"Error in extract_financials_node for {symbol}: {e}")
        return {
            "financials": {
                "revenue_yoy_pct": 0.0,
                "net_profit_margin_pct": 0.0,
                "net_debt": 0.0,
                "pe_ratio": 0.0,
                "pb_ratio": 0.0,
            },
            "error": str(e),
        }


def extract_shareholding_node(state: FundamentalState) -> dict:
    """Node 2: Extract FII, DII, Promoter, and Retail shareholding distribution."""
    symbol = state.get("symbol", "").strip().upper()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        inst_pct = info.get("heldPercentInstitutions", 0.0) or 0.0
        insider_pct = info.get("heldPercentInsiders", 0.0) or 0.0

        fii_dii = round(inst_pct * 100, 2)
        promoter = round(insider_pct * 100, 2)

        # Estimate FII / DII breakdown and remaining retail float
        fii = round(fii_dii * 0.6, 2)
        dii = round(fii_dii * 0.4, 2)
        retail = max(0.0, round(100.0 - (fii_dii + promoter), 2))

        shareholding = {
            "fii_pct": fii,
            "dii_pct": dii,
            "promoter_pct": promoter,
            "retail_pct": retail,
        }
        return {"shareholding": shareholding}
    except Exception as e:
        print(f"Error in extract_shareholding_node for {symbol}: {e}")
        return {
            "shareholding": {
                "fii_pct": 0.0,
                "dii_pct": 0.0,
                "promoter_pct": 0.0,
                "retail_pct": 100.0,
            }
        }


def retrieve_rag_profile_node(state: FundamentalState) -> dict:
    """Node 3: Retrieve company origin & background using Agentic ChromaDB RAG."""
    symbol = state.get("symbol", "").strip().upper()
    try:
        profile_data = query_or_index_company_history(symbol)
        return {
            "company_profile": profile_data,
            "is_rag_retrieved": profile_data.get("is_rag_retrieved", False),
        }
    except Exception as e:
        print(f"Error in retrieve_rag_profile_node for {symbol}: {e}")
        return {
            "company_profile": {
                "company_origin": "Profile unavailable.",
                "business_summary": "Summary unavailable.",
                "is_rag_retrieved": False,
            }
        }


def fetch_recent_news_node(state: FundamentalState) -> dict:
    """Node 4: Fetch ticker-specific news headlines."""
    symbol = state.get("symbol", "").strip().upper()
    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news or []
        cleaned_news = []

        for item in raw_news[:5]:
            content = item.get("content", item)
            cleaned_news.append({
                "title": content.get("title", "Market Update"),
                "publisher": content.get("provider", {}).get(
                    "displayName", "Financial Press"
                ),
                "link": content.get("canonicalUrl", {}).get("url", "#"),
                "publish_time": content.get("pubDate", "Recent"),
            })

        return {"recent_news": cleaned_news, "status": "completed"}
    except Exception as e:
        print(f"Error in fetch_recent_news_node for {symbol}: {e}")
        return {"recent_news": [], "status": "completed_with_errors"}