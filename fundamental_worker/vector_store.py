import os
import chromadb
from chromadb.utils import embedding_functions
import yfinance as yf

# Initialize persistent ChromaDB storage in the project root
CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".chroma_db"
)


def get_chroma_collection():
    """Initializes or connects to local ChromaDB with graceful embedding fallback and minimal RAM."""
    try:
        from chromadb.config import Settings
        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        return client.get_or_create_collection(name="company_histories")
    except Exception:
        client = chromadb.Client()
        return client.get_or_create_collection(name="company_histories")


def query_or_index_company_history(symbol: str) -> dict:
    """Agentic RAG Flow:

    1. Check ChromaDB vector store for existing company history.
    2. If missing (Fallback), fetch summary -> chunk -> embed & save in ChromaDB -> return context.
    """
    symbol = symbol.strip().upper()
    collection = get_chroma_collection()

    # 1. Try querying ChromaDB
    try:
        results = collection.get(ids=[f"profile_{symbol}"])
        if (
            results
            and results.get("documents")
            and len(results["documents"]) > 0
        ):
            doc = results["documents"][0]
            meta = results["metadatas"][0] if results.get("metadatas") else {}
            return {
                "company_origin": doc,
                "business_summary": meta.get("summary", doc[:300]),
                "is_rag_retrieved": True,
            }
    except Exception as e:
        print(f"ChromaDB lookup info: {e}")

    # 2. Agentic RAG Fallback: Auto-fetch & Index on Demand
    print(
        f"ℹ️ Ticker '{symbol}' not found in ChromaDB. Triggering Agentic RAG fallback indexing..."
    )
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        long_summary = info.get("longBusinessSummary", "")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        city = info.get("city", "")
        country = info.get("country", "")

        if not long_summary:
            long_summary = f"{symbol} is a publicly traded company operating in the {sector} sector ({industry})."

        profile_text = (
            f"Company Name: {info.get('longName', symbol)}\n"
            f"Sector: {sector} | Industry: {industry}\n"
            f"Headquarters: {city}, {country}\n"
            f"Business Overview & Origin:\n{long_summary}"
        )

        # Store in ChromaDB
        collection.add(
            documents=[profile_text],
            metadatas=[{
                "symbol": symbol,
                "sector": sector,
                "summary": long_summary[:300],
            }],
            ids=[f"profile_{symbol}"],
        )

        return {
            "company_origin": profile_text,
            "business_summary": long_summary[:300],
            "is_rag_retrieved": False,  # Freshly indexed on demand
        }
    except Exception as e:
        print(f"Error during RAG fallback for {symbol}: {e}")
        return {
            "company_origin": f"Origin and profile information currently unavailable for {symbol}.",
            "business_summary": f"Sector overview pending for {symbol}.",
            "is_rag_retrieved": False,
        }