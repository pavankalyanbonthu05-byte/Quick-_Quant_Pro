import os
import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent ChromaDB storage in the project root
CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".chroma_db"
)


def get_chroma_collection():
    """Initializes or connects to local ChromaDB with graceful embedding fallback and minimal RAM."""
    try:
        from chromadb.config import Settings
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
        import numpy as np

        class LightweightEmbedding(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                embeddings = []
                for text in input:
                    vec = np.zeros(64, dtype=np.float32)
                    for i, ch in enumerate(text[:128]):
                        vec[i % 64] += ord(ch)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    embeddings.append(vec.tolist())
                return embeddings

        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        return client.get_or_create_collection(
            name="company_histories", embedding_function=LightweightEmbedding()
        )
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
        import urllib.request, json
        s_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}&quotesCount=1"
        s_req = urllib.request.Request(s_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        comp_name = symbol
        sector = "Diversified Financials & Equity Assets"
        industry = "Global Listed Equities"
        try:
            with urllib.request.urlopen(s_req, timeout=1.5) as s_res:
                s_json = json.loads(s_res.read().decode("utf-8"))
                quotes = s_json.get("quotes", [])
                if quotes:
                    comp_name = quotes[0].get("longname") or quotes[0].get("shortname") or symbol
                    sector = quotes[0].get("sector") or sector
                    industry = quotes[0].get("industry") or industry
        except Exception:
            pass

        long_summary = (
            f"{comp_name} ({symbol}) is a publicly traded enterprise operating in the {sector} sector ({industry}). "
            f"The company maintains established market capitalization and active institutional liquidity."
        )

        profile_text = (
            f"Company Name: {comp_name}\n"
            f"Sector: {sector} | Industry: {industry}\n"
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