from typing import Any, Dict, List, Optional, TypedDict


class FundamentalState(TypedDict):
    # Input parameters
    symbol: str
    as_of_date: str
    # Financial metrics node output
    financials: Dict[str, Any]
    # Shareholding pattern node output
    shareholding: Dict[str, Any]
    # ChromaDB RAG node output
    company_profile: Dict[str, Any]
    is_rag_retrieved: bool
    # News & sentiment node output
    recent_news: List[Dict[str, Any]]
    # Final worker output status
    status: str
    error: Optional[str]