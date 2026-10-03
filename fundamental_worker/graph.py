from datetime import datetime
from langgraph.graph import END, START, StateGraph
from .nodes import (
    extract_financials_node,
    extract_shareholding_node,
    fetch_recent_news_node,
    retrieve_rag_profile_node,
)
from .state import FundamentalState


def build_fundamental_graph():
    """Assembles and compiles the LangGraph workflow for Worker 2."""
    builder = StateGraph(FundamentalState)

    # 1. Add Execution Nodes
    builder.add_node("extract_financials", extract_financials_node)
    builder.add_node("extract_shareholding", extract_shareholding_node)
    builder.add_node("retrieve_rag_profile", retrieve_rag_profile_node)
    builder.add_node("fetch_recent_news", fetch_recent_news_node)

    # 2. Wire Sequence Edges
    builder.add_edge(START, "extract_financials")
    builder.add_edge("extract_financials", "extract_shareholding")
    builder.add_edge("extract_shareholding", "retrieve_rag_profile")
    builder.add_edge("retrieve_rag_profile", "fetch_recent_news")
    builder.add_edge("fetch_recent_news", END)

    # 3. Compile Graph
    return builder.compile()


# Single compiled graph instance
fundamental_graph = build_fundamental_graph()


def analyze_fundamentals(symbol: str) -> dict:
    """Public entrypoint tool for Worker 2 (LangGraph execution).

    Executes state graph nodes sequentially and returns final fundamental
    analysis.
    """
    initial_state: FundamentalState = {
        "symbol": symbol.strip().upper(),
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "financials": {},
        "shareholding": {},
        "company_profile": {},
        "is_rag_retrieved": False,
        "recent_news": [],
        "status": "pending",
        "error": None,
    }

    final_state = fundamental_graph.invoke(initial_state)
    return final_state