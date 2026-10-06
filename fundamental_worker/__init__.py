try:
    from .graph import analyze_fundamentals
except ImportError as e:
    print(f"⚠️ Warning importing analyze_fundamentals: {e}")
    analyze_fundamentals = None

__all__ = ["analyze_fundamentals"]