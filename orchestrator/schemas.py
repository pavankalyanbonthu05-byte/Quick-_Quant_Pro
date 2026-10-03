from typing import List, Optional
from pydantic import BaseModel, Field


class TradeTarget(BaseModel):
    target_number: int = Field(
        ..., description="Target segment index (1, 2, 3, or 4)"
    )
    target_price: float = Field(..., description="Target price level in $ / ₹")
    points_gain: float = Field(
        ..., description="Gain in points from current price"
    )
    return_pct: float = Field(
        ..., description="Percentage return at this target"
    )
    probability_tier: str = Field(
        ..., description="High (80-90%), Medium (60-70%), Speculative (50%)"
    )


class TradeRecommendation(BaseModel):
    symbol: str
    current_price: float
    signal: str = Field(..., description="BUY, SELL, or NO_CALL")
    win_probability_pct: float = Field(
        ..., description="Overall confidence / winning probability score"
    )

    # Trade execution parameters (Only present if BUY or SELL)
    entry_price: float
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    targets: List[TradeTarget] = []

    # Range bound parameters (Only present if NO_CALL)
    breakeven_range: Optional[List[float]] = Field(
        default=None, description="[Lower Support, Upper Resistance]"
    )

    rationale: str = Field(
        ..., description="Detailed breakdown of why this call was made"
    )