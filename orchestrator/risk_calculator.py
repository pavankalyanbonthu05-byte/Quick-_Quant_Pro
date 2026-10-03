import numpy as np


def calculate_segmented_trade_plan(
    current_price: float,
    forecast_trajectory: list[float],
    volatility: float,
    win_probability: float,
) -> dict:
    """Breaks down the 30-day prediction trajectory into up to 4 risk-managed target segments.

    If win probability is < 60%, flags as NO_CALL and calculates the Breakeven
    range.
    """
    if not forecast_trajectory or len(forecast_trajectory) == 0:
        return {
            "signal": "NO_CALL",
            "win_probability_pct": win_probability,
            "entry_price": current_price,
            "stop_loss": None,
            "targets": [],
            "breakeven_range": [
                round(current_price * 0.97, 2),
                round(current_price * 1.03, 2),
            ],
            "rationale": (
                "Insufficient prediction trajectory data to issue a"
                " high-confidence trade call."
            ),
        }

    max_projected = float(max(forecast_trajectory))
    min_projected = float(min(forecast_trajectory))
    net_points = max_projected - current_price

    # 1. LOW PROBABILITY CHECK (<60% Win Probability or <2% Expected Move)
    if win_probability < 60.0 or abs(net_points / current_price) < 0.02:
        support = round(min_projected * 0.98, 2)
        resistance = round(max_projected * 1.02, 2)
        return {
            "signal": "NO_CALL",
            "win_probability_pct": win_probability,
            "entry_price": current_price,
            "stop_loss": None,
            "targets": [],
            "breakeven_range": [support, resistance],
            "rationale": (
                f"Winning probability is moderate/low ({win_probability}%)."
                f" Market is range-bound between {support} and {resistance}."
                " Recommending NO CALL."
            ),
        }

    # 2. HIGH / MEDIUM PROBABILITY BUY CALL: Break Trade into 2 to 4 Segments
    total_gain_points = round(max_projected - current_price, 2)

    # Calculate Stop Loss (Based on Volatility or 1.5x Daily Risk)
    daily_vol = volatility if volatility > 0 else 0.015
    stop_loss_dist = current_price * (daily_vol * 1.5)
    stop_loss = round(current_price - stop_loss_dist, 2)

    # Divide trade into 4 targets along the trajectory
    # T1: First 30% of move (High Probability: 80-90%)
    # T2: 60% of move (Medium Probability: 70%)
    # T3: 85% of move (Moderate Probability: 60%)
    # T4: 100% of move (Peak Projection: 50%)
    segment_ratios = [
        (0.30, "High Probability (80-90%)"),
        (0.60, "Medium Probability (65-75%)"),
        (0.85, "Moderate Probability (60%)"),
        (1.00, "Full Trajectory Target (50%)"),
    ]

    targets = []
    for idx, (ratio, prob_tier) in enumerate(segment_ratios, 1):
        target_pts = round(total_gain_points * ratio, 2)
        if target_pts <= 0:
            continue
        t_price = round(current_price + target_pts, 2)
        t_pct = round((target_pts / current_price) * 100, 2)
        targets.append({
            "target_number": idx,
            "target_price": t_price,
            "points_gain": target_pts,
            "return_pct": t_pct,
            "probability_tier": prob_tier,
        })

    rr_ratio = round(total_gain_points / (stop_loss_dist + 1e-5), 2)

    return {
        "signal": "BUY",
        "win_probability_pct": win_probability,
        "entry_price": current_price,
        "stop_loss": stop_loss,
        "risk_reward_ratio": rr_ratio,
        "targets": targets,
        "breakeven_range": None,
        "rationale": (
            f"High probability setup ({win_probability}% win rate). Strong"
            f" 30-day projection of +{total_gain_points} points with a"
            f" Risk-Reward ratio of {rr_ratio}."
        ),
    }