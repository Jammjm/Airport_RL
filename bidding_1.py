"""
Simple bidding helper compatible with the current simulator (sim_model_RL).

- Lower score is better.
- Score uses travel time to the gate and SoC (prefers higher SoC).
- If a path is missing, the bid is penalized heavily so it will be selected last.

Usage idea (manual hook):
    candidates = bidding_1.rank_units_by_bid(eligible, gate_label)
    selected = candidates[:n]

Integration: replace `_select_units_by_rule` inside AMRFleet with this ranking
if you want bidding-based dispatch instead of FIFO/random.
"""

from __future__ import annotations

from typing import List, Tuple

import config as cfg
from routing import _get_path_waypoints, _calculate_path_distance_and_time

# Large penalty to push infeasible paths to the end
_INF_PENALTY = 1e9


def compute_bid(unit, gate_label: str, weight_time: float = 1.0, weight_soc: float = 20) -> float:
    """Compute a bid score for one unit to serve a gate.

    Lower is better. Combines travel time (minutes) and SoC preference.
    """
    soc = getattr(unit, "soc_percent", 0.0)

    try:
        waypoints = _get_path_waypoints(unit.location, gate_label)
        _, travel_time = _calculate_path_distance_and_time(waypoints)
    except Exception:
        # No path; deprioritize this unit
        return _INF_PENALTY

    # Score: prefer shorter travel and higher SoC
    score = weight_time * travel_time - weight_soc * soc

    # Small tie-breaker by unit id to keep ordering stable
    unit_id = getattr(unit, "unit_id", 0)
    score += unit_id * 1e-6
    return float(score)


def rank_units_by_bid(units: List, gate_label: str, weight_time: float = 1.0, weight_soc: float = 0.5) -> List[Tuple[object, float]]:
    """Return units sorted by bid score (ascending)."""
    scored = []
    for u in units:
        score = compute_bid(u, gate_label, weight_time, weight_soc)
        scored.append((u, score))
    scored.sort(key=lambda x: x[1])
    return scored
