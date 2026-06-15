"""Queueing sanity checks for capacity plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import factorial
from typing import Any, Dict, List

from .planner import CapacityPlan


@dataclass(frozen=True)
class QueuePoint:
    utilization: float
    expected_wait_seconds: float
    expected_system_seconds: float
    expected_queue_depth: float


def queue_sanity_curve(plan: CapacityPlan) -> List[QueuePoint]:
    service_time = max(plan.estimated_end_to_end_seconds, 0.001)
    servers = max(plan.required_gpus, 1)
    points = []
    for utilization in (0.35, 0.55, 0.70, 0.82, 0.90, 0.95):
        arrival_rate = utilization * servers / service_time
        wait = erlang_c_wait_seconds(arrival_rate, service_time, servers)
        system_time = wait + service_time
        queue_depth = arrival_rate * wait
        points.append(
            QueuePoint(
                utilization=utilization,
                expected_wait_seconds=wait,
                expected_system_seconds=system_time,
                expected_queue_depth=queue_depth,
            )
        )
    return points


def validation_report(plan: CapacityPlan) -> Dict[str, Any]:
    points = queue_sanity_curve(plan)
    waits = [point.expected_wait_seconds for point in points]
    monotonic_wait = all(current <= nxt for current, nxt in zip(waits, waits[1:]))
    return {
        "method": "Erlang C M/M/c approximation plus Little's Law",
        "service_time_seconds": plan.estimated_end_to_end_seconds,
        "servers": plan.required_gpus,
        "monotonic_wait_growth": monotonic_wait,
        "points": [asdict(point) for point in points],
    }


def erlang_c_wait_seconds(arrival_rate: float, service_time_seconds: float, servers: int) -> float:
    if servers < 1:
        raise ValueError("servers must be positive")
    service_rate = 1 / service_time_seconds
    traffic_intensity = arrival_rate / service_rate
    utilization = traffic_intensity / servers
    if utilization >= 1:
        return float("inf")

    sum_terms = sum((traffic_intensity**n) / factorial(n) for n in range(servers))
    final_term = (traffic_intensity**servers) / (factorial(servers) * (1 - utilization))
    wait_probability = final_term / (sum_terms + final_term)
    return wait_probability / (servers * service_rate - arrival_rate)
