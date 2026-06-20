"""Capacity forecasting built on top of point-in-time plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import assumption_value
from .planner import CapacityPlan, build_capacity_plan


@dataclass(frozen=True)
class ForecastPeriod:
    month: int
    requests_per_day: float
    required_gpus: int
    committed_gpus: int
    gpu_shortfall: int
    binding_constraint: str
    monthly_cost_mid_usd: float
    peak_requests_per_second: float
    utilization_pressure: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastReport:
    hardware: str
    model: str
    pricing: str
    workload: str
    monthly_growth_rate: float
    committed_gpus: int
    first_shortfall_month: Optional[int]
    periods: List[ForecastPeriod]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware": self.hardware,
            "model": self.model,
            "pricing": self.pricing,
            "workload": self.workload,
            "monthly_growth_rate": self.monthly_growth_rate,
            "committed_gpus": self.committed_gpus,
            "first_shortfall_month": self.first_shortfall_month,
            "periods": [period.to_dict() for period in self.periods],
        }


def build_forecast_report(
    configs: Dict[str, Dict[str, Any]],
    hardware_key: str,
    model_key: str,
    pricing_key: str,
    workload_key: str,
    months: Optional[int] = None,
) -> ForecastReport:
    workload = configs["workloads"][workload_key]
    growth_rate = float(assumption_value(workload, "monthly_growth_rate"))
    committed_gpus = int(assumption_value(workload, "committed_gpus"))
    horizon = months or int(assumption_value(workload, "forecast_months"))

    periods: List[ForecastPeriod] = []
    first_shortfall_month: Optional[int] = None

    for month in range(1, horizon + 1):
        scenario_configs = _configs_for_forecast_month(configs, workload_key, month, growth_rate)
        plan = build_capacity_plan(
            scenario_configs,
            hardware_key=hardware_key,
            model_key=model_key,
            pricing_key=pricing_key,
            workload_key=workload_key,
        )
        shortfall = max(0, plan.required_gpus - committed_gpus)
        if shortfall and first_shortfall_month is None:
            first_shortfall_month = month
        periods.append(_period_from_plan(month, plan, committed_gpus, shortfall))

    return ForecastReport(
        hardware=hardware_key,
        model=model_key,
        pricing=pricing_key,
        workload=workload_key,
        monthly_growth_rate=growth_rate,
        committed_gpus=committed_gpus,
        first_shortfall_month=first_shortfall_month,
        periods=periods,
    )


def _configs_for_forecast_month(
    configs: Dict[str, Dict[str, Any]],
    workload_key: str,
    month: int,
    growth_rate: float,
) -> Dict[str, Dict[str, Any]]:
    scenario_configs = deepcopy(configs)
    workload = scenario_configs["workloads"][workload_key]
    growth_multiplier = (1 + growth_rate) ** (month - 1)
    for key in ("requests_per_day", "agent_tasks_per_day"):
        workload[key]["value"] = float(workload[key]["value"]) * growth_multiplier
    return scenario_configs


def _period_from_plan(
    month: int,
    plan: CapacityPlan,
    committed_gpus: int,
    shortfall: int,
) -> ForecastPeriod:
    utilization_pressure = plan.required_gpus / committed_gpus if committed_gpus else float("inf")
    return ForecastPeriod(
        month=month,
        requests_per_day=plan.requests_per_day,
        required_gpus=plan.required_gpus,
        committed_gpus=committed_gpus,
        gpu_shortfall=shortfall,
        binding_constraint=plan.binding_constraint,
        monthly_cost_mid_usd=plan.monthly_cost_mid_usd,
        peak_requests_per_second=plan.peak_requests_per_second,
        utilization_pressure=utilization_pressure,
    )
