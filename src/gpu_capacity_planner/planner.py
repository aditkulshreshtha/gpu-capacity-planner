"""Core capacity planning model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Dict

from .config import assumption_value


BYTES_PER_GB = 1024**3
SECONDS_PER_DAY = 86_400
HOURS_PER_MONTH = 730


@dataclass(frozen=True)
class CapacityPlan:
    hardware: str
    model: str
    pricing: str
    workload: str
    requests_per_day: float
    peak_requests_per_second: float
    avg_input_tokens: float
    avg_output_tokens: float
    sequence_length_tokens: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    prefill_required_gpus: int
    decode_required_gpus: int
    compute_required_gpus: int
    memory_required_gpus: int
    required_gpus: int
    binding_constraint: str
    slo_concurrent_requests: float
    kv_cache_gb_at_slo_concurrency: float
    usable_kv_cache_gb_per_gpu: float
    estimated_ttft_ms: float
    estimated_decode_seconds: float
    estimated_end_to_end_seconds: float
    target_utilization: float
    monthly_cost_low_usd: float
    monthly_cost_mid_usd: float
    monthly_cost_high_usd: float
    cost_per_request_mid_usd: float
    cost_per_1m_tokens_low_usd: float
    cost_per_1m_tokens_mid_usd: float
    cost_per_1m_tokens_high_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_capacity_plan(
    configs: Dict[str, Dict[str, Any]],
    hardware_key: str,
    model_key: str,
    pricing_key: str,
    workload_key: str,
) -> CapacityPlan:
    hardware = configs["hardware"][hardware_key]
    model = configs["models"][model_key]
    pricing = configs["pricing"][pricing_key][hardware_key]["hourly_cost_usd"]
    workload = configs["workloads"][workload_key]

    requests_per_day = _requests_per_day(workload)
    avg_input_tokens = float(assumption_value(workload, "avg_input_tokens"))
    avg_output_tokens = float(assumption_value(workload, "avg_output_tokens"))
    sequence_length = avg_input_tokens + avg_output_tokens
    peak_rps = requests_per_day / SECONDS_PER_DAY * float(
        assumption_value(workload, "peak_to_average_ratio")
    )

    target_utilization = float(assumption_value(hardware, "target_utilization"))
    throughput = model["throughput"][hardware_key]
    prefill_mid = float(throughput["prefill_tokens_per_second"]["mid"])
    decode_mid = float(throughput["decode_tokens_per_second"]["mid"])
    prefill_demand_tps = peak_rps * avg_input_tokens
    decode_demand_tps = peak_rps * avg_output_tokens
    prefill_gpus = _ceil_at_least_one(prefill_demand_tps / (prefill_mid * target_utilization))
    decode_gpus = _ceil_at_least_one(decode_demand_tps / (decode_mid * target_utilization))
    compute_gpus = max(prefill_gpus, decode_gpus)

    latency_slo_seconds = float(assumption_value(workload, "p95_latency_slo_ms")) / 1000
    slo_concurrency = peak_rps * latency_slo_seconds
    kv_cache_gb = _kv_cache_gb(model, sequence_length, slo_concurrency)
    usable_kv_gb = _usable_kv_cache_gb(hardware, model)
    memory_gpus = _ceil_at_least_one(kv_cache_gb / usable_kv_gb)

    required_gpus = max(compute_gpus, memory_gpus)
    binding_constraint = "memory" if memory_gpus > compute_gpus else "compute"

    estimated_ttft_ms = (avg_input_tokens / (prefill_mid * target_utilization)) * 1000
    estimated_decode_seconds = avg_output_tokens / (decode_mid * target_utilization)
    estimated_end_to_end_seconds = estimated_ttft_ms / 1000 + estimated_decode_seconds

    monthly_low = required_gpus * float(pricing["low"]) * HOURS_PER_MONTH
    monthly_mid = required_gpus * float(pricing["mid"]) * HOURS_PER_MONTH
    monthly_high = required_gpus * float(pricing["high"]) * HOURS_PER_MONTH
    monthly_requests = requests_per_day * 365 / 12
    cost_per_request_mid = monthly_mid / monthly_requests

    requests_per_second_per_gpu = min(
        prefill_mid / max(avg_input_tokens, 1),
        decode_mid / max(avg_output_tokens, 1),
    )
    aggregate_tokens_per_second = (
        required_gpus
        * requests_per_second_per_gpu
        * (avg_input_tokens + avg_output_tokens)
        * target_utilization
    )
    cost_1m_low = _cost_per_1m_tokens(float(pricing["low"]), aggregate_tokens_per_second)
    cost_1m_mid = _cost_per_1m_tokens(float(pricing["mid"]), aggregate_tokens_per_second)
    cost_1m_high = _cost_per_1m_tokens(float(pricing["high"]), aggregate_tokens_per_second)

    return CapacityPlan(
        hardware=hardware_key,
        model=model_key,
        pricing=pricing_key,
        workload=workload_key,
        requests_per_day=requests_per_day,
        peak_requests_per_second=peak_rps,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        sequence_length_tokens=sequence_length,
        prefill_tokens_per_second=prefill_demand_tps,
        decode_tokens_per_second=decode_demand_tps,
        prefill_required_gpus=prefill_gpus,
        decode_required_gpus=decode_gpus,
        compute_required_gpus=compute_gpus,
        memory_required_gpus=memory_gpus,
        required_gpus=required_gpus,
        binding_constraint=binding_constraint,
        slo_concurrent_requests=slo_concurrency,
        kv_cache_gb_at_slo_concurrency=kv_cache_gb,
        usable_kv_cache_gb_per_gpu=usable_kv_gb,
        estimated_ttft_ms=estimated_ttft_ms,
        estimated_decode_seconds=estimated_decode_seconds,
        estimated_end_to_end_seconds=estimated_end_to_end_seconds,
        target_utilization=target_utilization,
        monthly_cost_low_usd=monthly_low,
        monthly_cost_mid_usd=monthly_mid,
        monthly_cost_high_usd=monthly_high,
        cost_per_request_mid_usd=cost_per_request_mid,
        cost_per_1m_tokens_low_usd=cost_1m_low,
        cost_per_1m_tokens_mid_usd=cost_1m_mid,
        cost_per_1m_tokens_high_usd=cost_1m_high,
    )


def _requests_per_day(workload: Dict[str, Any]) -> float:
    direct = float(assumption_value(workload, "requests_per_day"))
    agent_tasks = float(assumption_value(workload, "agent_tasks_per_day"))
    calls_per_task = float(assumption_value(workload, "avg_model_calls_per_task"))
    return direct + agent_tasks * calls_per_task


def _kv_cache_gb(model: Dict[str, Any], sequence_length: float, concurrent_requests: float) -> float:
    layers = float(assumption_value(model, "layers"))
    kv_heads = float(assumption_value(model, "kv_heads"))
    head_dim = float(assumption_value(model, "head_dim"))
    bytes_per_element = float(assumption_value(model, "bytes_per_element"))
    bytes_total = (
        2
        * layers
        * kv_heads
        * head_dim
        * sequence_length
        * bytes_per_element
        * concurrent_requests
    )
    return bytes_total / BYTES_PER_GB


def _usable_kv_cache_gb(hardware: Dict[str, Any], model: Dict[str, Any]) -> float:
    vram = float(assumption_value(hardware, "vram_gb"))
    reserve = float(assumption_value(hardware, "runtime_memory_reserve_gb"))
    weights = float(assumption_value(model, "weight_memory_gb"))
    usable = vram - reserve - weights
    if usable <= 0:
        raise ValueError("Model weights and runtime reserve exceed GPU VRAM")
    return usable


def _cost_per_1m_tokens(hourly_cost: float, aggregate_tokens_per_second: float) -> float:
    return hourly_cost / 3600 / aggregate_tokens_per_second * 1_000_000


def _ceil_at_least_one(value: float) -> int:
    return max(1, ceil(value))
