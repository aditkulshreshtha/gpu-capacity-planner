"""Command-line interface for GPU Capacity Planner v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR, load_default_configs
from .planner import CapacityPlan, build_capacity_plan
from .validation import validation_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU Capacity Planner v0")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--hardware", default="h100_sxm")
    parser.add_argument("--model", default="llama3_70b")
    parser.add_argument("--pricing", default="on_demand_cloud")
    parser.add_argument("--workload", default="enterprise_agentic")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    configs = load_default_configs(args.config_dir)
    plan = build_capacity_plan(
        configs=configs,
        hardware_key=args.hardware,
        model_key=args.model,
        pricing_key=args.pricing,
        workload_key=args.workload,
    )

    payload: dict[str, Any] = {"plan": plan.to_dict()}
    if args.validate:
        payload["validation"] = validation_report(plan)

    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_plan(plan))
        if args.validate:
            print()
            print(_format_validation(payload["validation"]))
    return 0


def _format_plan(plan: CapacityPlan) -> str:
    lines = [
        f"Scenario: {plan.workload} on {plan.model} / {plan.hardware}",
        f"Required GPUs: {plan.required_gpus}",
        f"Binding constraint: {plan.binding_constraint}",
        f"Compute GPUs: {plan.compute_required_gpus}",
        f"  Prefill GPUs: {plan.prefill_required_gpus}",
        f"  Decode GPUs: {plan.decode_required_gpus}",
        f"Memory GPUs: {plan.memory_required_gpus}",
        f"Peak requests/sec: {plan.peak_requests_per_second:.2f}",
        f"Input tokens/sec: {plan.prefill_tokens_per_second:,.0f}",
        f"Output tokens/sec: {plan.decode_tokens_per_second:,.0f}",
        f"SLO concurrency: {plan.slo_concurrent_requests:.2f} requests",
        f"KV cache at SLO concurrency: {plan.kv_cache_gb_at_slo_concurrency:.2f} GB",
        f"Usable KV cache per GPU: {plan.usable_kv_cache_gb_per_gpu:.2f} GB",
        f"Estimated TTFT: {plan.estimated_ttft_ms:.1f} ms",
        f"Estimated decode time: {plan.estimated_decode_seconds:.2f} sec",
        (
            "Estimated monthly cost: "
            f"${plan.monthly_cost_low_usd:,.2f} - ${plan.monthly_cost_high_usd:,.2f}"
        ),
        f"Mid cost/request: ${plan.cost_per_request_mid_usd:.5f}",
        (
            "Cost per 1M tokens: "
            f"${plan.cost_per_1m_tokens_low_usd:.4f} - "
            f"${plan.cost_per_1m_tokens_high_usd:.4f}"
        ),
    ]
    return "\n".join(lines)


def _format_validation(report: dict[str, Any]) -> str:
    lines = [
        "Validation:",
        f"Method: {report['method']}",
        f"Monotonic wait growth: {report['monotonic_wait_growth']}",
        "Utilization  Wait(s)  System(s)  QueueDepth",
    ]
    for point in report["points"]:
        lines.append(
            f"{point['utilization']:.2f}         "
            f"{point['expected_wait_seconds']:.3f}    "
            f"{point['expected_system_seconds']:.3f}     "
            f"{point['expected_queue_depth']:.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
