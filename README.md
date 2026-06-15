# GPU Capacity Planner v0

GPU Capacity Planner v0 estimates inference GPU requirements from explicit workload, model, hardware, and pricing assumptions. It is the first layer of a larger inference control plane simulator, focused on capacity planning before scheduling, cost attribution, or incident simulation.

The planner answers:

- How many GPUs are required for a workload?
- Is the workload compute-bound or memory-bound?
- How do input tokens, output tokens, context length, and agentic fan-out change demand?
- What is the rough cost per request and per million tokens?
- Does queueing behavior look sane as utilization approaches saturation?

All hardware throughput, pricing, and memory assumptions are approximate and date-sensitive. They are intended for comparative simulation, not procurement decisions. Update configs before using this model for real planning.

## Quick Start

Run the bundled example:

```bash
python3 -m gpu_capacity_planner.cli \
  --hardware h100_sxm \
  --model llama3_70b \
  --pricing on_demand_cloud \
  --workload enterprise_agentic \
  --output text
```

If running directly from a clone without installing the package:

```bash
PYTHONPATH=src python3 -m gpu_capacity_planner.cli --output json
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Model

The planner separates prefill and decode demand:

- Prefill demand is driven by input tokens.
- Decode demand is driven by generated output tokens.
- Context length influences KV-cache memory pressure.
- Agentic workloads can be modeled as user tasks that fan out into multiple model calls.

GPU count is calculated as:

```text
compute_required_gpus = max(prefill_required_gpus, decode_required_gpus)
memory_required_gpus = ceil(total_kv_cache_gb / usable_kv_cache_gb_per_gpu)
required_gpus = max(compute_required_gpus, memory_required_gpus)
```

KV cache is estimated with:

```text
KV cache bytes ~= 2 * layers * kv_heads * head_dim * sequence_length * bytes_per_element * concurrent_requests
```

The factor of 2 accounts for keys and values.

## Configuration

Assumptions live in YAML files under `configs/`:

- `hardware_profiles.yaml`: GPU memory, utilization target, and reservation assumptions.
- `model_profiles.yaml`: model architecture and throughput ranges by hardware.
- `pricing_profiles.yaml`: dated hourly GPU price ranges.
- `workload_profiles.yaml`: request volume, token shape, SLOs, seasonality, and agent fan-out.

Every assumption includes units, source notes, observed date, confidence, and notes. Ranges are preferred over point estimates where the real value varies by provider, deployment stack, or commitment model.

## Example Output

```text
Scenario: enterprise_agentic on llama3_70b / h100_sxm
Required GPUs: 9
Binding constraint: compute
Compute GPUs: 9
Memory GPUs: 1
Peak requests/sec: 23.15
KV cache at SLO concurrency: 5.79 GB
Estimated monthly cost: $20,995.20 - $44,582.40
Cost per 1M tokens: $0.20 - $0.43
```

Actual output may differ as assumptions are updated.

## Validation

The planner includes a queue sanity check based on an Erlang C M/M/c approximation and Little's Law. It is not a production performance model, but it verifies that latency and queue depth rise non-linearly as utilization approaches saturation, which is expected queueing behavior.

Run:

```bash
PYTHONPATH=src python3 -m gpu_capacity_planner.cli --validate
```

## Roadmap

1. Capacity Planner v0
2. Scheduler Simulator
3. Cost Attribution
4. Reliability and SLO Guardrails
5. Incident Simulation

## Repository Status

This repository intentionally starts with Layer 1 only. Scheduler policies, tenant fairness, cost attribution, and reliability simulations are future layers so the capacity model can stand on its own first.
