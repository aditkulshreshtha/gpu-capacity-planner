# GPU Capacity Planner

GPU Capacity Planner estimates inference GPU requirements from explicit workload, model, hardware, and pricing assumptions. It is the first layer of a larger inference control plane simulator, focused on capacity planning before scheduling, cost attribution, or incident simulation.

The planner answers:

- How many GPUs are required for a workload?
- Is the workload compute-bound or memory-bound?
- How do input tokens, output tokens, context length, and agentic fan-out change demand?
- What is the rough cost per request and per million tokens?
- When does committed capacity run short under growth?
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

Generate a six-month capacity forecast:

```bash
PYTHONPATH=src python3 -m gpu_capacity_planner.cli --forecast --months 6
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

## Capacity Forecasting

Forecast mode compounds workload growth month by month and reruns the capacity model for each period. It reports:

- Required GPUs
- Committed GPUs
- Capacity shortfall
- Binding constraint
- Peak requests per second
- Mid-case monthly cost

This keeps the project focused on operational planning questions: when to buy or reserve capacity, what workload shape is driving the shortage, and whether the bottleneck is compute or KV-cache memory.

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
Required GPUs: 7
Binding constraint: compute
Compute GPUs: 7
  Prefill GPUs: 1
  Decode GPUs: 7
Memory GPUs: 2
Peak requests/sec: 23.15
Input tokens/sec: 92,593
Output tokens/sec: 11,574
SLO concurrency: 34.72 requests
KV cache at SLO concurrency: 47.68 GB
Estimated monthly cost: $17,885.00 - $37,814.00
Mid cost/request: $0.00087
Cost per 1M tokens: $0.0082 - $0.0174

Capacity Forecast:
Monthly growth rate: 12.0%
Committed GPUs: 8
First shortfall: month 4
Month  Req/day      Peak RPS  GPUs  Shortfall  Binding  Mid monthly cost
1       1,000,000     23.15     7          0  compute  $     26,572.00
2       1,120,000     25.93     7          0  compute  $     26,572.00
3       1,254,400     29.04     8          0  compute  $     30,368.00
4       1,404,928     32.52     9          1  compute  $     34,164.00
5       1,573,519     36.42    10          2  compute  $     37,960.00
6       1,762,342     40.79    11          3  compute  $     41,756.00
```

Actual output may differ as assumptions are updated.

## Validation

The planner includes a queue sanity check based on an Erlang C M/M/c approximation and Little's Law. It is not a production performance model, but it verifies that latency and queue depth rise non-linearly as utilization approaches saturation, which is expected queueing behavior.

Run:

```bash
PYTHONPATH=src python3 -m gpu_capacity_planner.cli --validate
```

## Roadmap

1. Capacity Planner and Forecast Report
2. Scheduler Policy Benchmarking
3. Cost Attribution
4. Reliability and SLO Guardrails
5. Incident Simulation

## Repository Status

This repository intentionally starts with Layer 1 only. Scheduler policies, tenant fairness, cost attribution, and reliability simulations are future layers so the capacity model can stand on its own first.
