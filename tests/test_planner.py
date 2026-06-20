import unittest

from gpu_capacity_planner.config import load_default_configs
from gpu_capacity_planner.forecast import build_forecast_report
from gpu_capacity_planner.planner import build_capacity_plan
from gpu_capacity_planner.validation import validation_report


class CapacityPlannerTests(unittest.TestCase):
    def test_agentic_workload_expands_requests(self):
        configs = load_default_configs()
        plan = build_capacity_plan(
            configs,
            hardware_key="h100_sxm",
            model_key="llama3_70b",
            pricing_key="on_demand_cloud",
            workload_key="enterprise_agentic",
        )

        self.assertEqual(plan.requests_per_day, 1_000_000)
        self.assertGreaterEqual(plan.required_gpus, plan.compute_required_gpus)
        self.assertGreaterEqual(plan.required_gpus, plan.memory_required_gpus)
        self.assertGreater(plan.cost_per_1m_tokens_mid_usd, 0)

    def test_memory_can_bind_for_long_context(self):
        configs = load_default_configs()
        workload = configs["workloads"]["support_chat"]
        workload["avg_input_tokens"]["value"] = 200_000
        workload["avg_output_tokens"]["value"] = 2_000
        workload["requests_per_day"]["value"] = 2_000_000

        plan = build_capacity_plan(
            configs,
            hardware_key="h100_sxm",
            model_key="llama3_70b",
            pricing_key="on_demand_cloud",
            workload_key="support_chat",
        )

        self.assertGreater(plan.memory_required_gpus, 1)
        self.assertEqual(plan.required_gpus, max(plan.compute_required_gpus, plan.memory_required_gpus))

    def test_validation_waits_increase_with_utilization(self):
        configs = load_default_configs()
        plan = build_capacity_plan(
            configs,
            hardware_key="h100_sxm",
            model_key="llama3_70b",
            pricing_key="on_demand_cloud",
            workload_key="enterprise_agentic",
        )
        report = validation_report(plan)

        self.assertTrue(report["monotonic_wait_growth"])

    def test_forecast_growth_increases_capacity_pressure(self):
        configs = load_default_configs()
        report = build_forecast_report(
            configs,
            hardware_key="h100_sxm",
            model_key="llama3_70b",
            pricing_key="on_demand_cloud",
            workload_key="enterprise_agentic",
            months=4,
        )

        self.assertEqual(len(report.periods), 4)
        self.assertGreater(report.periods[-1].requests_per_day, report.periods[0].requests_per_day)
        self.assertGreaterEqual(report.periods[-1].required_gpus, report.periods[0].required_gpus)
        self.assertIsNotNone(report.first_shortfall_month)


if __name__ == "__main__":
    unittest.main()
