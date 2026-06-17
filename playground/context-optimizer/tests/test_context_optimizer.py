import unittest

from context_optimizer import build_mock_log_cache as packaged_build_mock_log_cache
from context_optimizer_benchmark import (
    build_mock_log_cache,
    query_log_cache,
    run_compression_step,
    run_pipeline_a,
    run_pipeline_b,
)


class ContextOptimizerTests(unittest.TestCase):
    def test_package_namespace_exports(self):
        logs = packaged_build_mock_log_cache(total_lines=25)
        self.assertEqual(len(logs), 25)

    def test_mock_log_cache_size(self):
        logs = build_mock_log_cache(total_lines=300)
        self.assertEqual(len(logs), 300)

    def test_query_log_cache_returns_hits(self):
        output = query_log_cache.invoke({"keyword": "CosmosDB", "lines_context": 2})
        self.assertIn("keyword='CosmosDB'", output)

    def test_mock_compression_schema(self):
        compressed, latency = run_compression_step(None, "AKS timeout 21012 with CosmosDB retries", "mock")
        self.assertGreaterEqual(latency, 0.0)
        self.assertTrue(compressed.core_issue)
        self.assertIsInstance(compressed.observed_symptoms, list)
        self.assertIsInstance(compressed.technical_identifiers, list)

    def test_pipeline_raw_mock(self):
        output, latency, lines = run_pipeline_a(None, "incident", build_mock_log_cache(200), "mock")
        self.assertIn("Most likely root cause", output)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(lines, 200)

    def test_pipeline_optimized_mock(self):
        compressed, _ = run_compression_step(None, "AKS timeout 21012 with CosmosDB retries", "mock")
        output, latency, tool_calls, retrieved_lines = run_pipeline_b(None, compressed, "mock")
        self.assertIn("Most likely root cause", output)
        self.assertGreaterEqual(latency, 0.0)
        self.assertGreater(tool_calls, 0)
        self.assertGreater(retrieved_lines, 0)


if __name__ == "__main__":
    unittest.main()
