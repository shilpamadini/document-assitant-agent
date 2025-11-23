import unittest
from src.tools import ToolLogger, create_calculator_tool


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        # Use a separate logs directory for tests
        self.logger = ToolLogger(logs_dir="./logs_test")
        self.calculator = create_calculator_tool(self.logger)

    def test_simple_addition(self):
        """Calculator should correctly compute a simple addition."""
        result = self.calculator.run("2 + 3")
        self.assertIn("2 + 3", result)
        self.assertIn("5", result)

    def test_complex_expression(self):
        """Calculator should evaluate valid complex expressions."""
        result = self.calculator.run("(10 + 5) * 2")
        self.assertIn("30", result)

    def test_invalid_expression_rejected(self):
        """Calculator must reject unsafe or malformed input."""
        result = self.calculator.run("__import__('os').system('rm -rf /')")
        self.assertIn("Invalid expression", result)

    def test_logging_occurs(self):
        """ToolLogger should record tool usage on each call."""
        initial_len = len(self.logger.get_logs())
        _ = self.calculator.run("1 + 1")
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), initial_len + 1)
        last_log = logs[-1]
        self.assertEqual(last_log["tool_name"], "calculator")
        self.assertIn("input", last_log)
        self.assertIn("output", last_log)


if __name__ == "__main__":
    unittest.main()
