"""Run every stable default script, with socket connections forbidden."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "payoffs.py", "bayes_scoring.py", "lmsr.py", "point_in_time.py", "forecast_models.py",
    "text_and_neural.py", "execution_replay.py", "constraints.py", "market_making.py", "risk.py",
    "policy_learning.py", "paper_session.py", "quickstart_pattern.py", "quickstart_event.py",
    "quickstart_consistency.py", "quickstart_market_making.py", "first_trade_checklist.py",
)
BOOTSTRAP = """
import runpy, socket, sys
from pathlib import Path
def no_network(*args, **kwargs):
    raise AssertionError("default example attempted network access")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
path = Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
runpy.run_path(str(path), run_name="__main__")
"""


class ExampleTests(unittest.TestCase):
    def test_all_17_examples_run_offline_and_emit_honest_json(self):
        results = {}
        for filename in EXAMPLES:
            with self.subTest(example=filename):
                process = subprocess.run([sys.executable, "-c", BOOTSTRAP, str(ROOT / "examples" / filename)],
                                         cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stderr, "", process.stderr)
                result = json.loads(process.stdout)
                self.assertEqual(result["data"], "SYNTHETIC / ILLUSTRATIVE")
                self.assertIs(result["live_edge_claim"], False)
                self.assertTrue(result["result"])
                results[filename] = result["result"]
        self.assertEqual(len(results), 17)
        self.assertEqual(results["payoffs.py"]["initial_cost"], "4.10")
        self.assertEqual(results["payoffs.py"]["resale_realized_profit"], "0.40")
        self.assertEqual(results["quickstart_consistency.py"]["only_first_leg_fills_then_loses_profit"], "-1.24")
        self.assertEqual(results["first_trade_checklist.py"]["status"], "DO_NOT_PROCEED_FROM_SYNTHETIC_EVIDENCE")

    def test_canonical_output_is_deterministic(self):
        command = [sys.executable, str(ROOT / "examples" / "payoffs.py")]
        first = subprocess.check_output(command, cwd=ROOT, text=True)
        second = subprocess.check_output(command, cwd=ROOT, text=True)
        self.assertEqual(first, second)

    def test_default_code_has_no_network_or_credential_loading_imports(self):
        forbidden = {"requests", "httpx", "urllib", "socket", "web3", "dotenv", "keyring"}
        for directory in (ROOT / "src" / "prediction_market_lab", ROOT / "examples"):
            for path in directory.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = {item.name.split(".")[0] for item in node.names}
                    elif isinstance(node, ast.ImportFrom):
                        names = {(node.module or "").split(".")[0]}
                    else:
                        continue
                    allowed = {"urllib"} if path.name == "market_data.py" else set()
                    self.assertFalse((names & forbidden) - allowed, f"{path.name}: {names & forbidden}")

    def test_runtime_lock_has_pins_and_no_local_editable_path(self):
        lines = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 3)
        for line in lines:
            if line.strip() and not line.startswith("#"):
                self.assertIn("==", line)
                self.assertNotIn("file:", line)
                self.assertFalse(line.startswith("-e"))


if __name__ == "__main__":
    unittest.main()
