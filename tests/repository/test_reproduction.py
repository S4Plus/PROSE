"""Regression checks for portable experiment orchestration; no model imports."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_act as runner
from prose.paths import lerobot_root


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/reproduction/act.example.json").read_text(encoding="utf-8"))
        for key in ("dataset_root", "pretrained_dir", "rollout_dir", "output_root"):
            self.config[key] = str(self.base / key)
        self.file = self.base / "config.json"
        self.file.write_text(json.dumps(self.config), encoding="utf-8")
        self.config = runner.load_config(self.file)

    def test_relative_paths_use_checkout(self):
        with patch.object(runner, "ROOT", self.base):
            self.assertEqual(runner.resolve_path("datasets/task"), self.base / "datasets/task")

    def test_environment_override_for_lerobot(self):
        with patch.dict("os.environ", {"LEROBOT_REPO": str(self.base)}):
            self.assertEqual(lerobot_root(), self.base.resolve())

    def test_rejects_invalid_configuration(self):
        cases = [("run_name", "../other"), ("steps", True), ("seed", -1),
                 ("window_stride", 0), ("episode_drop_ratio", float("nan")), ("unknown", 1)]
        for key, value in cases:
            with self.subTest(key=key):
                payload = dict(self.config)
                payload[key] = value
                self.file.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_config(self.file)

    def test_commands_match_actual_algorithm_options(self):
        for stage in ("score", "select", "retrain"):
            with self.subTest(stage=stage):
                command = runner.build_command(self.config, stage)
                tree = ast.parse(Path(command[1]).read_text(encoding="utf-8-sig"))
                supported = {
                    argument.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                }
                used = {token.split("=", 1)[0] for token in command[2:] if token.startswith("--")}
                self.assertFalse(used - supported, f"Unsupported options: {used - supported}")

    def test_baseline_disables_hub_upload(self):
        command = runner.build_command(self.config, "train")
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn("--wandb.enable=false", command)

    def test_retraining_applies_selection_without_warm_start(self):
        command = runner.build_command(self.config, "retrain")
        self.assertIn("--selection-override-episodes", command)
        self.assertIn("--segment-mask", command)
        self.assertNotIn("--pretrained-dir", command)

    def test_preflight_is_stage_specific(self):
        dataset = Path(self.config["dataset_root"])
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text("{}", encoding="utf-8")
        self.assertEqual(runner.preflight(self.config, "train"), [])
        self.assertEqual(len(runner.preflight(self.config, "score")), 3)
        checkpoint = Path(self.config["pretrained_dir"])
        checkpoint.mkdir()
        for name in ("config.json", "model.safetensors"):
            (checkpoint / name).touch()
        rollout = Path(self.config["rollout_dir"])
        rollout.mkdir()
        (rollout / "episode.pkl").touch()
        self.assertEqual(runner.preflight(self.config, "score"), [])
        paths = runner.artifacts(self.config)
        self.assertTrue(runner.preflight(self.config, "select"))
        paths["score"].parent.mkdir(parents=True)
        paths["score"].touch()
        self.assertEqual(runner.preflight(self.config, "select"), [])
        self.assertTrue(runner.preflight(self.config, "retrain"))
        paths["selection"].mkdir()
        paths["selection_npz"].touch()
        self.assertEqual(runner.preflight(self.config, "retrain"), [])

    def test_existing_training_output_stops_execution(self):
        paths = runner.artifacts(self.config)
        paths["baseline"].mkdir(parents=True)
        errors = runner.preflight(self.config, "train")
        self.assertTrue(any("already exists" in error for error in errors))

    def test_preview_does_not_create_outputs_or_launch(self):
        with patch.object(runner, "execute") as execute, redirect_stdout(StringIO()):
            self.assertEqual(runner.main(["--config", str(self.file), "--stage", "train"]), 0)
        execute.assert_not_called()
        self.assertFalse(Path(self.config["output_root"]).exists())

    def test_execution_records_real_process_and_failure(self):
        with redirect_stdout(StringIO()):
            result = runner.execute(self.config, "score", [sys.executable, "-c", "print('recorded output'); raise SystemExit(3)"])
        self.assertEqual(result, 3)
        log_dir = next((runner.artifacts(self.config)["run"] / "logs").iterdir())
        manifest = json.loads((log_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["returncode"], 3)
        self.assertEqual(manifest["config"]["dataset_root"], self.config["dataset_root"])
        self.assertIn("recorded output", (log_dir / "stdout.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
