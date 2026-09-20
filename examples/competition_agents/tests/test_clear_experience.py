import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runner_server
from hypergraph_memory_method.store import SQLiteHypergraphMemoryStore


class ClearExperienceTests(unittest.TestCase):
    config = {"backend": "local", "localModel": "Qwen/Qwen2.5-1.5B-Instruct",
              "collaborationMode": "plain", "memoryMode": "hypergraph", "retrievalMode": "hypergraph"}

    def test_preserves_other_layers_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            result.write_text('{"saved":true}', encoding="utf-8")
            target_name = runner_server.memory_database_name(self.config)
            other_configs = [{**self.config, "collaborationMode": "hidden"},
                             {**self.config, "localModel": "Qwen/Qwen2.5-0.5B-Instruct"},
                             {**self.config, "retrievalMode": "fusion"}]
            names = [target_name, *(runner_server.memory_database_name(c) for c in other_configs)]
            for name in names:
                with SQLiteHypergraphMemoryStore(root / name) as store:
                    for layer in ("knowledge", "episodic", "experience"):
                        store.upsert({
                            "memory_id": "mem-0001",
                            "global_memory_id": f"global-{layer}",
                            "memory_layer": layer,
                            "memory_type": "experience" if layer == "experience" else "evidence",
                            "summary": layer,
                            "source_agent": "test",
                        })
            with patch.dict(os.environ, {"COMPETITION_MEMORY_ROOT": directory}), patch.object(
                runner_server, "has_running_process", return_value=False
            ):
                report = runner_server.clear_experience_memory(self.config)
                self.assertEqual(report["deletedCount"], 1)
                self.assertEqual(report["databaseCount"], 1)
                self.assertEqual(runner_server.clear_experience_memory(self.config)["deletedCount"], 0)
            for name in names:
                with SQLiteHypergraphMemoryStore(root / name) as store:
                    expected = {"knowledge", "episodic"} if name == target_name else {"knowledge", "episodic", "experience"}
                    self.assertEqual({r["memory_layer"] for r in store.list()}, expected)
            self.assertEqual(result.read_text(encoding="utf-8"), '{"saved":true}')

    def test_refuses_during_run(self):
        with patch.object(runner_server, "has_running_process", return_value=True):
            with self.assertRaises(ValueError):
                runner_server.clear_experience_memory(self.config)


if __name__ == "__main__":
    unittest.main()
