from __future__ import annotations

import pytest

import runner_server


VISIBLE_DATASETS = {
    "topiocqa-1",
    "topiocqa-2",
    "hotpot-3",
    "codeact-1",
}

TEXT_RETRIEVAL_MODES = {"keyword", "tag", "semantic"}
HYPERGRAPH_RETRIEVAL_MODES = {"hypergraph", "fusion"}


def test_runner_supports_visible_datasets_and_collaboration_modes() -> None:
    assert set(runner_server.DATASETS) >= VISIBLE_DATASETS
    assert runner_server.EXPERIMENT_MODES == {
        "plain": "baseline_text",
        "structured": "protocol_api",
        "protocol-vector": "protocol_api_embedding",
        "latent": "hidden_api",
    }


def test_text_memory_modes_build_effective_commands() -> None:
    for collaboration_mode in runner_server.EXPERIMENT_MODES:
        for dataset in VISIBLE_DATASETS:
            for retrieval_mode in TEXT_RETRIEVAL_MODES:
                config = runner_server.normalize_config(
                    {
                        "collaborationMode": collaboration_mode,
                        "dataset": dataset,
                        "memoryMode": "text",
                        "retrievalMode": retrieval_mode,
                        "taskCount": 1,
                    }
                )

                command = runner_server.build_command(config)
                assert "--memory-retrieval-mode" in command
                assert command[command.index("--memory-retrieval-mode") + 1] == retrieval_mode


def test_hypergraph_memory_modes_build_effective_commands() -> None:
    for collaboration_mode in runner_server.EXPERIMENT_MODES:
        config = runner_server.normalize_config(
            {
                "collaborationMode": collaboration_mode,
                "dataset": "topiocqa-1",
                "memoryMode": "hypergraph",
                "retrievalMode": "fusion",
                "taskCount": 1,
            }
        )
        command = runner_server.build_command(config)

    assert "run_demo.py" in command[1]
    assert "--experiment-mode" in command
    assert runner_server.EXPERIMENT_MODES[collaboration_mode] in command
    assert "--context-source" in command
    assert command[command.index("--context-source") + 1] == "hypergraph"


def test_memory_organization_rejects_incompatible_retrieval_modes() -> None:
    with pytest.raises(ValueError, match="普通文本记忆"):
        runner_server.normalize_config(
            {"memoryMode": "text", "retrievalMode": "fusion"}
        )
    with pytest.raises(ValueError, match="超图记忆"):
        runner_server.normalize_config(
            {"memoryMode": "hypergraph", "retrievalMode": "keyword"}
        )
