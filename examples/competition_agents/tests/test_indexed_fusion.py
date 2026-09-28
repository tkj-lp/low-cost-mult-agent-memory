from unittest.mock import Mock

from hypergraph_memory_method import retrieval
from hypergraph_memory_method.fields import render_evidence


def test_fusion_uses_index_once_without_repeating_text_channels(monkeypatch):
    index = Mock()
    index.search.return_value = [{"memory_id": "a", "memory_type": "evidence",
                                  "retrieval_score": 9, "recall_channels": ["keyword", "semantic"]}]
    monkeypatch.setattr(retrieval, "get_cached_hypergraph_index", lambda memory: index)
    for name in ("_prepared", "search_keyword_memory", "search_tag_memory", "search_semantic_memory"):
        monkeypatch.setattr(retrieval, name, Mock(side_effect=AssertionError("重复执行文本检索")))
    memory = [{"memory_id": str(i), "memory_type": "evidence"} for i in range(100)]
    hits = retrieval.search_memory_by_mode(memory, "question", top_k=2, mode="fusion", type_limits={"evidence": 2})
    index.search.assert_called_once_with("question", top_k=2)
    assert hits[0]["fusion_channels"] == ["keyword", "semantic"]


def test_grouped_fields_preserve_subject_role_and_qualifiers():
    values = ["Larch Station: Larch Station is not in South Bay.",
              "Larch Station: The staff at Larch Station record temperature and rainfall every day."]
    rendered = render_evidence(values, fields=True)
    assert "is not in South Bay." in rendered
    assert "staff record every day: temperature and rainfall" in rendered
    assert rendered.count("Larch Station") == 1
