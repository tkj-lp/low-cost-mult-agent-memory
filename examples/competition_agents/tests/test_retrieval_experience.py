import json

import pytest
import torch

from competition_agents.latent_hidden.experienced_graph_tool import ExperiencedGraphTool
from hypergraph_memory_method.retrieval_experience import RetrievalExperienceMemory
from test_calibrated_latent import fake_tool


@pytest.fixture
def experience(tmp_path):
    memory = RetrievalExperienceMemory(tmp_path/'experience.sqlite', 'fixed-knowledge-and-model')
    yield memory
    memory.close()


def learn_two(tool):
    tool.begin_question('Alice color?', 'session')
    tool(torch.tensor([[1., 0., 0., 0.]]))
    tool.finish_task('task-1', completed=True)
    tool.begin_question('Where does Alice live?', 'session')
    tool(torch.tensor([[0., 1., 0., 0.]]))
    tool.finish_task('task-2', completed=True)


def test_latest_strategy_is_stable_when_clock_timestamps_tie(experience, monkeypatch):
    from datetime import datetime, timezone
    import hypergraph_memory_method.retrieval_experience as module
    fixed = datetime(2026, 9, 27, tzinfo=timezone.utc)
    class FrozenClock:
        @staticmethod
        def now(zone):
            return fixed
    monkeypatch.setattr(module, 'datetime', FrozenClock)
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    learn_two(tool)
    chosen = experience.lookup('companions')
    assert chosen['observed_categories'] == ['location']
    assert chosen['observation_order'] == 2
    restored = RetrievalExperienceMemory(experience.store.path, experience.namespace)
    try:
        assert restored.lookup('companions')['memory_id'] == chosen['memory_id']
    finally:
        restored.close()


def test_experience_changes_candidate_search_without_changing_evidence(experience, monkeypatch):
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    assert not experience.records
    learn_two(tool)
    assert len(experience.records) == 2
    tool.begin_question('Does Alice live alone?', 'session')
    scored = []
    original = tool._scores
    def capture(query, paths):
        scored.extend(paths)
        return original(query, paths)
    monkeypatch.setattr(tool, '_scores', capture)
    result = tool(torch.tensor([[0., 0., 1., 0.]]))
    audit = result['audit']
    assert audit['experience_reads'] == audit['experience_applied'] == 1
    assert audit['experience_candidates_skipped'] == 1
    assert 'color' not in scored
    assert set(audit['source_ids']) == {'location', 'companions'}
    assert audit['knowledge_searches'] == 0
    assert audit['experience_certificate']['passed']
    tool.finish_task('task-3', completed=True)
    used = experience.records[audit['experience_id']]
    assert used['reuse_count'] == used['reuse_success_count'] == 1
    assert used['answer_correctness_verified'] is False
    assert used['observed_categories'] == ['companions', 'location']
    serialized = json.dumps(list(experience.records.values()))
    assert 'Alice' not in serialized and 'North' not in serialized
    assert all(experience.store.get(ref) for r in experience.records.values() for ref in r['source_memory_ids'])


def test_uncertain_bound_falls_back_without_losing_competing_relation(experience):
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    learn_two(tool)
    tool.begin_question('Alice?', 'session')
    tool._experience_audit = {}
    query = torch.nn.functional.normalize(torch.tensor([[.75, .7, .2, 0.]]), dim=-1)
    paths = {k: v['path'] for k, v in tool.resident_paths.items()}
    expected = tool._scores(query, paths)
    actual = tool._memory_scores(query, paths, 'companions')
    assert actual == expected
    assert tool._experience_audit['experience_rejection'] == 'bound_or_evidence_not_sufficient'


def test_random_queries_preserve_full_search_selection(experience):
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    learn_two(tool)
    paths = {k: v['path'] for k, v in tool.resident_paths.items()}
    generator = torch.Generator().manual_seed(19)
    for vector in torch.randn(100, 4, generator=generator):
        query = torch.nn.functional.normalize(vector[None, :], dim=-1)
        category = ['color', 'location', 'companions', 'time'][int(vector.argmax())]
        tool._experience_audit = {}
        expected = tool._select(tool._scores(query, paths), paths)[0]
        actual = tool._select(tool._memory_scores(query, paths, category), paths)[0]
        assert actual == expected


def test_group_radius_does_not_hide_a_strong_competing_fact(experience):
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    learn_two(tool)
    for key, vector in [('other-low', [1., 0., 0., 0.]), ('other-high', [0., 0., 1., 0.])]:
        tool.positions[key] = len(tool.vectors)
        tool.vectors = torch.cat([tool.vectors, torch.tensor([vector])])
        tool.records[key] = {'relation': 'painted'}
        tool.resident_paths[key] = {'path': (key,)}
    paths = {k: v['path'] for k, v in tool.resident_paths.items()}
    query = torch.tensor([[0., 0., 1., 0.]])
    tool._experience_audit = {}
    expected = tool._select(tool._scores(query, paths), paths)[0]
    actual = tool._select(tool._memory_scores(query, paths, 'companions'), paths)[0]
    assert actual == expected and 'other-high' in actual
    assert not tool._experience_audit['experience_certificate']['passed']


def test_constraints_missing_evidence_and_failed_tasks_do_not_become_success(experience):
    tool = fake_tool(ExperiencedGraphTool, experience=experience)
    learn_two(tool)
    tool.begin_question('Where did Alice live before 2020?', 'session')
    result = tool(torch.tensor([[0., 1., 0., 0.]]))
    assert result['audit']['experience_applied'] == 0
    assert result['audit']['knowledge_searches'] == 1
    count = sum(r['observations'] for r in experience.records.values())
    tool.finish_task('failed', completed=False)
    assert sum(r['observations'] for r in experience.records.values()) == count
    tool.begin_question('Does Alice live alone?', 'session')
    tool.records.pop('companions')
    result = tool(torch.tensor([[0., 0., 1., 0.]]))
    assert not result['audit']['experience_applied']
    assert result['audit']['knowledge_searches'] == 1
    tool.begin_question('Alice?', 'different-session')
    assert not tool.records
    assert tool(torch.tensor([[0., 0., 1., 0.]]))['audit']['knowledge_searches'] == 1


def test_persistence_scope_idempotency_and_pruning(tmp_path):
    path = tmp_path/'experience.sqlite'
    memory = RetrievalExperienceMemory(path, 'first', capacity=1)
    params = dict(task_key='one', category='location', relation_paths=[('lived in',)], source_ids=['source-1'])
    assert memory.commit(**params)['created'] == 1
    assert memory.commit(**params)['writes'] == 0
    assert memory.commit(task_key='two', category='color', relation_paths=[('was',)], source_ids=['source-2'])['pruned'] == 1
    assert len(memory.records) == 1
    saved = set(memory.records)
    assert memory.store.count() == 2  # 一条经验与仍被引用的一条轨迹。
    memory.close()
    with_scope = RetrievalExperienceMemory(path, 'first')
    assert set(with_scope.records) == saved
    with_scope.close()
    other_scope = RetrievalExperienceMemory(path, 'other-knowledge-version')
    assert not other_scope.records
    other_scope.close()
