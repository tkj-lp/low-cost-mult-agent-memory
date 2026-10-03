"""核验上游算法、完整状态接续和生命周期；随机小模型不代表答题成绩。"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from competition_agents.llm.latentmas_chat import LanguageForward, LatentMASChat, RecordedCore
from competition_agents.llm.native_local_chat import NativeLocalChat, text_decoder
from competition_agents.llm.generation import standard_generate
from competition_agents.vendor.latentmas.models import ModelWrapper
from competition_agents.collaboration_transport import CollaborationTransport
from competition_agents.role_prompts import role_prompt
from competition_agents.unified_collaboration import UnifiedCollaboration
from runtime_fixtures import cache_bundle, scripted_logits
from test_shared_runtime_end_to_end import ObservedMemory
from test_typed_decision_lifecycle import task_snapshot


def test_vendored_latent_algorithm_matches_fixed_upstream_functions():
    project = Path(__file__).resolve().parents[1]
    upstream = project/'tmp/upstream_latentmas_9a9e4d3/models.py'
    if not upstream.exists():
        pytest.skip('仅来源核验需要上游检出；算法和运行回归不依赖该目录。')
    def functions(path):
        tree = ast.parse(path.read_text(encoding='utf8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        return {n.name:ast.dump(n, include_attributes=False) for n in cls.body
                if isinstance(n, ast.FunctionDef)}
    left, right = functions(upstream), functions(project/'competition_agents/vendor/latentmas/models.py')
    for name in ('_build_latent_realign_matrix','_ensure_latent_realign_matrix',
                 '_apply_latent_realignment','generate_latent_batch'):
        assert left[name] == right[name]


def test_multimodal_full_core_continues_after_official_generation_warmup():
    bundle = cache_bundle('multimodal')
    ids = torch.tensor([[3, 4, 5, 6]])
    # 生产入口会先预热；官方生成会把位置偏移留在多模态外壳中。
    bundle.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                          max_new_tokens=1, do_sample=False)
    original_model = bundle.model
    full, small = RecordedCore(bundle, 'full'), RecordedCore(bundle, 'release_transients')
    past_full = past_small = None
    for request in (ids, torch.tensor([[7, 8]])):
        past_full = full.generate_latent_batch(request, latent_steps=4, past_key_values=past_full)
        past_small = small.generate_latent_batch(request, latent_steps=4, past_key_values=past_small)
        assert past_full.get_seq_length() == past_small.get_seq_length()
        for layer_a, layer_b in zip(past_full.layers, past_small.layers):
            for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
                a, b = getattr(layer_a, name, None), getattr(layer_b, name, None)
                if a is not None:
                    assert torch.equal(a, b)
    assert bundle.model is original_model
    assert full.model.get_input_embeddings() is original_model.get_input_embeddings()
    assert full.model.get_output_embeddings() is original_model.get_output_embeddings()


def test_language_entry_preserves_full_outputs_against_explicit_text_positions():
    bundle = cache_bundle('multimodal')
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.inference_mode():
        past = text_decoder(bundle)(input_ids=ids, use_cache=True).past_key_values
        suffix = torch.tensor([[7, 8]])
        mask = torch.ones((1, 6), dtype=torch.long)
        kwargs = dict(input_ids=suffix, attention_mask=mask, use_cache=True,
                      return_dict=True, output_hidden_states=True)
        actual = LanguageForward(bundle)(**kwargs, past_key_values=copy.deepcopy(past))
        expected = bundle.model(**kwargs, past_key_values=copy.deepcopy(past),
                                position_ids=torch.tensor([[4, 5]]))
    assert torch.equal(actual.logits, expected.logits)
    assert len(actual.hidden_states) == len(expected.hidden_states)
    for a, b in zip(actual.hidden_states, expected.hidden_states):
        assert torch.equal(a, b)


@pytest.mark.parametrize('architecture', ['attention', 'hybrid', 'multimodal'])
def test_role_request_uses_upstream_template_and_hands_off_exact_cache(architecture, monkeypatch):
    bundle = cache_bundle(architecture)
    bundle.tokenizer.chat_template += (
        "{% if add_generation_prompt %}{{ '<think>\\n' }}"
        "{% if enable_thinking is defined and not enable_thinking %}{{ '</think>\\n' }}{% endif %}"
        "{% endif %}")
    bundle.template_kwargs = {'enable_thinking': False}
    chat = LatentMASChat(NativeLocalChat(bundle), latent_steps=4, memory_policy='release_transients')
    transport = CollaborationTransport('latent', None, SimpleNamespace(source_aliases=[]))
    snapshot = task_snapshot()
    captured = {}
    original = chat.core.generate_latent_batch
    def observe(ids, **kwargs):
        captured.update(ids=ids.clone(), past=copy.deepcopy(kwargs['past_key_values']))
        return original(ids, **kwargs)
    monkeypatch.setattr(chat.core, 'generate_latent_batch', observe)
    chat.call_role('planner', role_prompt('continuous', 'planner'), snapshot,
                   scope='s', turn_key='1', packet=transport.packet(snapshot))
    request = next(row for row in chat.calls[-1]['inputs'] if row['category'] == 'latent_role_request')
    assert request['text'].endswith('<think>\n')
    assert '</think>' not in request['text']
    # 相同职责通过上游默认模板渲染得到同一请求；不是另外拼一个假思考前缀。
    current = request['text'].split('<user>\n', 1)[1].split('\n[end]\n', 1)[0]
    reference = ModelWrapper.render_chat(SimpleNamespace(tokenizer=bundle.tokenizer),
                                        [{'role': 'user', 'content': current}])
    assert request['text'] == reference
    assert bundle.template_kwargs == {'enable_thinking': False}
    assert chat._render('user', 'ordinary output', open_assistant=True).endswith('</think>\n')
    assert len(chat.ids) == captured['past'].get_seq_length() + captured['ids'].shape[1] + 4
    assert chat.ids[-4:] == [bundle.tokenizer.pad_token_id] * 4
    reference_core = RecordedCore(bundle, 'full')
    with torch.inference_mode():
        expected = reference_core.generate_latent_batch(captured['ids'], latent_steps=4,
                                                       past_key_values=captured['past'])
    for a, b in zip(chat.past.layers, expected.layers):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            if getattr(a, name, None) is not None:
                assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize('architecture', ['attention', 'hybrid', 'multimodal'])
def test_transient_release_preserves_upstream_full_layer_cache_and_next_scores(architecture):
    bundle = cache_bundle(architecture)
    full = RecordedCore(bundle, 'full')
    small = RecordedCore(bundle, 'release_transients')
    ids = torch.tensor([[3, 4, 5, 6]])
    past_full = full.generate_latent_batch(ids, latent_steps=4)
    past_small = small.generate_latent_batch(ids, latent_steps=4)
    assert past_full.get_seq_length() == past_small.get_seq_length() == 8
    for a, b in zip(full.latent_inputs, small.latent_inputs):
        assert torch.equal(a, b)
    for layer_a, layer_b in zip(past_full.layers, past_small.layers):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            a, b = getattr(layer_a, name, None), getattr(layer_b, name, None)
            if a is not None:
                assert torch.equal(a, b)
    suffix = torch.tensor([[7, 8]])
    with torch.inference_mode():
        a = bundle.model(input_ids=suffix, past_key_values=past_full).logits
        b = bundle.model(input_ids=suffix, past_key_values=past_small).logits
    assert torch.equal(a, b)


@pytest.mark.parametrize('architecture', ['attention', 'hybrid', 'multimodal'])
def test_latent_plan_has_real_steps_no_token_decode_and_exact_recovery(architecture, monkeypatch):
    bundle = cache_bundle(architecture)
    chat = LatentMASChat(NativeLocalChat(bundle), latent_steps=4, memory_policy='release_transients')
    original = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    transport = CollaborationTransport('latent', None, SimpleNamespace(source_aliases=[]))
    snapshot = task_snapshot()
    def forbidden(*args, **kwargs):
        raise AssertionError('隐式规划不调用文字生成、词表输出或文本解码。')
    monkeypatch.setattr(bundle.model, 'generate', forbidden)
    monkeypatch.setattr(bundle.model.get_output_embeddings(), 'forward', forbidden)
    monkeypatch.setattr(bundle.tokenizer, 'decode', forbidden)
    chat.call_role('planner', role_prompt('continuous', 'planner'), snapshot,
                   scope='s', turn_key='1', packet=transport.packet(snapshot))
    call = chat.calls[-1]
    assert call['latent_steps'] == 4 and call['output_tokens'] == 0
    assert sum(t.shape[1] for kind, t in chat.replay_segments if kind == 'embeds') == 4
    before = copy.deepcopy(chat.past)
    chat.past = None
    chat.metrics = {k: 0 for k in ('appended_forward_ms', 'forward_positions', 'cache_recovery_positions')}
    chat._ensure_state()
    with torch.inference_mode():
        expected = text_decoder(bundle)(input_ids=torch.tensor([[3]]), past_key_values=before).last_hidden_state
        actual = text_decoder(bundle)(input_ids=torch.tensor([[3]]), past_key_values=copy.deepcopy(chat.past)).last_hidden_state
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)
    assert not chat.finish_task(persisted=False)['released']
    assert chat.finish_task(persisted=True)['released']
    assert not any(kind == 'embeds' for kind, t in chat.replay_segments)
    assert all(torch.equal(v, original[k]) for k,v in bundle.model.state_dict().items())


@pytest.mark.parametrize('task_kind', ['continuous', 'codeact'])
@pytest.mark.parametrize('architecture', ['attention', 'multimodal'])
@pytest.mark.parametrize('memory_policy', ['full', 'release_transients'])
def test_latent_tool_memory_and_executor_complete_without_intermediate_work_text(task_kind, architecture, memory_policy, monkeypatch):
    bundle = cache_bundle(architecture)
    warmup = torch.tensor([[3, 4, 5]])
    bundle.model.generate(input_ids=warmup, attention_mask=torch.ones_like(warmup),
                          max_new_tokens=1, do_sample=False)
    client = NativeLocalChat(bundle, max_tokens=256)
    runtime = UnifiedCollaboration.__new__(UnifiedCollaboration)
    runtime.method, runtime.task_kind, runtime.max_searches = 'latent', task_kind, 2
    runtime.memory, runtime.client = ObservedMemory(), client
    runtime.transport = CollaborationTransport('latent', client, SimpleNamespace(source_aliases=[]))
    runtime.resident = LatentMASChat(client, latent_steps=2, memory_policy=memory_policy)
    def generated(*args, **kwargs):
        if kwargs['tool_only']:
            body = ({'action': 'deliver', 'query': None} if runtime.memory.records
                    else {'action': 'lookup', 'query': 'Alice Council year'})
        elif kwargs['role'] == 'reasoner':
            body = '```python\nresult = sum(inputs["values"])\n```'
        else:
            body = '6' if task_kind == 'codeact' else '1990'
        handle = scripted_logits(bundle, body)
        try:
            return standard_generate(*args, **kwargs)
        finally:
            handle.remove()
    monkeypatch.setattr('competition_agents.llm.continuous_state_chat.standard_generate', generated)
    for index, question in enumerate(('When did Alice join Council?', 'What was that year?')):
        task = {'task': question, 'inputs': {'values': [1,2,3]}, 'allowed_imports': [],
                'output_schema': {'type':'number'}, 'output_fields': []}
        row = runtime.run_task(task, history_scope='s') if task_kind == 'codeact' else runtime.run(question, history_scope='s')
        assert row['workflow_complete'] and not row['workflow_issues'] and not row['error']
        assert row['prediction'] == ('6' if task_kind == 'codeact' else '1990')
        assert all(c['latent_steps'] == 2 for c in row['calls'] if c['role'] != 'summarizer')
        assert all(c['intermediate_work_output_tokens'] == 0 for c in row['calls'])
        if index:
            assert row['metrics']['memory_only_tasks'] == 1
            assert row['calls'][0]['cached_input_tokens'] == row['cache_lifecycle']['retained_prefix_positions']
        if task_kind == 'codeact':
            assert row['execution_result']['result'] == 6
