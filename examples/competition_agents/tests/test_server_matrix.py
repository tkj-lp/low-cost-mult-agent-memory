import json
from types import SimpleNamespace

import pytest

from competition_agents.methods.catalog import BASE, DATASETS
from competition_agents.methods.codeact import CodeActCollaboration, evaluate_execution
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from competition_agents.llm.response_content import final_content, model_messages


@pytest.mark.parametrize('method', ['plain', 'structured', 'autoform', 'llmlingua'])
def test_code_modes_execute_generated_program_and_use_real_transport(method, monkeypatch):
    from competition_agents.methods import llmlingua
    from baselines.llmlingua2 import llmlingua2_adapter
    monkeypatch.setattr(llmlingua, 'prepare_compressor', lambda: None)
    compressed = []
    def compress(text, **kwargs):
        compressed.append(text)
        return SimpleNamespace(provider='llmlingua2', payload='compressed narrative')
    monkeypatch.setattr(llmlingua2_adapter, 'compress_message', compress)
    class Client:
        def __init__(self):
            self.calls = []
        def call(self, role, messages):
            content = {'planner': '{"steps":["sum"],"query":"arithmetic"}',
                       'retriever': '{"sources":[]}', 'reasoner': '```python\nresult={"sum":sum(inputs["numbers"])}\n```',
                       'summarizer': 'The actual sum is 8.', 'autoform': 'formatted narrative'}[role]
            self.calls.append({'role': role, 'input_tokens': 1, 'output_tokens': 1,
                               'elapsed_ms': 1, 'finish_reason': 'stop'})
            return {'content': content}
    client = Client()
    runtime = CodeActCollaboration(client, {'knowledge': BASE/DATASETS['codeact']['knowledge'],
                                           'encode': lambda _: [1., 0.],
                                           'embed': lambda _: SimpleNamespace(vector=[1., 0.], model='test', provider='test')},
                                   structured=method == 'structured', method=method)
    task = {'task': 'Add the numbers', 'inputs': {'numbers': [3, 5]}, 'allowed_imports': [], 'output_fields': ['sum']}
    result = runtime.run_task(task, history_scope='test')
    runtime.memory.close()
    assert result['error'] is None
    assert evaluate_execution(result['execution_result'], {'sum': 8})['correct']
    assert len(result['calls']) == (7 if method == 'autoform' else 4)
    assert len(compressed) == (3 if method == 'llmlingua' else 0)
    executor = next(m for m in result['messages'] if m['to'] == 'executor')
    assert json.loads(executor['content']) == {'code': 'result={"sum":sum(inputs["numbers"])}\n', 'inputs': task['inputs']}
    if method in ('autoform', 'llmlingua'):
        assert result['messages'][0]['content'] in ('formatted narrative', 'compressed narrative')


def test_code_latent_decodes_only_for_executor_and_final_role():
    from test_latent_only import bundle_with_tokenizer
    bundle = bundle_with_tokenizer()
    chat = LatentOnlyChat(bundle, task_kind='codeact', max_tokens=2)
    chat.call('planner', question='Compute using inputs.', scope='code')
    chat.call('retriever')
    assert bundle.tokenizer.decode_calls == 0
    chat.call('reasoner', decode_external=True)
    chat.call('summarizer', external_result={'result': 4})
    assert bundle.tokenizer.decode_calls == 2
    assert any(i['category'] == 'external_tool_result' for i in chat.calls[-1]['inputs'])
    assert chat.calls[-1]['cached_prefix_positions'] > 0
    with pytest.raises(ValueError, match='执行器'):
        LatentOnlyChat(bundle).call('reasoner', decode_external=True)


def test_thinking_is_separated_only_when_completed():
    assert final_content('<think>draft</think>{"query":"x"}') == '{"query":"x"}'
    assert final_content('draft</think>answer', thinking_prefix=True) == 'answer'
    assert final_content('<think>unfinished') == '<think>unfinished'


def test_reasoning_template_adaptation_preserves_every_message_and_instruction():
    messages = [{'role': 'system', 'content': 'Role instructions'}, {'role': 'user', 'content': 'Actual task'},
                {'role': 'assistant', 'content': 'Previous output'}, {'role': 'user', 'content': 'Correction'}]
    adapted = model_messages('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages)
    assert adapted[0] == {'role': 'user', 'content': 'Role instructions\n\nActual task'}
    assert adapted[1:] == messages[2:]
    assert messages[0]['role'] == 'system'
    assert model_messages('unmodified-model', messages) is messages


def test_phi_state_fragments_do_not_end_the_whole_conversation():
    from test_latent_only import bundle_with_tokenizer
    bundle = bundle_with_tokenizer()
    bundle.model_name = 'microsoft/Phi-4-mini-instruct'
    class Tokenizer:
        bos_token = '<B>'
        eos_token = '<EOS>'
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            m = messages[0]
            return '<B><|' + m['role'] + '|>' + m['content'] + '<|end|>' + ('<|assistant|>' if add_generation_prompt else '<EOS>')
    bundle.tokenizer = Tokenizer()
    client = LatentOnlyChat(bundle)
    assert client._message('user', 'new fact') == '<B><|user|>new fact<|end|>'
    client.past = object()
    assert client._message('user', 'new fact') == '<|user|>new fact<|end|>'
    assert client._role_end() == '<|end|>'


def test_protocol_accepts_a_single_json_block_with_prose_and_rejects_forged_value():
    from test_field_collaboration import Knowledge, Client
    from competition_agents.field_collaboration import FieldCollaboration
    class Forged(Client):
        def call(self, role, messages):
            result = super().call(role, messages)
            if role == 'retriever':
                result['content'] = 'Selected facts:\n```json\n{"facts":[["s1","Cotton","color","purple"]]}\n```'
            return result
    result = FieldCollaboration(Knowledge(), Forged()).run('What color was Cotton?')
    assert result['error'] is None
    assert len(result['calls']) == 4
    delivered = json.loads(result['messages'][1]['content'])['facts'][0]
    assert delivered[0] == 's1' and delivered[3] == 'little white kitten'
    assert 'purple' not in result['messages'][1]['content']
    assert 'purple' in result['outputs']['retriever']
