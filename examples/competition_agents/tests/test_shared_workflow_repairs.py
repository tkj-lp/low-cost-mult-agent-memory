import copy
from types import SimpleNamespace

from competition_agents.collaboration_transport import CollaborationTransport, full_evidence_packet
from competition_agents.llm.native_local_chat import state_message
from competition_agents.llm.response_content import model_messages
from competition_agents.methods.text import SimilarityTextMemory
from competition_agents.role_prompts import role_prompt, role_prompt_parts
from competition_agents.unified_collaboration import UnifiedCollaboration


class LastQueryTemplate:
    """模拟官方模板按最后用户位置处理助手思考区的行为。"""

    bos_token = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        last_user = max(i for i, message in enumerate(messages) if message['role'] == 'user')
        fragments = []
        for i, message in enumerate(messages):
            if message['role'] == 'system' and i:
                raise ValueError('System message must be at the beginning.')
            reasoning = '<think></think>' if message['role'] == 'assistant' and i > last_user else ''
            fragments.append('<' + message['role'] + '>' + reasoning + message['content'] + '</end>\n')
        return ''.join(fragments) + ('<assistant><think></think>' if add_generation_prompt else '')


def test_official_template_continuation_has_stable_prefix_and_no_anchor_content():
    bundle = SimpleNamespace(tokenizer=LastQueryTemplate(), template_kwargs={'enable_thinking': False})
    for role in ('planner', 'retriever', 'reasoner', 'summarizer'):
        common, prompt = role_prompt_parts('continuous', role)
        # 续接只追加当前职责，公共系统消息保留在真实前缀中。
        messages = [{'role': 'user', 'content': prompt}]
        closed = state_message(bundle, messages, continuation=True, open_assistant=False)
        opened = state_message(bundle, messages, continuation=True, open_assistant=True)
        assert closed == '<user>' + prompt + '</end>\n'
        assert opened == closed + '<assistant><think></think>'
        assert '__COMPETITION_PRIOR' not in opened


def test_shared_system_rules_and_current_role_follow_data_for_every_model():
    common, current = role_prompt_parts('continuous', 'summarizer')
    original = [{'role': 'system', 'content': common + current},
                {'role': 'user', 'content': 'Evidence and diagnostics. Current task: When?'}]
    for model in ('Qwen/example', 'XiaomiMiMo/example', 'another-model'):
        effective = model_messages(model, original, shared_roles=True)
        assert effective[0] == {'role': 'system', 'content': common}
        assert effective[1]['content'] == original[1]['content'] + '\n\n' + current


def test_all_shared_roles_preserve_identical_instruction_text_without_permanent_system_role():
    original = [{'role': 'system', 'content': 'Exact role duties.'}, {'role': 'user', 'content': 'All evidence.'}]
    before = copy.deepcopy(original)
    for model in ('Qwen/Qwen3.5-9B', 'deepseek-ai/DeepSeek-R1-example', 'another-model'):
        effective = model_messages(model, original, shared_roles=True)
        assert effective == [{'role': 'user', 'content': 'Exact role duties.'},
                             {'role': 'user', 'content': 'All evidence.'}]
    assert original == before
    assert model_messages('another-model', original) is original


def test_protocol_keeps_every_relation_and_alias_identity_but_separates_audit_quotes():
    fact = {'document_id': 'doc', 'subject': 'A', 'relation': 'worked at', 'object': 'B',
            'polarity': 'negative', 'qualifiers': [{'start': 1990, 'condition': 'during winter'}],
            'sentences': [1, 4]}
    evidence = {'one': {'title': 'Source', 'fact_fields': fact},
                'two': {'title': 'Source', 'fact_fields': {**fact, 'object': 'C'}}}
    aliases = [{'document_id': 'doc', 'alias': 'Alice', 'name': 'A', 'sentences': [0],
                'scoped': True, 'basis': 'Extraction explanation', 'quote': 'Full source ' * 100},
               {'document_id': 'outside', 'alias': 'X', 'name': 'Y'}]
    original = copy.deepcopy((evidence, aliases))
    packet = full_evidence_packet(evidence, aliases)
    assert len(packet['facts']) == 2
    assert packet['facts'][0] == ['one', 'A', 'worked at', 'B', 'negative', fact['qualifiers'], 'doc', [1, 4]]
    assert packet['facts'][1][3] == 'C'
    assert packet['sources'] == {'doc': 'Source'}
    assert packet['aliases'] == [{key: aliases[0][key] for key in
        ('alias', 'name', 'document_id', 'sentences', 'scoped')}]
    assert (evidence, aliases) == original


class Memory:
    enabled = True

    def __init__(self, *, text_units=False):
        self.records = {'old': {'context_id': 'old', 'title': 'Old source', 'text': 'A worked at B in 1990.'}}
        self.read_queries, self.search_queries = [], []
        self.text_units = text_units

    def recall(self, query, scope, kind):
        self.read_queries.append(query)
        hits = [dict(self.records['old'], from_memory=True)] if query == 'A worked at B date' else []
        units = ([{'id': 0, 'text': 'Original whole first turn.'}] if query == 'When was that?' else
                 [{'id': 0, 'text': 'Original whole first turn.'}, {'id': 1, 'text': 'Original whole second turn.'}])
        if not self.text_units:
            units = []
        return {'history': [], 'evidence': hits, 'text': '\n\n'.join(unit['text'] for unit in units),
                'text_units': units, 'retrieval': {'selected': [unit['id'] for unit in units]},
                'experience': None}

    def search(self, query, scope, *, expanded):
        self.search_queries.append(query)
        return [{'context_id': 'new', 'title': 'New source', 'text': 'Additional original evidence.'}]

    def observe(self, query, hits, scope):
        for hit in hits:
            self.records.setdefault(hit['context_id'], hit)
        return len(hits)

    def commit(self, *args, **kwargs):
        return {'written': [], 'experience': None, 'experience_writes': 0}


class Client:
    model = 'same-base-model'

    def __init__(self, *, insist=False):
        self.calls, self.retriever_calls = [], 0
        self.insist = insist

    def call(self, role, messages, *, shared_roles, role_control):
        assert shared_roles is True
        assert messages[0]['content'] == role_prompt('continuous', role)
        action = 'deliver'
        if role == 'retriever':
            self.retriever_calls += 1
            if self.retriever_calls == 1 or self.insist and self.retriever_calls == 2:
                action = 'lookup'
            if self.retriever_calls >= 2:
                assert 'A worked at B in 1990.' in messages[1]['content']
        response = {'content': '1990', 'action': action, 'lookup_query':
                    'A worked at B date' if action == 'lookup' else None}
        self.calls.append({'role': role, 'messages': copy.deepcopy(messages), 'response': response,
                          'finish_reason': 'stop', 'input_tokens': 10, 'output_tokens': 2, 'elapsed_ms': 1})
        return response


def workflow(memory, client):
    runtime = UnifiedCollaboration.__new__(UnifiedCollaboration)
    runtime.method, runtime.task_kind, runtime.max_searches = 'structured', 'continuous', 1
    runtime.memory, runtime.client, runtime.resident = memory, client, None
    runtime.transport = CollaborationTransport('structured', client, SimpleNamespace(source_aliases=[]))
    return runtime


def test_resolved_query_recalls_old_memory_then_returns_it_to_role_without_searching_knowledge():
    memory, client = Memory(), Client()
    result = workflow(memory, client).run('When was that?', history_scope='one')
    assert memory.read_queries == ['When was that?', 'A worked at B date']
    assert memory.search_queries == []
    assert result['workflow_complete'] is True
    assert result['retrieval_audit']['memory_source_ids'] == ['old']
    assert result['metrics']['memory_searches'] == 2
    assert result['metrics']['memory_requeries'] == 1
    assert result['metrics']['memory_only_tasks'] == 1
    assert result['memory_read']['reads'][1]['new_source_ids'] == ['old']
    assert client.retriever_calls == 2


def test_nonempty_memory_does_not_override_roles_later_request_for_knowledge():
    memory, client = Memory(), Client(insist=True)
    result = workflow(memory, client).run('When was that?', history_scope='one')
    assert memory.read_queries == ['When was that?', 'A worked at B date']
    assert memory.search_queries == ['A worked at B date']
    assert result['metrics']['memory_searches'] == 2
    assert result['metrics']['knowledge_searches'] == 1
    assert result['metrics']['memory_only_tasks'] == 0
    assert result['metrics']['memory_evidence_delivered'] == 1
    assert client.retriever_calls == 3
    assert set(result['source_evidence']) == {'old', 'new'}


def test_whole_text_units_are_merged_once_without_replacing_first_recall():
    memory, client = Memory(text_units=True), Client()
    result = workflow(memory, client).run('When was that?', history_scope='one')
    assert result['memory_read']['text'] == 'Original whole first turn.\n\nOriginal whole second turn.'
    assert len(result['memory_read']['text_units']) == 2
    assert result['memory_read']['reads'][1]['new_text_unit_ids'] == [1]


def test_memory_query_is_available_even_after_knowledge_budget_is_exhausted():
    memory, client = Memory(), Client()
    runtime = workflow(memory, client)
    runtime.run('When was that?', history_scope='one')
    runtime.memory_queries_checked.clear()
    runtime.memory_evidence.clear()
    runtime.searches = [{'source_ids': ['new']}]
    snapshot = {'task': {'question': 'When was that?'}, 'history': [], 'remembered_text': '',
                'evidence': {}, 'work': [], 'experience': None, 'diagnostics': []}
    assert runtime._lookup(snapshot, 'one', 'reasoner', query='A worked at B date') is True
    assert set(snapshot['evidence']) == {'old'}
    assert memory.search_queries == []


def test_text_unit_accessor_keeps_complete_text_and_original_unit_numbers():
    memory = SimilarityTextMemory(lambda _: [1, 0])
    for text in ('Original first turn.\nAll its relations.', 'Original second turn.'):
        memory.append('one', text, corpus_version='version')
    assert memory.read_units('one', [1, 0]) == [
        {'id': 1, 'text': 'Original second turn.'}, {'id': 0, 'text': 'Original first turn.\nAll its relations.'}]


def test_short_ids_round_trip_every_fact_without_replacing_entity_or_task_values():
    import json
    document, fact_id = 'a' * 64, 'b' * 64
    fact = {'document_id': document, 'subject': document, 'relation': 'reports', 'object': fact_id,
            'polarity': 'negative', 'qualifiers': [{'value': document}], 'sentences': [2, 4]}
    aliases = [{'document_id': document, 'alias': 'Person', 'name': document, 'sentences': [1], 'scoped': True}]
    transport = CollaborationTransport('structured', None, SimpleNamespace(source_aliases=aliases))
    snapshot = {'task': {'question': document, 'inputs': {'checksum': fact_id}},
                'history': [{'question': fact_id, 'answer': document, 'evidence_ids': [fact_id]}],
                'evidence': {fact_id: {'title': 'Original title', 'fact_fields': fact}},
                'work': [], 'diagnostics': []}
    before = copy.deepcopy(snapshot)
    payload = json.loads(transport.render(snapshot))
    packet = payload['evidence']
    reverse = {value: key for key, value in transport.identifiers.items()}
    restored = copy.deepcopy(packet['facts'][0])
    restored[0], restored[6] = reverse[restored[0]], reverse[restored[6]]
    assert restored == full_evidence_packet(snapshot['evidence'])['facts'][0]
    assert packet['facts'][0][1] == document and packet['facts'][0][3] == fact_id
    assert packet['facts'][0][5] == fact['qualifiers']
    assert packet['aliases'][0]['name'] == document
    assert payload['task'] == snapshot['task']
    assert payload['conversation'][0]['question'] == fact_id
    assert payload['conversation'][0]['answer'] == document
    assert reverse[payload['conversation'][0]['evidence_ids'][0]] == fact_id
    assert snapshot == before
    assert transport.packet(snapshot) == packet


def test_common_history_metadata_projection_leaves_text_sources_and_memory_whole():
    original_id = 'c' * 64
    snapshot = {'task': {'question': 'Next question'},
                'history': [{'question': 'Earlier question', 'answer': 'Earlier answer', 'evidence_ids': [original_id]}],
                'evidence': {original_id: {'title': 'Original title', 'text': 'Complete source.\nSecond paragraph.'}},
                'remembered_text': 'Complete memory.\nAll its relations.', 'work': [], 'diagnostics': []}
    transport = CollaborationTransport('plain', None, SimpleNamespace(source_aliases=[]))
    text = transport.render(snapshot)
    assert 'Complete source.\nSecond paragraph.' in text
    assert 'Complete memory.\nAll its relations.' in text
    assert original_id not in text
    assert snapshot['history'][0]['evidence_ids'] == [original_id]
