import importlib
import json
import os
import unittest
from unittest.mock import patch

from competition_agents.state import (
    initial_state,
    put_latent_state,
    received_state_ref,
    received_state_refs,
    record_message,
)
from competition_agents.state_store import clear_state_store
from competition_agents.handoff import delivered_answer, memory_object
from baselines.llmlingua2.llmlingua2_adapter import CompressionResult

reasoner = importlib.import_module('agents.reasoner.agent')


class DeliveredHandoffTests(unittest.TestCase):
    def setUp(self):
        clear_state_store()
        self.env = patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'hash', 'TEXT_COMM_STYLE': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_bytes_are_delivered_bytes_not_pool_size(self):
        for mode in ('text', 'protocol', 'hidden'):
            state = initial_state('question', mode)
            state['metrics']['state_store_bytes'] = 1000000
            payload = 'Evidence: Paris' if mode == 'text' else {'s': ['state-0001'], 'action': 'reason'}
            messages, metrics = record_message(state, sender='retriever', receiver='reasoner', action='reason', payload=payload)
            expected = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            self.assertEqual(messages[-1]['wire_payload'], expected)
            for key in ('protocol_bytes', 'communication_bytes', 'logical_protocol_bytes', 'reference_wire_bytes'):
                self.assertEqual(metrics[key], len(expected.encode('utf-8')))
            self.assertEqual(metrics['state_store_bytes'], 1000000)

    def test_global_ref_is_not_a_delivered_message(self):
        state = initial_state('question', 'protocol')
        state['state_refs'] = {'evidence': 'state-0001'}
        with self.assertRaises(ValueError):
            received_state_ref(state, sender='retriever', receiver='reasoner')

    def test_all_refs_are_read_from_the_delivered_packet_in_order(self):
        state = initial_state('question', 'protocol')
        messages, metrics = record_message(
            state,
            sender='retriever',
            receiver='reasoner',
            action='reason',
            payload={'s': ['evidence-1', 'plan-1']},
        )
        state.update(messages=messages, metrics=metrics)

        self.assertEqual(
            received_state_refs(state, sender='retriever', receiver='reasoner'),
            ['evidence-1', 'plan-1'],
        )
        self.assertEqual(
            received_state_ref(state, sender='retriever', receiver='reasoner'),
            'evidence-1',
        )

    def test_llmlingua_answer_contract_is_part_of_the_counted_wire_payload(self):
        state = initial_state('question', 'text')
        source = 'Answer candidate: North Sea\nReasoning summary: supported'
        compressed = CompressionResult(
            payload='Answer candidate North Sea reasoning supported',
            raw_chars=len(source),
            compressed_chars=45,
            latency_ms=1.0,
            provider='test',
            ratio=0.5,
        )
        with patch.dict(os.environ, {'TEXT_COMM_STYLE': 'llmlingua2'}), \
             patch('competition_agents.state.compress_message', return_value=compressed):
            messages, metrics = record_message(
                state,
                sender='reasoner',
                receiver='summarizer',
                action='summarize',
                payload=source,
            )
        wire = messages[-1]['wire_payload']
        self.assertEqual(delivered_answer(wire), 'North Sea')
        self.assertIn('Answer candidate North Sea', json.loads(wire)['payload'])
        self.assertEqual(messages[-1]['communication_bytes'], len(wire.encode('utf-8')))
        self.assertEqual(metrics['communication_bytes'], len(wire.encode('utf-8')))
        self.assertEqual(metrics['compressed_text_chars'], len(wire))

    def test_autoform_reasoner_handoff_preserves_answer_outside_generated_body(self):
        state = initial_state(
            'Question: Where?',
            'text',
            task_meta={'question': 'Where?', 'context_source': 'hypergraph'},
        )
        reasoning_input = json.dumps(memory_object(['The answer is North Sea.'], [], []))
        messages, metrics = record_message(
            state,
            sender='retriever',
            receiver='reasoner',
            action='reason',
            payload=reasoning_input,
        )
        state.update(messages=messages, metrics=metrics)

        class Client:
            last_generation_metrics = {}

            def reason(self, **kwargs):
                return {
                    'answer_candidate': 'North Sea',
                    'reasoning_summary': 'Supported by evidence',
                    'limitations': [],
                }

        with patch.object(reasoner, 'get_llm_client', return_value=Client()), \
             patch.object(reasoner, 'autoform_enabled', return_value=True), \
             patch.object(
                 reasoner,
                 'generate_autoform_message',
                 return_value=('{"summary":"renamed by AutoForm"}', {}),
             ):
            output = reasoner.reasoner_agent(state)

        wire = output['messages'][-1]['wire_payload']
        self.assertEqual(delivered_answer(wire), 'North Sea')
        self.assertEqual(json.loads(wire)['payload'], '{"summary":"renamed by AutoForm"}')
        self.assertEqual(output['messages'][-1]['communication_bytes'], len(wire.encode('utf-8')))

    def test_reasoner_model_input_is_mode_independent(self):
        calls = []
        for mode in ('text', 'protocol', 'hidden'):
            state = initial_state('Question: where?', mode, task_meta={'question': 'where?', 'context_source': 'hypergraph'})
            state['evidence'] = ['GLOBAL STATE MUST NOT ENTER PROMPT']
            state['plan'] = ['GLOBAL PLAN MUST NOT ENTER PROMPT']
            content = json.dumps(memory_object(
                ['The location is Paris.'],
                [{'history_qa': {'question': 'Which location?', 'answer': 'Earlier location', 'turn': 1}}],
                [{'solution_pattern': 'Verify the location using evidence'}],
            ))
            if mode == 'text':
                payload = content
            else:
                ref, latent, metrics = put_latent_state(state, producer='retriever', topic='question', text=content,
                    consumers=['reasoner'], semantic_payload={'reasoning_input': content, 'evidence': ['Paris']})
                state.update(latent_states=latent, metrics=metrics)
                payload = {'s': [ref]}
            messages, metrics = record_message(state, sender='retriever', receiver='reasoner', action='reason', payload=payload)
            state.update(messages=messages, metrics=metrics)
            class Client:
                def reason(self, **kwargs):
                    calls.append(kwargs)
                    return {'answer_candidate': 'Paris', 'reasoning_summary': 'Evidence agrees', 'limitations': []}
            with patch.object(reasoner, 'get_llm_client', return_value=Client()), patch.object(reasoner, 'put_hidden_state', side_effect=lambda state, **kw: put_latent_state(state, **{k: v for k, v in kw.items() if k not in ('hidden_result', 'hidden_tensor')})):
                output = reasoner.reasoner_agent(state)
            if mode != 'text':
                self.assertEqual(len(output['messages'][-1]['payload']['s']), 1)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[1], calls[2])
        self.assertEqual(calls[0]['evidence'], ['The location is Paris.'])
        self.assertEqual(calls[0]['history_qa'][0]['question'], 'Which location?')
        self.assertEqual(calls[0]['history_qa'][0]['answer'], 'Earlier location')
        self.assertEqual(calls[0]['experience_rules'][0]['solution'], 'Verify the location using evidence')


if __name__ == '__main__':
    unittest.main()
