import importlib
import json
import os
import unittest
from unittest.mock import patch

from competition_agents.handoff import (
    decide_qa_retrieval, decode_memory_object, delivered_answer,
    encode_answer_handoff, historical_qa, memory_object, parse_planner_output,
)
from competition_agents.state import initial_state
from run_demo import prediction_from_result

summarizer = importlib.import_module("agents.summarizer.agent")
planner = importlib.import_module("agents.planner.agent")


class HandoffContractTests(unittest.TestCase):
    def test_planner_retrieves_experience_before_json_planning(self):
        from competition_agents.llm.mock import MockLLMClient

        sequence = []

        class Client(MockLLMClient):
            def plan_query(self, task):
                sequence.append("plan")
                self.prompt = task
                return super().plan_query(task)

        client = Client()
        def search(*args, **kwargs):
            sequence.append("experience")
            self.assertEqual(kwargs["memory_layers"], ["experience"])
            return [{"memory_id": "exp-test", "solution_pattern": "Inspect previous questions"}]

        state = initial_state("Question: What about its designer?", "text", task_meta={"question": "What about its designer?"})
        with patch.dict(os.environ, {"TEXT_COMM_STYLE": ""}), \
             patch.object(planner, "search_memory", side_effect=search), \
             patch.object(planner, "get_llm_client", return_value=client):
            result = planner.planner_agent(state)
        self.assertEqual(sequence, ["experience", "plan"])
        self.assertIn("Inspect previous questions", client.prompt)
        self.assertEqual(result["retrieval_policy"]["decision_source"], "retriever_pending")
        self.assertNotIn("need_qa_memory", result["retrieval_policy"])

    def test_original_qa_is_not_truncated_or_mixed_with_evidence(self):
        answer = "original answer " * 100
        hit = {"memory_id": "qa-1", "summary": answer,
               "trace": {"task_meta": {"question": "Original question?", "turn_id": "turn-1"}}}
        obj = memory_object(["External source"], [hit], [{"solution_pattern": "Check sources"}])
        decoded = decode_memory_object(json.dumps(obj))
        self.assertEqual(decoded["history_qa"][0]["answer"], answer)
        self.assertEqual(decoded["history_qa"][0]["question"], "Original question?")
        self.assertEqual(decoded["history_qa"][0]["verification_status"], "unverified")
        self.assertEqual(decoded["evidence"][0]["text"], "External source")
        self.assertEqual(decoded["experience"][0]["solution"], "Check sources")

    def test_saved_empty_answer_does_not_resurrect_summary(self):
        qa = historical_qa({"history_qa": {"answer": "", "question": "Q"}, "summary": "old"})
        self.assertEqual(qa["answer"], "")

    def test_planner_valid_invalid_and_truncated_output(self):
        raw = json.dumps({"query_type": "standalone", "reference_target": "", "steps": ["Retrieve the relevant evidence"]})
        self.assertEqual(parse_planner_output(raw)[1], "")
        self.assertEqual(parse_planner_output("Retrieve")[1], "invalid_json_contract")
        self.assertEqual(parse_planner_output(raw, {"output_tokens": 128, "max_new_tokens": 128})[1], "")

    def test_planner_keeps_complete_partial_contract_and_ignores_trailing_text(self):
        raw = (
            '```json\n{"query_type":"follow_up_fact",'
            '"steps":["Retrieve information about U2 origin"]}\n```\n'
            "Additional text that reached the token limit"
        )
        parsed, reason = parse_planner_output(
            raw,
            {"output_tokens": 48, "max_new_tokens": 48},
        )
        self.assertEqual(reason, "")
        self.assertEqual(parsed["reference_target"], "")
        self.assertEqual(parsed["steps"], ["Retrieve information about U2 origin"])

    def test_retriever_owns_gate_and_handles_empty_history(self):
        policy = {"query_type": "standalone", "reference_target": ""}
        self.assertTrue(decide_qa_retrieval(policy, "What about its designer?", True)["need_qa_memory"])
        self.assertFalse(decide_qa_retrieval(policy, "What about its designer?", False)["need_qa_memory"])
        self.assertFalse(decide_qa_retrieval(policy, "When was Example founded?", True)["need_qa_memory"])
        self.assertTrue(decide_qa_retrieval({}, "A follow-up?", True)["need_qa_memory"])

    def test_score_uses_final_not_candidate(self):
        result = {"final_answer": "delivered", "execution_result": {"answer_candidate": "original"}}
        self.assertEqual(prediction_from_result(result), "delivered")
        result["final_answer"] = ""
        self.assertEqual(prediction_from_result(result), "")

    def test_delivery_parser_preserves_empty_and_rejects_unlabelled(self):
        self.assertEqual(delivered_answer('{"final_answer":""}'), "")
        self.assertEqual(delivered_answer("Answer candidate: Alpha\nReasoning summary: detail"), "Alpha")
        self.assertIsNone(delivered_answer("Incomplete message"))

    def test_answer_handoff_preserves_candidate_with_arbitrary_body(self):
        wire = encode_answer_handoff(
            "North Sea",
            '{"summary":"AutoForm chose a different field name"}',
            "autoform",
        )
        self.assertEqual(delivered_answer(wire), "North Sea")
        self.assertEqual(json.loads(wire)["payload"], '{"summary":"AutoForm chose a different field name"}')

    def test_summarizer_display_write_and_score_match(self):
        state = initial_state("Question: Q", "text", task_meta={
            "dataset_type": "continuous", "question": "Q", "group_id": "test", "turn_id": "turn-1"})
        state["execution_result"] = {"answer_candidate": "GLOBAL ANSWER"}
        for answer in ("Received answer", ""):
            state["messages"] = [{"from": "reasoner", "to": "summarizer",
                                  "payload": json.dumps({"answer_candidate": answer})}]
            writes = []

            def add_memory(memory, **kwargs):
                writes.append(kwargs)
                return [{"memory_id": "qa-test", "summary": kwargs["summary"]}]

            with patch.object(summarizer, "add_memory", side_effect=add_memory), \
                 patch.object(summarizer, "get_llm_client") as client:
                result = summarizer.summarizer_agent(state)
            client.assert_not_called()
            self.assertEqual(result["final_answer"], answer)
            self.assertEqual(writes[0]["summary"], answer)
            self.assertEqual(writes[0]["trace"]["execution_result"]["final_answer"], answer)
            self.assertEqual(prediction_from_result(result), answer)

    def test_unrecoverable_message_does_not_use_global_candidate(self):
        state = initial_state("Question: Q", "text", task_meta={"dataset_type": "continuous"})
        state["execution_result"] = {"answer_candidate": "GLOBAL ANSWER"}
        state["messages"] = [{"from": "reasoner", "to": "summarizer", "payload": "broken handoff"}]
        with patch.object(summarizer, "add_memory", return_value=[{"memory_id": "qa-test"}]), \
             patch.object(summarizer, "get_llm_client") as factory:
            factory.return_value.generate.return_value = "invalid recovery"
            factory.return_value.last_generation_metrics = {}
            result = summarizer.summarizer_agent(state)
        self.assertEqual(result["final_answer"], "")
        self.assertNotIn("GLOBAL ANSWER", factory.return_value.generate.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
