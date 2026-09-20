# Stage Handoff Contract

New runs identify this contract as `structured_history_v3`, with scoring contract
`final_answer_only_v1`.

- Planner retrieves experience before producing JSON containing `query_type`,
  `reference_target`, and brief knowledge retrieval steps. Invalid or truncated
  output is recorded and replaced with a generic retrieval plan.
- Retriever decides whether to retrieve QA history. Empty QA memory is skipped;
  uncertain or conversational queries use conservative recall. Retrieval methods
  retain their own ranking, selected memories, and evidence.
- Retriever sends a `memory_object_v1` object with separate `evidence`,
  `history_qa`, and `experience` lists. QA contains the original question and
  final answer, not a regenerated summary. Historical answers are unverified
  conversational context, not independent factual evidence.
- Transport-specific compression remains enabled. If compressed content is no
  longer structured JSON, Reasoner receives the delivered content, without
  restoring missing fields from global state.
- Summarizer preserves a recoverable received answer. If the received message
  cannot be parsed, answer recovery uses that message only; unrecoverable output
  becomes an empty answer. It never substitutes the global candidate.
- For QA, display, QA writeback, and scoring use `final_answer`. Results also
  retain `reasoner_answer` and `score_answer` for auditing. CodeAct continues to
  evaluate execution results.

Restart the runner before new experiments. Existing result files are not
rewritten. Compare methods on fresh runs with the same model, datasets, and
initial experience-memory conditions. This change does not guarantee improved
accuracy; inspect per-turn answers and retrieval traces before drawing conclusions.

Focused checks (set `PYTHONPATH` to `examples/competition_agents`):

```text
python -m unittest discover -s examples/competition_agents/tests -p test_handoff_contract.py
python -m unittest discover -s examples/competition_agents/tests -p test_delivered_handoff.py
```
