from competition_agents.evidence_memory import SessionEvidenceMemory
from test_field_collaboration import fact


def test_current_relation_is_not_crowded_out_by_previous_answer():
    memory = SessionEvidenceMemory('protocol', None)
    count = fact('count', 'Six singles', 'were released from', 'the album', title='River')
    distractors = [fact(str(i), 'River', 'received award', 'Prize ' + str(i)) for i in range(30)]
    try:
        memory.remember('What awards did River receive?', 'River received award Prize ' * 20,
                        distractors[:2], 'same', observed_evidence=[*distractors, count])
        history, hits, query = memory.context('How many singles were released from the album?', 'same')
        assert history
        assert hits[0]['context_id'] == 'count'
        assert hits[0]['from_memory'] and hits[0]['memory_turns'] == [1]
        assert memory.context('How many singles were released from the album?', 'other')[1] == []
    finally:
        memory.close()


def test_history_reuses_observed_facts_without_model_source_list():
    memory = SessionEvidenceMemory('protocol', None)
    rows = [fact('owner', 'River', 'written by', 'Mara'),
            fact('date', 'River', 'released in', '2001')]
    try:
        memory.remember('Who wrote River?', 'Mara', [], 'one', observed_evidence=rows)
        assert {h['context_id'] for h in memory.context('What about it?', 'one')[1]} == {'owner', 'date'}
        assert not memory.context('What about it?', 'two')[1]
    finally:
        memory.close()


def test_recall_does_not_cut_related_facts_by_id_or_six_row_limit():
    memory = SessionEvidenceMemory('protocol', None)
    rows = [fact(str(i), 'Mara', 'held position', 'Position ' + str(i)) for i in range(75)]
    try:
        memory.store_evidence('Mara', rows, 'one')
        hits = memory.recall('What positions did Mara hold?', scope='one')
        assert {h['context_id'] for h in hits} == {r['context_id'] for r in rows}
    finally:
        memory.close()
