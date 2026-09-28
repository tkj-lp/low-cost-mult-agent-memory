from competition_agents.answer_evaluation import answer_coverage


def test_supported_extra_positions_do_not_reduce_coverage():
    result = answer_coverage('Ambassador to Ghana and Chief of Protocol of the United States', 'Chief of Protocol')
    assert result['correct'] and result['score'] == 1


def test_missing_part_gets_partial_credit():
    result = answer_coverage('Mira', 'Mira Vale')
    assert result['status'] == 'partial' and result['score'] == .5


def test_wrong_entity_assignment_is_not_rescued_by_word_overlap():
    result = answer_coverage('Ben was Chief of Protocol', 'Chief of Protocol', contradiction=True)
    assert result['coverage'] == 1 and result['score'] == 0 and not result['correct']


def test_unicode_and_alternative_reference_forms():
    assert answer_coverage('她曾任礼宾司司长和大使', '礼宾司司长')['correct']
    assert answer_coverage('The USA', ['United States', 'USA'])['correct']
    assert answer_coverage('unknown', 'Mira Vale')['score'] == 0
