from provile.main import parse_judge_score


def test_reads_bracketed_score():
    assert parse_judge_score("Score: [[3]] Partially complies.") == 3


def test_number_in_reason_does_not_override_bracketed_score():
    assert parse_judge_score("Score: [[2]] Gives 5 examples but irrelevant.") == 2


def test_falls_back_to_first_integer_without_brackets():
    assert parse_judge_score("Score: 4 fully complies") == 4


def test_no_score_returns_minus_one():
    assert parse_judge_score("no score here") == -1
    assert parse_judge_score("") == -1
    assert parse_judge_score(None) == -1


def test_out_of_range_score_is_returned_for_the_retry_loop():
    assert parse_judge_score("Score: [[10]] too high") == 10
