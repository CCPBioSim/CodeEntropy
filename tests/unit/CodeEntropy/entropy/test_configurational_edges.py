from CodeEntropy.entropy.configurational import ConformationalEntropy


def test_to_1d_array_returns_none_for_non_iterable_state_input():
    ce = ConformationalEntropy()
    # int is not iterable -> list(states) raises TypeError -> returns None
    assert ce._to_1d_array(123) is None


def test_to_1d_array_returns_none_when_states_is_none():
    ce = ConformationalEntropy()
    assert ce._to_1d_array(None) is None
