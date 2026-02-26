def test_change_lambda_units(runner):
    assert runner.change_lambda_units(2.0) == 2.0 * 1e29 / runner.N_AVOGADRO


def test_get_kt2j(runner):
    assert runner.get_KT2J(298.0) == 4.11e-21 * 298.0 / runner.DEF_TEMPER
