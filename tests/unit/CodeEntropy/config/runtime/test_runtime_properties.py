def test_n_avogadro_property_returns_internal_value(runner):
    # uses the runner fixture (tmp folder)
    assert runner.N_AVOGADRO == runner._N_AVOGADRO


def test_def_temper_property_returns_internal_value(runner):
    assert runner.DEF_TEMPER == runner._DEF_TEMPER
