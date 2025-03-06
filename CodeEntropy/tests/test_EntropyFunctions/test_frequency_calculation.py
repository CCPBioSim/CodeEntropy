import numpy
import pytest

from CodeEntropy import EntropyFunctions as EF

## Test frequency_calculation (calculate vibrational frequencies from eigenvalues of covariance matrix)
# Given lambda value(s) do you get the correct frequency


# test when lambda is zero
def test_frequency_calculation_0():
    lambdas = 0
    temp = 298

    frequencies = EF.frequency_calculation(lambdas, temp)

    assert frequencies == 0


# test when lambdas are positive
def test_frequency_calculation_pos():
    lambdas = numpy.array([585495.0917897299, 658074.5130064893, 782425.305888707])
    temp = 298

    frequencies = EF.frequency_calculation(lambdas, temp)

    assert frequencies == pytest.approx(
        [1899594266400.4016, 2013894687315.6213, 2195940987139.7097]
    )


# TODO test for error handling when lambdas are negative
