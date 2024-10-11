import pytest
import numpy
from CodeEntropy import EntropyFunctions as EF

## Test vibrational_entropy
# Given a matrix does the code calculate the correct entropy value

# test for matrix_type force, highest level=yes
def test_vibrational_entropy_polymer_force():
    matrix = numpy.array([[4.67476, -0.04069, -0.19714],[-0.04069, 3.86300, -0.17922],[-0.19714, -0.17922, 3.66307]])
    matrix_type = "force"
    temp = 298
    highest_level = "yes"

    S_vib = EF.vibrational_entropy(matrix, matrix_type, temp, highest_level)

    assert S_vib == pytest.approx(52.88123410327823)

# test for matrix_type force, highest level=no

# test for matrix_type torque
def test_vibrational_entropy_polymer_torque():
    matrix = numpy.array([[6.69611, 0.39754, 0.57763],[0.39754, 4.63265, 0.38648],[0.57763, 0.38648, 6.34589]])
    matrix_type = "torque"
    temp = 298
    highest_level = "yes"

    S_vib = EF.vibrational_entropy(matrix, matrix_type, temp, highest_level)

    assert S_vib == pytest.approx(48.45003266069881)

# TODO test for error handling on invalid inputs

