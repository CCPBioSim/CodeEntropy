import pytest
import numpy
from CodeEntropy import GeometricFunctions as GF

## Test get_sphCoord_axes
# Given a vector does the code calculate the correct spherical basis to use as rotational axes

# The data for this test comes from running CodeEntropy0.3 at the united atom level with 1AKI_ws.tpr and 1AKI_ws.trr from the CodeEntropy0.3/Examples
# Using the value from the first bead in the first frame
def test_get_sphCoord_axes():
    arg_r = [-0.1240921, 0.2787145, -0.0764211] # input vector

    spherical_basis = GF.get_sphCoord_axes(arg_r)

    ref = [[-0.39454843, 0.88616733, -0.24297941],[0.0988289, -0.22197261, -0.97003145],[-0.91354496, -0.40673777, 0.0]]
  
    assert spherical_basis[0] == pytest.approx(ref[0])
    assert spherical_basis[1] == pytest.approx(ref[1])
    assert spherical_basis[2] == pytest.approx(ref[2])

# TODO test for error handling on invalid inputs

