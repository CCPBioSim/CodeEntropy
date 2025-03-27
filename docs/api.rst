API Documentation
=================

Main
----
.. autosummary::
    :toctree: autosummary

    CodeEntropy.main_mcc.create_job_folder
    CodeEntropy.main_mcc.main

Calculations
------------

Conformation Functions
^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.calculations.ConformationFunctions.assign_conformation

Entropy Functions
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.calculations.EntropyFunctions.frequency_calculation
   CodeEntropy.calculations.EntropyFunctions.vibrational_entropy
   CodeEntropy.calculations.EntropyFunctions.conformational_entropy
   CodeEntropy.calculations.EntropyFunctions.orientational_entropy

Geometric Functions
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.calculations.GeometricFunctions.get_beads
   CodeEntropy.calculations.GeometricFunctions.get_axes
   CodeEntropy.calculations.GeometricFunctions.get_avg_pos
   CodeEntropy.calculations.GeometricFunctions.get_sphCoord_axes
   CodeEntropy.calculations.GeometricFunctions.get_weighted_forces
   CodeEntropy.calculations.GeometricFunctions.get_weighted_torques
   CodeEntropy.calculations.GeometricFunctions.create_submatrix
   CodeEntropy.calculations.GeometricFunctions.filter_zero_rows_columns

Level Functions
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.calculations.LevelFunctions.select_levels
   CodeEntropy.calculations.LevelFunctions.get_matrices
   CodeEntropy.calculations.LevelFunctions.get_dihedrals

Config
------
.. autosummary::
   :toctree: autosummary
   :recursive:

   CodeEntropy.config

