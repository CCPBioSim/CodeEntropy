API Documentation
=================

Main
----
.. autosummary::
    :toctree: autosummary

    CodeEntropy.main.main

Run Manager
------------
.. autosummary::
    :toctree: autosummary

    CodeEntropy.run.RunManager
    CodeEntropy.run.RunManager.create_job_folder
    CodeEntropy.run.RunManager.run_entropy_workflow
    CodeEntropy.run.RunManager.new_U_select_frame
    CodeEntropy.run.RunManager.new_U_select_atom

Level Manager
-------------
.. autosummary::
    :toctree: autosummary

    CodeEntropy.levels.LevelManager
    CodeEntropy.levels.LevelManager.select_levels
    CodeEntropy.levels.LevelManager.get_matrices
    CodeEntropy.levels.LevelManager.get_dihedrals
    CodeEntropy.levels.LevelManager.compute_dihedral_conformations
    CodeEntropy.levels.LevelManager.get_beads
    CodeEntropy.levels.LevelManager.get_axes
    CodeEntropy.levels.LevelManager.get_avg_pos
    CodeEntropy.levels.LevelManager.get_sphCoord_axes
    CodeEntropy.levels.LevelManager.get_weighted_forces
    CodeEntropy.levels.LevelManager.get_weighted_torques
    CodeEntropy.levels.LevelManager.create_submatrix
    CodeEntropy.levels.LevelManager.build_covariance_matrices
    CodeEntropy.levels.LevelManager.update_force_torque_matrices
    CodeEntropy.levels.LevelManager.filter_zero_rows_columns
    CodeEntropy.levels.LevelManager.build_conformational_states

Entropy Manager
---------------
.. autosummary::
   :toctree: autosummary

   CodeEntropy.entropy.EntropyManager
   CodeEntropy.entropy.EntropyManager.execute

Vibrational Entropy
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.entropy.VibrationalEntropy
   CodeEntropy.entropy.VibrationalEntropy.frequency_calculation
   CodeEntropy.entropy.VibrationalEntropy.vibrational_entropy_calculation
   

Conformational Entropy
^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.entropy.ConformationalEntropy
   CodeEntropy.entropy.ConformationalEntropy.assign_conformation
   CodeEntropy.entropy.ConformationalEntropy.conformational_entropy_calculation

