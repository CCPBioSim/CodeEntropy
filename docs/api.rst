API Documentation
=================

MDAnalysis Helper
-------------------
.. autosummary::
   :toctree: autosummary

   CodeEntropy.IO.MDAUniverseHelper.new_U_select_frame
   CodeEntropy.IO.MDAUniverseHelper.new_U_select_atom
   CodeEntropy.IO.MDAUniverseHelper.write_universe
   CodeEntropy.IO.MDAUniverseHelper.read_universe

Solute
-------

Import Data
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.ClassCollection.DataContainer.DataContainer

Solute Entropy Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whole Molecule Level
""""""""""""""""""""""""

.. autosummary::
   :toctree: autosummary

   CodeEntropy.FunctionCollection.EntropyFunctions.compute_entropy_whole_molecule_level

Residue Level
""""""""""""""""""""

.. autosummary::
   :toctree: autosummary
   
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_entropy_residue_level
   
United-Atom Level
"""""""""""""""""""

.. autosummary::
   :toctree: autosummary
      
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_entropy_UA_level
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_entropy_UA_level_multiprocess

Topographical Level
""""""""""""""""""""""""
.. autosummary::
   :toctree: autosummary
   
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_topographical_entropy0_SC
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_topographical_entropy0_BB
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_topographical_entropy1_SC
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_topographical_entropy1_BB
   CodeEntropy.FunctionCollection.EntropyFunctions.compute_topographical_entropy_AEM

Solvent
--------

Import Data
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: autosummary

   CodeEntropy.ClassCollection.PoseidonClass.Poseidon

Run Analysis
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: autosummary

   CodeEntropy.ClassCollection.PoseidonClass.Poseidon.run_analysis