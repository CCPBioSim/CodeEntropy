Frequently asked questions
==============================
What are FF ant TT entropy?
---------------------------------
FF and TT stand for Force-Force and Torque-Torque, respectively. But they really are the translational and rotational components of vibration entropy at a given level of hierarchy, also respectively. The former is computed by diagonalizing a force-force covariance matrix <F\ :sub:`i` • F\ :sub:`j``> and latter using a torque-torque covariance matrix <T\ :sub:`i` • T\ :sub:`j`> - each normalized separately using masses and inertias, respectively, to have the same dimensions. 

Why do I get ``nan`` or complex number result?
--------------------------------------------------

Try increasing the sampling time. This is especially true for residue level. For example in a lysozyme system, residue level we have largest FF and TT matrices because at this level we have the largest number of cells/beads (which is equal to the number of resides) compared to the molecule level (3 beads) and UA level (~10 beads per amino acid). So insufficient sampling might introduce noise and cause matrix elements to deviate to values that would not reflect the uncorrelated nature of force-force covariance of distantly positioned residues.