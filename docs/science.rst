Multiscale Cell Correlation Theory
==================================

This section is to describe the scientific theory behind the methods used in CodeEntropy.

The multiscale cell correlation (MCC) method [1,2] has been developed in the group of Richard Henchman to calculate entropy from molecular dynamics (MD) simulations. It has been applied to liquids [3], proteins [4], and water [5].
The purpose of this project is to develop a unified, well written and well tested code that would enable users from any group to calculate the entropy from their simulations using the MCC method. The latest code can be found at github.com/ccpbiosim/codeentropy.

The method requires forces to be written to the MD trajectory files along with the coordinates.

Key References
--------------
[1] Ulf Hensen, Frauke Grater, and Richard H. Henchman. “Macromolecular
Entropy Can Be Accurately Computed from Force”. In: Journal of Chemi-
cal Theory and Computation 10 (2014), pp. 4777–4781.

[2] Jonathan Higham et al. “Entropy of Flexible Liquids from Hierarchical
Force-Torque Covariance and Coordination”. In: Molecular Physics 116
(2018), pp. 1965–1976.

[3] Hafiz Saqib Ali, Jonathan Higham, and Richard H. Henchman. “Entropy
of Simulated Liquids Using Multiscale Cell Correlation”. In: Entropy 21
(2019), p. 750.

[4] Arghya Chakravorty, Jonathan Higham, and Richard H. Henchman. “En-
tropy of Proteins Using Multiscale Cell Correlation”. In: J. Chem. Inf.
Model. 60 (2020), pp. 5540–5551.

[5] Jas Kalayan et al. “Total Free Energy Analysis of Fully Hydrated Proteins”.
In: Proteins 91 (2023), pp. 74–90.


Hierarchy
---------
   
Atoms are grouped into beads. 
The levels refer to the size of the beads and the different entropy terms are calculated at each level, taking care to avoid over counting.
This is done at three different levels of the hierarchy - united atom, residues, and polymers. Not all molecules have all the levels of hierarchy, for example water has only the united atom level, benzene would have united atoms and residue, and a protein would have all three levels.

Vibrational Entropy
-------------------

The vibrational entropy is calculated using the force covariance matrix for the translational contributions to entropy and using the torque covariance matrix for the rotational contributions.
The eigenvalues of the covariance matrices are use to calculate the frequencies.

.. math::
   \nu_i = \frac{1}{2\pi} \sqrt{\frac{\lambda_i}{k_BT}}

Then the frequencies are used in the quantum harmonic oscillator equation to calculate the vibrational entropy.

.. math::
   S_{\mathrm{vib}} = k_B \sum_{i=1}^{3N} \left( \frac{\hbar\nu_i/k_BT}{e^{\hbar\nu_i/k_BT}-1} - \ln\left(1-e^{-\hbar\nu_i/k_BT}\right)\right)

Why Forces and Torques?
^^^^^^^^^^^^^^^^^^^^^^^


Axes
^^^^
It is important that the forces and torques are transformed into local coordinate systems, so that the covariance matrices represent the motions within the molecule not the diffusion of the molecule through the simulation box. The axes for this transformation are calculated for each bead in each time step.

For the polymer level, the translational and rotational axes are defined as the principal axes of the molecule.

For the residue level, there are two situations. When the residue is not bonded to any other residues, the translational and rotational axes are defined as the principal axes of the residue. When the residue is part of a larger polymer, the translational axes are defined as the principal axes of the polymer, and the rotational axes are defined from the average position of the bonds to neighbouring residues.

For the united atom level, the translational axes are defined as the principal axes of the residue and the rotational axes are defined from the average position of the bonds to neighbouring heavy atoms.

Conformational Entropy
----------------------

This is a topographical term based on the intramolecular conformational states.

Defining dihedrals
^^^^^^^^^^^^^^^^^^
The united atom level dihedrals are defined as a chemist would expect, but only using the heavy atoms no hydrogens are involved. 
The MDAnalysis package is used to find the united atom level dihedrals and calculate all the dihedral values.

For the residue level dihedrals, the bond between the first and second residues and the bond between the third and fourth residues are found. The four atoms at the ends of these two bonds are used as points for the dihedral angle calculation.

For each dihedral, the set of values from the trajectory frames is used to create histograms and identify peaks. Then at each frame, the dihedral is assigned to its nearest peak and a state is created from the peaks of every dihedral in the residue (for united atom level) or molecule (for residue level).

From conformation to entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the states are defined, the probability of finding the residue or molecule in each state is calculated.
Then the Boltzmann equation is used to calculate the entropy.

.. math::
   S_{\mathrm{conf}} = - k_B \sum_{i=1}^{N_{\mathrm{conf}}}p_i\ln{p_i}

Orientational Entropy
---------------------

This is the second topographical entropy term.
Orientational entropy is the term that comes from the molecule's environment. The different environments are the different states for the molecule, and the statistics can be used to calculate the entropy.
The simplest part is counting the number of neighbours, but symmetry should be accounted for in determining the number of orientations.

For water, the hydrogen bonds are very important and the number of hydrogen bond donors and acceptors in the shell around the water molecule affects the number of unique orientations.
