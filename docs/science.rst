Multiscale Cell Correlation Theory
==================================

This section is to describe the scientific theory behind the method used in CodeEntropy.

The multiscale cell correlation (MCC) method [1-3] has been developed in the group of Richard Henchman to calculate entropy from molecular dynamics (MD) simulations.
It has been applied to liquids [1,3,4], proteins [2,5,6], solutions [6,8-10], and complexes [6,8].
The purpose of this project is to develop and release well written code that enables users from any group to calculate the entropy from their simulations using the MCC.
The latest code can be found at github.com/ccpbiosim/codeentropy.

The method requires forces to be written to the MD trajectory files along with the coordinates.

Key References
--------------
[1] Richard H. Henchman. "Free energy of liquid water from a computer simulation via cell theory". In: Journal of Chemical Physics 126 (2007), 064504.

[2] Ulf Hensen, Frauke Grater, and Richard H. Henchman. “Macromolecular
Entropy Can Be Accurately Computed from Force”. In: Journal of Chemical Theory and Computation 10 (2014), pp. 4777–4781.

[3] Jonathan Higham et al. “Entropy of Flexible Liquids from Hierarchical
Force-Torque Covariance and Coordination”. In: Molecular Physics 116
(2018), pp. 1965–1976.

[4] Hafiz Saqib Ali, Jonathan Higham, and Richard H. Henchman. “Entropy
of Simulated Liquids Using Multiscale Cell Correlation”. In: Entropy 21
(2019), p. 750.

[5] Arghya Chakravorty, Jonathan Higham, and Richard H. Henchman. “En-
tropy of Proteins Using Multiscale Cell Correlation”. In: J. Chem. Inf.
Model. 60 (2020), pp. 5540–5551.

[6] Jas Kalayan et al. “Total Free Energy Analysis of Fully Hydrated Proteins”.
In: Proteins 91 (2023), pp. 74–90.

[7] Jonathan Higham and Richard H. Henchman. "Locally adaptive method to define coordination shell". In: J. Chem. Phys. 145 (2016), pp. 084108

Additional application examples
-------------------------------
[8] Hafiz Saqib Ali et al."Energy-entropy method using Multiscale Cell Correlation to calculate binding free energies in the SAMPL8 Host-Guest Challenge". In: Journal of Computer Aided Molecular Design 35 (2021), 911-921.

[9] Fabio Falcioni et al. "Energy-entropy prediction of octanol-water logP of SAMPL7 N-acylsulfonamide bioisosters". In Journal of Computer Aided Molecular Design 35 (2021) 831-840.

[10] Hafiz Saqib Ali et al. "Energy-entropy Multiscale Cell Correlation method to predict toluene–water log P in the SAMPL9 challenge". In Physical Chemistry Chemical Physics 25 (2023), 27524-27531.

Hierarchy
---------

Atoms are grouped into beads.
The levels refer to the size of the beads and the different entropy terms are calculated at each level, taking care to avoid over counting.
This is done at three different levels of the hierarchy - united atom, residues, and polymers.
Not all molecules have all the levels of hierarchy, for example water has only the united atom level, benzene would have united atoms and residue, and a protein would have all three levels.

Vibrational Entropy
-------------------

The vibrational entropy is calculated using the force covariance matrix for the translational contributions to entropy and using the torque covariance matrix for the rotational contributions.
The eigenvalues of the covariance matrices are use to calculate the frequencies.

.. math::
   \nu_i = \frac{1}{2\pi} \sqrt{\frac{\lambda_i}{k_BT}}

Then the frequencies are used in the quantum harmonic oscillator equation to calculate the vibrational entropy.

.. math::
   S_{\mathrm{vib}} = k_B \sum_{i=1}^{3N} \left( \frac{\hbar\nu_i/k_BT}{e^{\hbar\nu_i/k_BT}-1} - \ln\left(1-e^{-\hbar\nu_i/k_BT}\right)\right)


Forces and torques on each bead are transformed into the bead's local coordinate frame at every time step to ensure that anisotropy in each direction is captured and not averaged over.
The axes for this transformation are calculated for each bead in each time step.

For the polymer level, the translational and rotational axes are defined as the principal axes of the molecule.

For the residue level, there are two situations.
When the residue is not bonded to any other residues, the translational and rotational axes are the principal axes of the molecule.
When the residue is part of a larger polymer, the translational axes are the principal axes of the polymer, and the rotational axes are defined from the average position of the bonds to neighbouring residues.

For the united atom level, the translational axes are defined as the principal axes of the residue and the rotational axes are defined from the average position of the bonds to neighbouring heavy atoms.
If there are no bonds to other heavy atoms, the principal axes of the molecule are used.

Conformational Entropy
----------------------

This term is based on the intramolecular conformational states.

The united atom level dihedrals are defined for every linear sequence of four bonded atoms, but only using the heavy atoms no hydrogens are involved.
The MDAnalysis package is used to identify and calculate the united atom dihedral values.

For the residue level dihedrals, the bond between the first and second residues and the bond between the third and fourth residues are found.
The four atoms at the ends of these two bonds are used as points for the dihedral angle calculation.

To discretise dihedrals, a histogram is constructed from each set of dihedral values and peaks are identified.
Then at each timestep, every dihedral is assigned to its nearest peak and a state is created from all the assigned peaks in the residue (for united atom level) or molecule (for residue level).
Once the states are defined, the probability of finding the residue or molecule in each state is calculated.
Then the Boltzmann equation is used to calculate the entropy:

.. math::
   S_{\mathrm{conf}} = - k_B \sum_{i=1}^{N_{\mathrm{conf}}}p_i\ln{p_i}

Orientational Entropy
---------------------
Orientational entropy is the term that comes from the molecule's environment (or the intermolecular configuration).
The different environments are the different states for the molecule, and the statistics can be used to calculate the entropy.

The number of orientations :math:`\Omega_{\mathrm{orient}}` relates to the number of neighbors.
We are using the relative angular distance (RAD) method for identifying neighbours [7].
This method considers a molecule j as part of the coordination shell of the central molecule i, if for all other molecules k:

.. math::
   \frac{1}{r_{ij}^2} > \frac{1}{r_{ik}^2} \mathrm{cos} \theta_{jik}

where, :math:`r_{ij}` is the distance between i and j and :math:`\theta_{jik}` is the angle between j, i, and k (with i at the vertex).
The MDAnalysis NeighborSearch method can be used as an alternative to RAD, but the grid based search relies on an arbitrary cutoff.

The number of orientations also depends on the symmetry number of the molecule and if it is linear.
Linear molecules have 2 rotational degrees of freedom and non-linear molecules have 3 rotational degrees of freedom.
CodeEntropy is using the united atom beads to determine if a molecule is treated as the linear case (for example, carbon dioxide and methanol are both considered linear).

.. math::
   \Omega_{\mathrm{orient}} = max \left\{ 1, N^{3/2} \pi^{1/2} / \sigma \right\}

.. math::
   \Omega_{\mathrm{orient,linear}} = max \left\{ 1, N / \sigma \right\}

where N is the number of neighbours the molecule has averaged over the number of frames and :math:`\sigma` is the symmetry number of the molecule.
The max of 1 or the number of neighbours expression prevents the resulting orientational entropy value being less than zero.

.. math::
   S_{\mathrm{orient}} = R \ln{ \Omega_{\mathrm{orient}} }

where R is the gas constant.
