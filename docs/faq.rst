Frequently asked questions
==============================

Why do I get a ``WARNING`` about invalid eigenvalues?
-----------------------------------------------------

Insufficient sampling might introduce noise and cause matrix elements to deviate to values that would not reflect the uncorrelated nature of force-force covariance of distantly positioned residues.Try increasing the sampling time. This is especially true at the residue level. 

For example in a lysozyme system, the residue level contains the largest force and torque covariance matrices because at this level we have the largest number of beads (which is equal to the number of residues in a protein) compared to the molecule level (3 beads) and united-atom level (~10 beads per amino acid). 

