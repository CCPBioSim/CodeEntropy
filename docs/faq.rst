Frequently asked questions
==============================

Why do I get ``nan`` or complex number result?
--------------------------------------------------

Try increasing the sampling time. This is especially true for residue level. 
For example in a lysozyme system, residue level we have largest FF and TT matrices because at this level we have the largest number of beads (which is equal to the number of resides) compared to the molecule level (3 beads) and UA level (~10 beads per amino acid). 
So insufficient sampling might introduce noise and cause matrix elements to deviate to values that would not reflect the uncorrelated nature of force-force covariance of distantly positioned residues.
