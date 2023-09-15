import MDAnalysis as mda
tprfile ="data/molecules.prmtop"
trrfile="data/data.trr"
u=mda.Universe(tprfile,trrfile)
