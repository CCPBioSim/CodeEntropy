CodeEntropy
==============================
[//]: # (Badges)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CodeEntropy/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CodeEntropy/branch/master)


CodeEntropy tool with POSEIDON code integrated to form a complete and generally applicable set of tools for calculating entropy

## To run
### Requirements
- Python > 3.6
- gcc
- g++

### Install via
```
pip install .
```

### Command-line tools
A quick and easy way to get started is to use the command-line tool which you can run in bash by simply typing `CodeEntropyPoseidon`
#### For help
```
CodeEntropyPoseidon -h
```
#### Arguments
| Arguments  | Description | Default | type|
| ------------- | ------------- |----------- |--------------|
| `-f`, `--top_traj_file`  | Path to Structure/topology file (AMBER PRMTOP or GROMACS TPR) followed by Trajectory file(s)  | Requires either `--top_traj_file` or `--pickle`  | list of `str` |
| `-i`, `--pickle`  | Pickled MDAnalyiss.Universe for unsupported format, refer to `create_new_universe.py` for how to load unsuppported data into a universe  | Requires either `--top_traj_file` or `--pickle`  | `str` |
| `-l`, `--selectString`  | Selection string for CodeEntropy such as protein or resid, refer to `MDAnalysis.select_atoms` for more information. | `"all"`: select all atom in trajectory for CodeEntropy analysis for trajectory without solvent  | `str` |
| `-b`, `--begin`  | Start analysing the trajectory from this frame index. | `0`: From begining | `int` |
| `-e`, `--end`  | Stop analysing the trajectory at this frame index | `-1`: end of trajectory | `int` |
| `-d`, `--step`  | Stop analysing the trajectory at this frame index | `1` | `int` |
| `-k`, `--tempra`  | Temperature for entropy calculation (K) | `298.0` | `float` |
| `-t`, `--thread`  | How many multiprocess to use. | `1`: for single core execution | `int` |
| `-o`, `--out`  | Name of the file where the text format output will be written. | `outfile.out` | `str` |
| `-v`, `--csvout`  | Name of the file where the total entropy output will be written. | `outfile.out` | `str` |
| `-r`, `--resout`  | Name of the file where the residue entropy output will be written. | `res_outfile.csv` | `str` |
| `-m`, `--mout`  | Name of the file where certain matrices will be written. | `None` | `str` |
| `-n`, `--nmd`  | Name of the file where VMD compatible NMD format files with mode information will be printed. | `None` | `str` |
| `-a`, `--rotationalaxis`  | The 3 atom name in each residue for rotational axis. | `['C', 'CA', 'N']` : for protein | list of `str` |
| `-c`, `--cutShell`  | Include cutoff shell analysis, add cutoff distance in angstrom. | `None` : will ust the RAD Algorithm. See Higham, Jonathan, and Richard H Henchman. “Locally adaptive method to define coordination shell.” The Journal of chemical physics vol. 145,8 (2016): 084108. doi:10.1063/1.4961439 | list of `str` |
| `-p`, `--pureAtomNum`  | Reference molecule resid for system of pure liquid. | `1` | `int` |
| `-x`, `--excludedResnames`  | Exclude a list of molecule names from nearest non-like analysis. | `None` | list of `str` |
| `-w`, `--water`  | Resname for water molecules.  | `WAT` | list of `str` |
| `-s`, `--solvent`  | Include resname of solvent molecules (case-sensitive).  | `None` | list of `str` |
| `--wm`  | Do entropy calculation at whole molecule level (The whole molecule is treated as one single bead.).  | Flag, activate when included | Flag |
| `--res`  | Do entropy calculation at residue level (A residue as a whole represents a bead.).  | Flag, activate when included | Flag |
| `--uatom`  | Do entropy calculation at united atom level (A heavy atom and its covalently bonded H-atoms for an united atom and represent a bead.).  | Flag, activate when included | Flag |
| `--topog`  | Compute the topographical entropy using  <ul><li>1 : pLogP method</li><li>2: Corr. pLogP method</li><li>3: Corr. density function</li><li>4: Phi Coeff method </li><li>5: Corr. pLogP after adaptive enumeration of states</li></ul> | `0`: no topographical analysis | `int` |
| `--solwm`  | Do water entropy calculation at residue level (A residue as a whole represents a bead.).  | Flag, activate when included | Flag |
| `--solres`  | Do water entropy calculation at residue level (A residue as a whole represents a bead.  | Flag, activate when included | Flag |
| `--soluatom`  | Do solution entropy calculation at united atom level (A heavy atom and its covalently bonded H-atoms for an united atom and represent a bead.).  | Flag, activate when included | Flag |
| `--solContact`  | Do solute contact calculation.  | Flag, activate when included | Flag |
#### Example 
```
CodeEntropyPoseidon -f "Example/data/md_A4_dna.tpr" "Example/data/md_A4_dna_xf.trr" -a "C5'" "C4'" "C3'" -l "all" -t 8 --wm --res --uatom --topog 1 --solwm --solres --soluatom
```

## Script Examples
See `Example` folder
You can add your own trajectories by editing the path in the python script to point to your own trajectories
### `create_new_universe.py`
This repo uses MDAnalysis to parse values and it can only parse force natively for GROMACS TRR and AMBER NETCDF. This scripts shows you how to create a new universe from unsuppported data so that you can use trajectories created from other simulation software or reduce the size of universe to focus on a section of simulation.
### `CodeEntropy_non_topo.py`
Calculate entropy of target trajectory non topographical level
### `CodeEntropy_topo.py`
Calculate entropy of target trajectory based on different method
### `Poseidon_GROMACS`
!!! the trajectories are not included in this repo due to file size limit. 

Run POSEIDON analysis for a GROMACS trajectories
### `Poseidon_LAMMPS`
A LAMMPS example for POSEIDON
### `mcc_mdanalysis`
A DNA example for CodeEntropy
### `mcc_mdanalysis_multiprocess`
mcc_mdanalysis with multiprocess parallelization 

## Copyright

Copyright (c) 2022, DonaldChung-HK


## Acknowledgements
 
Project based on the 

- [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
- [arghya90/CodeEntropy](https://github.com/arghya90/CodeEntropy) version 0.3
- [jkalayan/PoseidonBeta](https://github.com/jkalayan/PoseidonBeta)
