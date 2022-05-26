CodeEntropy
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/CodeEntropy/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/CodeEntropy/actions?query=workflow%3ACI)
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
#### For help
```
CodeEntropyPoseidon -h
```
#### Example 
```
CodeEntropyPoseidon -f "Example/data/md_A4_dna.tpr" "Example/data/md_A4_dna_xf.trr" -a "C5'" "C4'" "C3'" -l "all" -t 8 --wm --res --uatom --topog 1 --solwm --solres --soluatom
```

## Examples
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
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
Modified from [arghya90/CodeEntropy](https://github.com/arghya90/CodeEntropy) version 0.3
