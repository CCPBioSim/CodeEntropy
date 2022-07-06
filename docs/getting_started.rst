Getting Started
===============

Requirements
----------------

* Python > 3.9
* gcc
* g++

Installation
----------------
Run the following at the root directory of this repository

.. code-block:: bash
    
    pip install CodeEntropy

Units
------------
The program assumes the following default unit

.. list-table:: Units
   :widths: 20 20
   :class: tight-table
   :header-rows: 1
   
   * - Qunatity
     - Unit
   * - Length
     - Å
   * - Time
     - ps
   * - Charge
     - `e`
   * - Mass
     - u
   * - Force
     - kJ/(mol·Å)

Quick start guide
--------------------
.. Warning::

     This doesn't work on Windows!!!

A quick and easy way to get started is to use the command-line tool which you can run in bash by simply typing ``CodeEntropyPoseidon``

For help
^^^^^^^^^^^
.. code-block:: bash
    
    CodeEntropyPoseidon -h

Arguments
^^^^^^^^^^^^^

.. Warning::

    Method 3 and 4 for topographical analysis is still in development!!!

.. list-table:: Arguments
   :widths: 20 30 10 10
   :class: tight-table
   :header-rows: 1
    
   * - Arguments
     - Description
     - Default
     - Type
   * - ``-f``, ``--top_traj_file`` 
     - Path to Structure/topology file(``AMBER PRMTOP``, ``GROMACS TPR`` or topology file with MDAnalysis readable dihedral information (not officially supported)) followed by Trajectory file(s) (``GROMAC TRR`` or ``AMBER NETCDF``)
     - Required
     - list of ``str`` 
   * - ``-l``, ``--selectString``
     - Selection string for CodeEntropy such as protein or resid, refer to ``MDAnalysis.select_atoms`` for more information.
     - ``"all"``: select all atom in trajectory
     - ``str``
   * - ``-b``, ``--begin``
     - Start analysing the trajectory from this frame index.
     - ``0``: From begining
     - ``int``
   * - ``-e``, ``--end``
     - Stop analysing the trajectory at this frame index
     - ``-1``: end of trajectory
     - ``int``
   * - ``-d``, ``--step``
     - Steps between frame
     - ``1``
     - ``int``
   * - ``-k``, ``--tempra``
     - Temperature for entropy calculation (K)
     - ``298.0``
     - ``float``
   * -  ``-t``, ``--thread``
     - How many multiprocess to use.
     - ``1``: for single core execution
     - ``int``
   * - ``-o``, ``--out``
     - Name of the file where the text format output will be written.
     - ``outfile.out``
     - ``str``
   * - ``-v``, ``--csvout``
     - Name of the file where the total entropy output will be written.
     - ``outfile.csv``
     - ``str`` 
   * - ``-r``, ``--resout``
     - Name of the file where the residue entropy output will be written.
     - ``res_outfile.csv``
     - ``str``
   * - ``-m``, ``--mout``
     - Name of the file where certain matrices will be written.
     - ``None``
     - ``str``
   * - ``-n``, ``--nmd`` 
     - Name of the file where VMD compatible NMD format files with mode information will be printed.
     - ``None``
     - ``str``
   * - ``-a``, ``--rotationalaxis``
     - The 3 atom name in each residue for rotational and translational axis.
     - ``['C', 'CA', 'N']`` : for protein 
     - list of ``str``
   * - ``-c``, ``--cutShell``
     -  Include cutoff shell analysis, add cutoff distance in angstrom. 
     - ``None`` \: will ust the RAD Algorithm. See Higham, Jonathan, and Richard H Henchman. “Locally adaptive method to define coordination shell.” The Journal of chemical physics vol. 145,8 (2016): 084108. doi:10.1063/1.4961439
     - list of ``str``
   * - ``-p``, ``--pureAtomNum``
     - Reference molecule resid for system of pure liquid.
     - ``1``
     - ``int``
   * - ``-x``, ``--excludedResnames`` 
     - Exclude a list of molecule names from nearest non-like analysis.
     - ``None``
     - list of ``str``
   * - ``-w``, ``--water``
     - Resname for water molecules. 
     - ``'Wat'``
     - list of ``str``
   * - ``-s``, ``--solvent``
     - Include resname of solvent molecules (case-sensitive).
     - ``None``
     - list of ``str``
   * - ``--wm``
     - Do entropy calculation at whole molecule level (The whole molecule is treated as one single bead.).
     - Flag, activate when included
     - Flag
   * - ``--res``
     - Do entropy calculation at residue level (A residue as a whole represents a bead.).
     - Flag, activate when included
     - Flag
   * - ``--uatom``
     - Do entropy calculation at united atom level (A heavy atom and its covalently bonded H-atoms for an united atom and represent a bead.).
     - Flag, activate when included
     - Flag
   * - ``--topog``
     - Compute the topographical entropy using :  
        * 1 : pLogP method (will separate between backbone and side chain)
        * 2 : Corr. pLogP method (will separate between backbone and side chain)
        * 5 : Corr. pLogP after adaptive enumeration of states
     -  ``0`` : no topographical analysis 
     -  ``int``
   * - ``--solwm``
     - Do solution entropy calculation at residue level (The whole molecule is treated as one single bead.).
     - Flag, activate when included
     - Flag
   * - ``--solres``
     - Do solution entropy calculation at residue level (A residue as a whole represents a bead.
     - Flag, activate when included
     - Flag
   * - ``--soluatom``
     - Do solution entropy calculation at united atom level (A heavy atom and its covalently bonded H-atoms for an united atom and represent a bead.).
     - Flag, activate when included
     - Flag
   * - ``--solContact``
     - Do solute contact calculation.
     - Flag, activate when included
     - Flag


Example
^^^^^^^^^^

.. code-block:: bash
    
    # example 1 DNA
    CodeEntropyPoseidon -f "Example/data/md_A4_dna.tpr" "Example/data/md_A4_dna_xf.trr" -a "C5'" "C4'" "C3'" -l "all" -t 8 --wm --res --uatom --topog 5

    # example 2 lysozyme in water
    CodeEntropyPoseidon -f "Example/data/1AKI_prod_60.tpr" "Example/data/1AKI_prod_60.trr" -l "protein" -b 1 -e 30 -d 2 --wm --res --uatom --topog 1 --solwm --solres --soluatom --solContact
