Getting Started
===============

Requirements
----------------

* Python > 3.9

Installation
----------------
Run the following at the root directory of this repository

.. code-block:: bash
    
    pip install CodeEntropy

Input
----------
For supported format (any topology and trajectory formats that can be read by `MDAnalysis <https://userguide.mdanalysis.org/stable/formats/index.html>`_) you will need to output the **coordinates** and **forces** to the **same file**.


Units
------------
The program assumes the following default unit

.. list-table:: Units
   :widths: 20 20
   :class: tight-table
   :header-rows: 1
   
   * - Quantity
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

A quick and easy way to get started is to use the command-line tool which you can run in bash by simply typing ``CodeEntropy``

For help
^^^^^^^^^^^
.. code-block:: bash
    
    CodeEntropy -h

Arguments
^^^^^^^^^^^^^
Arguments should go in a config.yaml file.
The values in the yaml file can be overridden by command line arguments.
The top_traj_file argument is necessary to identify your simulation data, the others can use default values.

.. list-table:: Arguments
   :widths: 20 30 10 10
   :class: tight-table
   :header-rows: 1
    
   * - Arguments
     - Description
     - Default
     - Type
   * - ``--top_traj_file`` 
     - Path to Structure/topology file(``AMBER PRMTOP``, ``GROMACS TPR`` or topology file with MDAnalysis readable dihedral information (not officially supported)) followed by Trajectory file(s) (``GROMAC TRR`` or ``AMBER NETCDF``) You will need to output the **coordinates** and **forces** to the **same file** . 
     - Required
     - list of ``str`` 
   * - ``--selection_string``
     - Selection string for CodeEntropy such as protein or resid, refer to ``MDAnalysis.select_atoms`` for more information.
     - ``"all"``: select all atom in trajectory
     - ``str``
   * - ``--start``
     - Start analysing the trajectory from this frame index.
     - ``0``: From begining
     - ``int``
   * - ``--end``
     - Stop analysing the trajectory at this frame index
     - ``-1``: end of trajectory
     - ``int``
   * - ``--step``
     - Interval between two consecutive frame indices to be read
     - ``1``
     - ``int``
   * - ``--bin_width``
     - Bin width in degrees for making the dihedral angle histogram
     - ``30``
     - ``int``
   * - ``--temperature``
     - Temperature for entropy calculation (K)
     - ``298.0``
     - ``float``
   * - ``--verbose``
     - Enable verbose output
     - ``False``
     - ``bool``
   * - ``--thread``
     - How many multiprocess to use.
     - ``1``: for single core execution
     - ``int``
   * - ``--outfile``
     - Name of the file where the text format output will be written.
     - ``outfile.out``
     - ``str``
   * - ``--force_partitioning``
     - Factor for partitioning forces when there are weak correlations
     - ``0.5``
     - ``float``
   * - ``--water_entropy``
     - Use Jas Kalayan's waterEntropy code to calculate the water conformational entropy
     - ``False``
     - ``bool``


Example
^^^^^^^^^^

.. code-block:: bash
    
    # example 1 DNA

    # example 2 lysozyme in water
