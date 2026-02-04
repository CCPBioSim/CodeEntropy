Getting Started
===============

This guide walks you through installing and running CodeEntropy, with examples ordered
from the smallest and fastest to larger, more realistic systems.

Each example includes:

* a complete ``config.yaml``
* the exact command used to run it
* an estimated runtime
* a clear explanation of where output files are written

If you are new to CodeEntropy, start with **Example 1**.


Requirements
------------

* Python >= 3.12


Installation
------------

To install the released version:

.. code-block:: bash

   pip install CodeEntropy


Input Files
-----------

For supported formats (any topology and trajectory formats that can be read by
`MDAnalysis <https://userguide.mdanalysis.org/stable/formats/index.html>`_) you will need
to output the **coordinates** and **forces** to the **same file**. Please consult the documentation for your MD simulation code if you need help outputting
the forces.

Units
-----

The program assumes the following default units:

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
     - e
   * - Mass
     - u
   * - Force
     - kJ/(mol·Å)


Quick Start
-----------

A quick and easy way to get started is to use the command-line tool:

.. code-block:: bash

   CodeEntropy --help


Working Directory and Output Location
-------------------------------------

CodeEntropy writes output **relative to the directory you run it from**.

In practice, you should:

#. Put (or download) your simulation input files and a ``config.yaml`` in a working directory.
#. Change into that directory.
#. Run CodeEntropy.

Example:

.. code-block:: bash

   cd /path/to/my/workdir
   CodeEntropy

When you rerun CodeEntropy in the same working directory, CodeEntropy creates sequential
output directories named ``job1/``, ``job2/``, etc. Each ``job*/`` directory contains the
output JSON file and a subdirectory with log files.


Configuration and Arguments
---------------------------

Arguments should go in a ``config.yaml`` file. Values in the YAML file can be overridden
by command-line arguments.

The ``top_traj_file`` argument is required; other arguments have default values.

.. list-table:: Arguments
   :widths: 20 30 10 10
   :class: tight-table
   :header-rows: 1

   * - Argument
     - Description
     - Default
     - Type
   * - ``--top_traj_file``
     - Path to structure/topology file followed by trajectory file. Any MDAnalysis readable
       files should work (for example ``GROMACS TPR and TRR`` or ``AMBER PRMTOP and NETCDF``).
     - Required
     - list of ``str``
   * - ``--force_file``
     - Path to a file with forces. Use this option if the forces are not in the same file
       as the coordinates. The force file must have the same number of atoms and frames as
       the trajectory file. Any MDAnalysis readable files should work (for example
       ``AMBER NETCDF`` or ``LAMMPS DCD``).
     - None
     - ``str``
   * - ``--file_format``
     - Use to tell MDAnalysis the format if the trajectory or force file does not have the
       standard extension recognised by MDAnalysis.
     - None
     - ``str``
   * - ``--kcal_force_units``
     - Set this to True if you have a separate force file with kcal units.
     - ``False``
     - ``bool``
   * - ``--selection_string``
     - Selection string for CodeEntropy such as ``protein`` or ``resid 1:10``. Refer to
       ``MDAnalysis.select_atoms`` for more information.
     - ``"all"``
     - ``str``
   * - ``--start``
     - Start analysing the trajectory from this frame index.
     - ``0``
     - ``int``
   * - ``--end``
     - Stop analysing the trajectory at this frame index (``-1`` means last frame).
     - ``-1``
     - ``int``
   * - ``--step``
     - Interval between two consecutive frame indices to be read.
     - ``1``
     - ``int``
   * - ``--bin_width``
     - Bin width in degrees for making the dihedral angle histogram.
     - ``30``
     - ``int``
   * - ``--temperature``
     - Temperature for entropy calculation (K).
     - ``298.0``
     - ``float``
   * - ``--verbose``
     - Enable verbose output.
     - ``False``
     - ``bool``
   * - ``--outfile``
     - Name of the JSON output file to write results to (filename only). Defaults to ``outfile.json``.
     - ``outfile.json``
     - ``str``
   * - ``--force_partitioning``
     - Factor for partitioning forces when there are weak correlations.
     - ``0.5``
     - ``float``
   * - ``--water_entropy``
     - Use Jas Kalayan's waterEntropy code to calculate the water conformational entropy.
     - ``False``
     - ``bool``
   * - ``--grouping``
     - How to group molecules for averaging.
     - ``molecules``
     - ``str``

Averaging
---------

The code is able to average over molecules of the same type. The ``grouping`` argument
controls how averaging is done.

* ``molecules`` (default): molecules are grouped by atom names and counts.
* ``each``: each molecule is treated as its own group (no averaging).


Examples
--------

The examples below are ordered so the smallest, fastest-running example appears first.


Example 1: DNA Fragment (Smallest / Fastest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Estimated runtime:** ~1-2 minutes (typical laptop/desktop; depends on I/O and CPU)

Data files:

`DNA fragment example (~1MB) <https://ccpbiosim.ac.uk/file-store/codeentropy-examples/dna_example.tar>`_

Create or edit ``config.yaml`` in your working directory:

.. code-block:: yaml

  ---

  run1:
    top_traj_file: ["md_A4_dna.tpr", "md_A4_dna_xf.trr"]
    selection_string: 'all'
    start: 0
    end: -1
    step: 1
  
Run CodeEntropy from that directory:

.. code-block:: bash

   cd /path/to/dna_example
   CodeEntropy

Run (equivalent CLI):

.. code-block:: bash

   cd /path/to/dna_example
   CodeEntropy --top_traj_file md_A4_dna.tpr md_A4_dna_xf.trr --temperature 298.0 --selection_string all --start 0 --end -1 --step 1

Example 2: Lysozyme (Larger / Slower)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Estimated runtime:** ~30–60 minutes (typical workstation; depends strongly on trajectory length and hardware)

Data files:

`Lysozyme example (~1.2GB) <https://ccpbiosim.ac.uk/file-store/codeentropy-examples/lysozyme_example.tar>`_

Create or edit ``config.yaml`` in your working directory:

.. code-block:: yaml

  ---

  run1:
    top_traj_file: ["1AKI_prod.tpr", "1AKI_prod.trr"]
    selection_string: 'all'
    start: 0
    end: 500
    step: 1
    bin_width: 30
    temperature: 300
    verbose: True

Run CodeEntropy from that directory:

.. code-block:: bash

   cd /path/to/lysozyme_example
   CodeEntropy

Run (equivalent CLI):

.. code-block:: bash

   cd /path/to/lysozyme_example
   CodeEntropy --top_traj_file 1AKI_prod.tpr 1AKI_prod.trr --temperature 300.0 --selection_string all --start 0 --end 500 --step 1 --verbose



Overriding YAML Values from the CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Values in ``config.yaml`` can be overridden using command-line flags.

Example (override the trajectory inputs):

.. code-block:: bash

   cd /path/to/dna_example
   CodeEntropy --top_traj_file md_A4_dna.tpr md_A4_dna_xf.trr


Output Structure
----------------

CodeEntropy creates ``job*`` directories for output, where ``*`` is a sequential job
number when you rerun CodeEntropy in the same working directory.

Each ``job*/`` directory contains:

* the output JSON file
* a subdirectory containing log files
