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

CodeEntropy can be installed using either pip or Conda.

Install with pip
^^^^^^^^^^^^^^^^

To install the released version from PyPI:

.. code-block:: bash

   pip install CodeEntropy


Install with Conda
^^^^^^^^^^^^^^^^^^

CodeEntropy is also available via the CCPBioSim Anaconda channel.

Create a dedicated environment:

.. code-block:: bash

   conda create -n codeentropy python=3.14
   conda activate codeentropy

Install CodeEntropy:

.. code-block:: bash

   conda install -c conda-forge -c CCPBioSim codeentropy


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
   * - ``--output_file``
     - Name of the JSON output file to write results to (filename only). Defaults to
       ``output_file.json``.
     - ``output_file.json``
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
   * - ``--combined_forcetorque``
     - Use the combined force-torque covariance matrix for the highest level to match the
       2019 paper.
     - ``True``
     - ``bool``
   * - ``--customised_axes``
     - Use custom bonded axes to get COM, MOI and PA that match the 2019 paper.
     - ``True``
     - ``bool``
   * - ``--search_type``
     - Method for finding neighbouring molecules.
     - ``RAD``
     - ``str``
   * - ``--parallel_frames``
     - Execute frame-local covariance calculations in parallel. When enabled, frame-level
       work is submitted to Dask and reduced in the parent process.
     - ``False``
     - ``bool``
   * - ``--use_dask``
     - Enable local Dask frame parallelism. This is useful for running frame-level work
       across local worker processes.
     - ``False``
     - ``bool``
   * - ``--dask_workers``
     - Number of local Dask worker processes to use for parallel frame execution. If unset,
       Dask chooses a default.
     - ``None``
     - ``int``
   * - ``--dask_threads_per_worker``
     - Number of threads per local Dask worker. ``1`` is recommended for trajectory safety
       with MDAnalysis.
     - ``1``
     - ``int``
   * - ``--hpc``
     - Use a SLURM-backed Dask cluster for parallel frame execution.
     - ``False``
     - ``bool``
   * - ``--submit``
     - Submit a master SLURM job and exit instead of running immediately in the current
       process. This is intended for HPC batch submission.
     - ``False``
     - ``bool``
   * - ``--hpc_queue``
     - SLURM partition or queue to use for Dask worker jobs.
     - ``None``
     - ``str``
   * - ``--hpc_nodes``
     - Number of SLURM Dask worker jobs to launch.
     - ``1``
     - ``int``
   * - ``--hpc_cores``
     - Number of CPU cores requested per Dask worker job.
     - ``1``
     - ``int``
   * - ``--hpc_processes``
     - Number of Dask worker processes per SLURM job.
     - ``1``
     - ``int``
   * - ``--hpc_memory``
     - Memory requested per Dask worker job, for example ``4GB`` or ``16GB``.
     - ``4GB``
     - ``str``
   * - ``--hpc_walltime``
     - Walltime requested for each Dask worker job, formatted as ``HH:MM:SS``.
     - ``01:00:00``
     - ``str``
   * - ``--hpc_account``
     - Optional SLURM account or project code.
     - ``None``
     - ``str``
   * - ``--hpc_qos``
     - Optional SLURM QoS value.
     - ``None``
     - ``str``
   * - ``--hpc_constraint``
     - Optional SLURM node constraint.
     - ``None``
     - ``str``
   * - ``--conda_path``
     - Path to the conda executable used in the SLURM worker prologue.
     - ``conda``
     - ``str``
   * - ``--conda_exec``
     - Conda-compatible executable to use for environment activation, usually ``conda`` or
       ``mamba``.
     - ``conda``
     - ``str``
   * - ``--conda_env``
     - Conda environment name to activate on SLURM workers.
     - ``None``
     - ``str``

Parallel Frame Execution
------------------------

CodeEntropy can optionally process trajectory frames in parallel using Dask. This is
most useful for larger trajectories where the frame-local covariance calculations are
one of the slowest parts of the workflow.

The parallel implementation works as a map/reduce workflow:

* each Dask worker processes one frame at a time;
* each worker returns a frame-local covariance result;
* the parent process reduces those frame-local results into the final running
  covariance averages;
* the entropy graph runs after frame reduction has completed.

This means workers do not directly modify the shared covariance accumulators. The
parent process remains responsible for reduction, which keeps the parallel execution
consistent with the sequential workflow.

Local Dask Execution
^^^^^^^^^^^^^^^^^^^^

For local workstation or laptop use, enable ``parallel_frames`` and ``use_dask`` in
``config.yaml``:

.. code-block:: yaml

  ---

  run1:
    top_traj_file: ["md_A4_dna.tpr", "md_A4_dna_xf.trr"]
    selection_string: "all"
    start: 0
    end: 100
    step: 1

    parallel_frames: true
    use_dask: true
    dask_workers: 4
    dask_threads_per_worker: 1

The recommended value for ``dask_threads_per_worker`` is ``1``. This keeps each worker
process independent and avoids thread-safety issues when reading trajectory data.

The same run can also be started from the command line:

.. code-block:: bash

   CodeEntropy \
     --parallel_frames true \
     --use_dask true \
     --dask_workers 4 \
     --dask_threads_per_worker 1

For very small systems or short trajectories, local Dask may not be faster than the
sequential path because there is overhead in starting workers and transferring frame
data. It is best suited to larger calculations with many frames.

SLURM / HPC Dask Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

On a SLURM-based HPC system, CodeEntropy can create a Dask cluster using SLURM worker
jobs. This is enabled with ``hpc: true``.

Example ``config.yaml``:

.. code-block:: yaml

  ---

  run1:
    top_traj_file: ["1AKI_prod_new.tpr", "1AKI_prod_new.trr"]
    selection_string: "all"
    start: 0
    end: 500
    step: 1

    parallel_frames: true
    hpc: true

    hpc_queue: standard
    hpc_nodes: 4
    hpc_cores: 8
    hpc_processes: 1
    hpc_memory: 16GB
    hpc_walltime: "02:00:00"

    hpc_account: null
    hpc_qos: null
    hpc_constraint: null

    conda_path: conda
    conda_exec: conda
    conda_env: codeentropy

The important HPC options are:

* ``hpc_queue``: SLURM partition or queue.
* ``hpc_nodes``: number of Dask worker jobs to launch.
* ``hpc_cores``: number of CPU cores requested per Dask worker job.
* ``hpc_processes``: number of Dask worker processes per SLURM job.
* ``hpc_memory``: memory requested per Dask worker job.
* ``hpc_walltime``: walltime requested for each worker job.
* ``conda_env``: environment to activate on the worker jobs.

Submitting a Master SLURM Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want CodeEntropy to submit a master SLURM job and then exit, set
``submit: true`` as well as ``hpc: true``:

.. code-block:: yaml

  ---

  run1:
    top_traj_file: ["1AKI_prod.tpr", "1AKI_prod.trr"]
    selection_string: "all"
    start: 0
    end: 500
    step: 1

    submit: true
    parallel_frames: true
    hpc: true

    hpc_queue: standard
    hpc_nodes: 4
    hpc_cores: 8
    hpc_processes: 1
    hpc_memory: 16GB
    hpc_walltime: "02:00:00"

    hpc_account: null
    hpc_qos: null
    hpc_constraint: null

    conda_path: conda
    conda_exec: conda
    conda_env: codeentropy

Run CodeEntropy from the working directory containing ``config.yaml``:

.. code-block:: bash

   CodeEntropy

In submit mode, CodeEntropy writes and submits a master SLURM script, then exits from
the current process. The submitted master job starts CodeEntropy again on the cluster,
where the SLURM-backed Dask workers are then launched.

Choosing a Parallel Mode
^^^^^^^^^^^^^^^^^^^^^^^^

Use sequential execution for small tests and debugging:

.. code-block:: yaml

  parallel_frames: false
  use_dask: false
  hpc: false
  submit: false

Use local Dask when running on a workstation:

.. code-block:: yaml

  parallel_frames: true
  use_dask: true
  dask_workers: 4
  dask_threads_per_worker: 1

Use HPC Dask when running inside an allocated HPC session or batch job:

.. code-block:: yaml

  parallel_frames: true
  hpc: true

Use submit mode when you want CodeEntropy to create and submit the master SLURM job
for you:

.. code-block:: yaml

  submit: true
  parallel_frames: true
  hpc: true

Averaging
---------

The code is able to average over molecules of the same type. The ``grouping`` argument
controls how averaging is done.

* ``molecules`` (default): molecules are grouped by atom names and counts.
* ``each``: each molecule is treated as its own group (no averaging).

Counting Neighbours
-------------------

The code counts the number of neighbours for the orientational entropy.
There are currently two options, chosen by the ``search_type`` argument.

* ``RAD`` (default): Uses the relative angular distance method.
* ``grid``: Uses the MDAnalysis NeighborSearch method.


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
