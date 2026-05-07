Developer Guide
===============

CodeEntropy is open-source, and we welcome contributions from the wider community
to help improve and extend its functionality. This guide walks you through setting
up a development environment, running tests, contributing code, and understanding
the continuous integration workflows.

Getting Started for Developers
------------------------------

Create a development environment using either venv or Conda.

Using venv::

    python -m venv codeentropy-dev
    source codeentropy-dev/bin/activate  # Linux/macOS
    codeentropy-dev\Scripts\activate     # Windows

Using Conda::

    conda create -n codeentropy-dev python=3.14
    conda activate codeentropy-dev

Clone the repository::

    git clone https://github.com/CCPBioSim/CodeEntropy.git
    cd CodeEntropy

Install development dependencies::

    pip install -e ".[testing,docs,pre-commit]"

Running Tests
-------------

CodeEntropy uses **pytest** with separate unit and regression suites.

Run all tests::

    pytest

Run only unit tests::

    pytest tests/unit

Run regression tests::

    pytest tests/regression

Run regression tests excluding slow systems::

    pytest tests/regression -m "not slow"

Run slow regression tests::

    pytest tests/regression --run-slow

Run tests with coverage::

    pytest --cov CodeEntropy --cov-report=term-missing

Run a specific test::

    pytest tests/unit/.../test_file.py::test_function

Updating Regression Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Regression baselines should only be updated when a change intentionally alters
the expected numerical output of CodeEntropy. Do not update baselines simply
because a regression test fails.

Before updating baselines, confirm that:

- The code change is intentional and understood.
- The new output has been reviewed.
- The regression difference is expected.
- The pull request explains why the baselines changed.

For local updates, run::

    pytest tests/regression --update-baselines

For pull requests, baselines can also be updated using the GitHub Actions
workflow **Update Regression Baselines**.

To use it:

1. Push your changes to your PR branch.
2. Open the **Actions** tab on GitHub.
3. Select **Update Regression Baselines**.
4. Click **Run workflow**.
5. Select your PR branch.
6. Run the workflow.

The workflow runs the regression tests with ``--update-baselines`` and commits
any changed baseline files back to the selected branch.

Only use this workflow when the baseline changes are intentional. If a regression
test fails unexpectedly, investigate the failure instead of updating the
baseline.

Regression Test Data
--------------------

Regression datasets are automatically downloaded from the CCPBioSim filestore
and cached locally in ``.testdata/`` when tests are run.

No manual setup is required.

The test configuration files reference datasets using the ``${TESTDATA}``
placeholder, which is expanded automatically during test execution.

Coding Standards
----------------

We use **pre-commit hooks** to maintain code quality and consistent style.

Enable hooks::

    pre-commit install

Our tooling stack:

- **Linting and formatting** via ``ruff``
- **Basic repository checks** via ``pre-commit-hooks``

Ruff performs:

- Code formatting
- Import sorting
- Static analysis
- Style enforcement

Run checks manually::

    pre-commit run --all-files

Skip checks for a commit (not recommended)::

    git commit -n

.. note::

    Pull requests must pass all pre-commit checks before being merged.

Continuous Integration (CI)
---------------------------

CodeEntropy uses **GitHub Actions** with multiple workflows to ensure stability
across platforms and Python versions.

Pull Request checks include:

- Unit tests on Linux, macOS, and Windows
- Python versions 3.12-3.14
- Quick regression tests
- Documentation build
- Pre-commit validation

Daily workflow:

- Runs automated test validation

Weekly workflows:

- Full regression suite including slow tests
- Documentation build across all Python versions

CI also caches regression datasets to improve performance.

Building Documentation
----------------------

Build documentation locally::

    cd docs
    make html

The generated HTML files will be in ``docs/build/html/``.

Open ``index.html`` in your browser to preview.

Documentation sources are located in:

- ``docs/user_guide/``
- ``docs/developer_guide/``

Contributing Code
-----------------

If you would like to contribute to **CodeEntropy**, please refer to the
`Contributing Guidelines <https://github.com/CCPBioSim/CodeEntropy?tab=contributing-ov-file>`_.

Creating an Issue
^^^^^^^^^^^^^^^^^

If you encounter bugs or want to request features:

1. Open an issue on GitHub.
2. Provide a clear description and input files if applicable.

Branching
^^^^^^^^^

- Never commit directly to ``main``.
- Create a branch named after the issue::

    git checkout -b 123-feature-description

Pull Requests
^^^^^^^^^^^^^

1. Make your changes in a branch.
2. Ensure tests and pre-commit checks pass.
3. Submit a pull request.
4. At least one core developer will review it.
5. Include updated documentation and tests for new code.

Summary
-------

Full developer setup::

    git clone https://github.com/CCPBioSim/CodeEntropy.git
    cd CodeEntropy
    pip install -e ".[testing,docs,pre-commit]"
    pre-commit install
    pytest
