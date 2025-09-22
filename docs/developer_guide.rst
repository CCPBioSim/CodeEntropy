Developer Guide
===============

CodeEntropy is open-source, and we welcome contributions from the wider community to help improve and extend its functionality. This guide walks you through setting up a development environment, running tests, submitting contributions, and maintaining coding standards.

Getting Started for Developers
------------------------------

Create a virtual environment::

    python -m venv codeentropy-dev
    source codeentropy-dev/bin/activate  # Linux/macOS
    codeentropy-dev\Scripts\activate     # Windows

Clone the repository::

    git clone https://github.com/CCPBioSim/CodeEntropy.git
    cd CodeEntropy

Install development dependencies::

    pip install -e ".[testing,docs,pre-commit]"

Running Tests
-------------

Run the full test suite::

    pytest -v

Run tests with code coverage::

    pytest --cov CodeEntropy --cov-report=term-missing

Run tests for a specific module::

    pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py

Run a specific test::

    pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py::test_select_levels

Coding Standards
----------------

We use **pre-commit hooks** to maintain code quality and consistent style. To enable these hooks::

    pre-commit install

This ensures:

- **Formatting** via ``black`` (`psf/black`)
- **Import sorting** via ``isort`` with the ``black`` profile
- **Linting** via ``flake8`` with ``flake8-pyproject``
- **Basic checks** via ``pre-commit-hooks``, including:
  
  - Detection of large added files
  - AST validity checks
  - Case conflict detection
  - Executable shebang verification
  - Merge conflict detection
  - TOML and YAML syntax validation

To skip pre-commit checks for a commit::

    git commit -n

.. note::

    Pull requests must pass all pre-commit checks before being merged.

Continuous Integration (CI)
---------------------------

CodeEntropy uses **GitHub Actions** to automatically:

- Run all tests
- Check coding style
- Build documentation
- Validate versioning

Every pull request will trigger these checks to ensure quality and stability.

Building Documentation
----------------------

Build locally::

    cd docs
    make html

The generated HTML files will be in ``docs/build/html/``. Open ``index.html`` in your browser to view the documentation.

Edit docs in the following directories:

- ``docs/user_guide/``
- ``docs/developer_guide/``

Contributing Code
-----------------

Creating an Issue
^^^^^^^^^^^^^^^^^

If you encounter bugs or want to request features:

1. Open an issue on GitHub.
2. Provide a clear description and input files if applicable.

Branching
^^^^^^^^^

- Never commit directly to ``main``.
- Create a branch named after the issue::

    git checkout -b 123-fix-levels

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
    pip install -e .[testing,docs,pre-commit]
    pre-commit install
    pytest --cov CodeEntropy --cov-report=term-missing
