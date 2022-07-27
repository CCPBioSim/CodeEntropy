Developer's Guide
==============================

CodeEntropy uses the Python programming language.

Running tests
-----------------------------
To run the full test suite, simply install ``pytest`` and run in root directory of the repository:

.. code-block:: bash

    pytest

To only run the unit tests in a particular part of program. For example only running test for solute part.

.. code-block:: bash

    pytest CodeEntropy/tests/test_CodeEntropy.py


To only run the a specific test. e.g.

.. code-block:: bash

    pytest CodeEntropy/tests/test_CodeEntropy.py::test_CodeEntropy_parser_labForces