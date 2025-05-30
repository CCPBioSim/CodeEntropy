Community guidelines
==============================

Contribute to the software
-----------------------------
To contribute to CodeEntropy, fork the repository and create a pull request when you want to share and push your changes upstream.

Report Issue
-----------------
Create an issue ticket on GitHub and the team will review it as soon as possible. Please send us the input file as well.

Seek support
----------------
Post on GitHub discussion and the team will reply as soon as possible.

Developer's Information
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
