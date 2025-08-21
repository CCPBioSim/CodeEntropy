Community guidelines
==============================

Contribute to the software
-----------------------------
To contribute to CodeEntropy, see the Developer's Information section below.

Report Issues
-----------------
Create an issue ticket on GitHub and the team will review it as soon as possible. Please send us the input file as well.

Seek support
----------------
Post on GitHub discussion and the team will reply as soon as possible.

Developer's Information
==============================

CodeEntropy uses the Python programming language.

Running tests
--------------
To run the full test suite, simply install ``pytest`` and run in root directory of the repository:

.. code-block:: bash

    pytest

To only run the unit tests in a particular part of program, for example only running tests for the levels functions:

.. code-block:: bash

    pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py


To only run the a specific test, e.g.:

.. code-block:: bash

    pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py::test_select_levels

Including your code into the CodeEntropy repository
---------------------------------------------------
Any bugs, problems, or feature requests should get an issue on the GitHub repostitory that clearly explains the situation.
Code cannot be committed directly to the main branch.
New branches should be named after the issue that is being worked on.

In order to add code to the main branch of CodeEntropy, a pull request must be created.
All pull requests will be reviewed by at least one of the core development team.
Up to date documentation and tests for all new code will be required before a pull request is approved.
Please use the pull request template, clearly explaining the purpose and effect of the pull request will aid in reviewing them quickly and accurately.
