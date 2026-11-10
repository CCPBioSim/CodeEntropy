:html_theme.sidebar_secondary.remove:

CodeEntropy
===========

.. ccpbiosim-hero::
   :logo-light: _static/logos/biosim-codeentropy_logo_dark.png
   :logo-dark: _static/logos/biosim-codeentropy_logo_light.png
   :logo-alt: CodeEntropy logo
   :tagline: CodeEntropy
   :button: getting_started | Get started
   :github: CCPBioSim/CodeEntropy

   Entropy of macromolecular systems from MD simulations

This code is a complete and generally applicable set of tools for computing
entropy of macromolecular systems from the forces sampled in a MD simulation.

Quick start
-----------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install CodeEntropy

   .. tab-item:: Conda

      .. code-block:: bash

         conda install -c conda-forge -c ccpbiosim codeentropy

   .. tab-item:: Run

      .. code-block:: bash

         CodeEntropy --help

Explore the documentation
-------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started
      :link-type: doc

      Install CodeEntropy and run the examples.

   .. grid-item-card:: 📖 API
      :link: reference/CodeEntropy/index
      :link-type: doc

      Reference for the CodeEntropy package.

   .. grid-item-card:: 🔬 Science
      :link: science
      :link-type: doc

      The Multiscale Cell Correlation theory behind the method.

   .. grid-item-card:: ❓ FAQ
      :link: faq
      :link-type: doc

      Answers to common warnings and questions.

   .. grid-item-card:: 👩‍💻 Developer Guide
      :link: developer_guide
      :link-type: doc

      Setting up a development environment and contributing.

.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started <getting_started>
   API <reference/CodeEntropy/index>
   Science <science>
   FAQ <faq>
   Developer Guide <developer_guide>
