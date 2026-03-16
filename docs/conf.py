#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/stable/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
import time

# Ensure the *repo root* is on sys.path so autodoc imports the local package.
# This works whether conf.py is in docs/ or docs/source/.
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

# -- Project information -----------------------------------------------------
project = "CodeEntropy"
copyright_first_year = "2022"
copyright_owners = "CCPBioSim"
author = "CCPBioSim"

current_year = str(time.localtime().tm_year)
copyright_year_string = (
    current_year
    if current_year == copyright_first_year
    else f"{copyright_first_year}-{current_year}"
)
copyright = f"{copyright_year_string}, {copyright_owners}. All rights reserved"

version = ""
release = ""

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "sphinx_copybutton",
]

# Autosummary (API stubs)
autosummary_generate = True

# Napoleon: Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Render Google "Args:" into :param: fields (recommended)
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# (Optional) If some modules have optional heavy deps, you can mock them here:
autodoc_mock_imports = ["MDAnalysis", "rdkit"]

templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"
language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "default"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "dark_logo": "logos/biosim-codeentropy_logo_light.png",
    "light_logo": "logos/biosim-codeentropy_logo_dark.png",
}

html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = "CodeEntropydoc"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}

latex_documents = [
    (
        master_doc,
        "CodeEntropy.tex",
        "CodeEntropy Documentation",
        "CodeEntropy",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [(master_doc, "CodeEntropy", "CodeEntropy Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "CodeEntropy",
        "CodeEntropy Documentation",
        author,
        "CodeEntropy",
        "CodeEntropy tool with POSEIDON code integrated to form a complete and "
        "generally applicable set of tools for calculating entropy",
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")
