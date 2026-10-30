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

# Put the repo root on sys.path so the local package can be imported.
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
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
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "autoapi.extension",
]

# Napoleon: Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Render Google "Args:" into :param: fields (recommended)
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# API reference generated from the source by sphinx-autoapi
autoapi_dirs = ["../../CodeEntropy"]
autoapi_root = "reference"
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
# Include the __init__ docstring, where the constructor arguments are documented
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Show type hints in the parameter list rather than the signature
autodoc_typehints = "description"
add_module_names = False
python_maximum_signature_line_length = 88

templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"
language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "default"

# -- Options for HTML output -------------------------------------------------
html_theme = "ccpbiosim_sphinx_theme"
html_context = {"github_repo": "CodeEntropy"}

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
