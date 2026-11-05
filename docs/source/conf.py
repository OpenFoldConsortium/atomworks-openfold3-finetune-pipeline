# Sphinx configuration: https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "atomworks-openfold3-finetune-pipeline"
copyright = "2026, The OpenFold Consortium"

extensions = ["myst_parser"]
myst_heading_anchors = 3
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = project
