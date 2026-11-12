# Sphinx configuration: https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "atomworks-openfold3-finetune-pipeline"
copyright = "2026, The OpenFold Consortium"

extensions = ["myst_parser", "sphinxcontrib.mermaid"]
# Plain ```mermaid fences render as diagrams in Sphinx, and GitHub renders the same fences.
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = project
