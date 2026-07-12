# Minimal Sphinx makefile.
#   make html    -> docs/_build/html/index.html
#   make strict  -> same, but warnings are errors (use in CI)
#   make clean

# Default build does NOT treat warnings as errors, so a stray cross-reference
# warning cannot block the build the day before a deadline. `make strict` keeps the
# warnings-as-errors behaviour for CI.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html strict clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "\nOpen $(BUILDDIR)/html/index.html"

strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going
	@echo "\nOpen $(BUILDDIR)/html/index.html"

clean:
	rm -rf $(BUILDDIR)

# Auto-rebuild on save (pip install sphinx-autobuild)
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)