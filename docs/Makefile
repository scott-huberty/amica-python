# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-noplot:
	@$(SPHINXBUILD) -D plot_gallery=0 -b html $(SOURCEDIR) $(BUILDDIR) $(O)

clean:
	@echo "Removing all files from $(BUILDDIR)..."
	rm -rf $(BUILDDIR)
	@echo "Removing Sphinx Gallery Files..."
	rm -rf $(SOURCEDIR)/auto_examples
	rm -rf $(SOURCEDIR)/gen_modules
	rm -f $(SOURCEDIR)/sg_execution_times.rst

# If we build the example gallery, the index file is in build/html/index.html
# Otherwise, it's in build/index.html
view:
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
	    python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/html/index.html')"; \
	elif [ -f "$(BUILDDIR)/index.html" ]; then \
	    python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/index.html')"; \
	else echo "No index.html file found to open. Did you build the docs?"; \
	fi

