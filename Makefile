# ---------------------------------------------------------------------------
# Build the technical report (REPORT.tex -> REPORT.pdf).
#
# Requires a LaTeX toolchain (TeX Live / MacTeX) providing `latexmk` + `pdflatex`.
#   macOS:   brew install --cask mactex-no-gui   (or: brew install basictex)
#   Debian:  sudo apt-get install texlive-full latexmk
#
# Overleaf users: no Makefile needed -- upload REPORT.tex, set it as the main
# document, and Overleaf compiles it with pdflatex automatically.
# ---------------------------------------------------------------------------

MAIN    = REPORT
LATEXMK = latexmk
FLAGS   = -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all watch clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEXMK) $(FLAGS) $(MAIN).tex

# Continuous preview: rebuilds on every save.
watch:
	$(LATEXMK) $(FLAGS) -pvc $(MAIN).tex

# Remove auxiliary build files, keep the PDF.
clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml

# Remove everything generated, including the PDF.
distclean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml
