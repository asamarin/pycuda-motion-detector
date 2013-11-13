#
# Makefile for Latex
##

LATEX=latex
PDFLATEX=xelatex

main=conference.tex
mainsin=$(main:.tex=)
pdf=$(main:.tex=.pdf)
# bbl=$(main:.tex=.bbl)
# bib=$(wildcard *.bib)
# imgs=$(wildcard img/*.pdf imgs/*.png imgs/*.eps)
# sources=$(wildcard *.tex)

all: $(pdf)

# $(dvi): $(main) $(bbl) $(imgs) $(sources)

$(dvi): $(main)
	$(LATEX) $(mainsin)
	$(LATEX) $(mainsin) 

# $(bbl): $(bib)
#	 $(PDFLATEX) $(mainsin)
#	 bibtex $(mainsin)

# $(pdf): $(main) $(bbl) $(imgs) $(sources)

$(pdf): $(main)
	$(PDFLATEX) $(mainsin)
	$(PDFLATEX) $(mainsin)

clean:
	rm -rf *.log *.out *.aux calibrado.dvi *.bbl *.blg

zip:
	zip -o
