FIGURES = $(wildcard figures/*.png)

all: solution.pdf

%.pdf: %.md
	pandoc --latex-engine=xelatex -V geometry:margin=1in $< -o $@

solution.pdf: solution.md $(FIGURES)