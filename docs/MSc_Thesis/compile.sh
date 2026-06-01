#!/bin/bash
set -e

pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

find . -name "*.aux" -delete
rm -f main.bbl main.bcf main.blg main.fdb_latexmk main.fls \
      main.lof main.log main.lot main.out main.run.xml main.toc

echo "Done: main.pdf"
