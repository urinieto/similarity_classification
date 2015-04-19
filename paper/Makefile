SRC = learnselfsim

all: $(SRC).tex
	latex $<
	bibtex $(SRC)
	latex $<
	latex $<
	pdflatex $<

check: $(SRC).tex
	chktex $<
	ispell -t $<

gray: $(SRC).pdf
	gs -sOutputFile=$(SRC)-gray.pdf -sDEVICE=pdfwrite \
        -sColorConversionStrategy=Gray -dProcessColorModel=/DeviceGray \
        -dCompatibilityLevel=1.4 -dNOPAUSE -dBATCH $(SRC).pdf

clean:
	rm -f *.aux *.log *.out

spotless: clean
	rm -f *.dvi *.bak *.bbl *.blg
