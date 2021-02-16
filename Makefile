PANDOC := pandoc -t beamer -s --highlight-style pygments -V colorlinks
TARGETS = \
	docs/pdf/class_1/presentation.pdf \
	docs/pdf/class_2/presentation.pdf \
	docs/pdf/class_3/presentation.pdf \
	docs/pdf/class_4/presentation.pdf \
	docs/pdf/class_5/presentation.pdf \

all: $(TARGETS)

docs/pdf/class_1/presentation.pdf: src/class_1/presentation.md
docs/pdf/class_2/presentation.pdf: src/class_2/presentation.md
docs/pdf/class_3/presentation.pdf: src/class_3/presentation.md
docs/pdf/class_4/presentation.pdf: src/class_4/presentation.md
docs/pdf/class_5/presentation.pdf: src/class_5/presentation.md

$(TARGETS):
	mkdir -p `echo $@ | sed 's|\(.*\)/.*|\1|'` && $(PANDOC) $^ -o $@