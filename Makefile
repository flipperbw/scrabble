_PY=python3
#_PYTHON_CMD=time -p $(_PY) -m
_PYTHON_CMD=$(_PY) -m

BUILD=$(_PY) ./setup.py build_ext -i

PARSE=$(_PYTHON_CMD) scrabble.e
OCR=$(_PYTHON_CMD) scrabble.o


.PHONY: run run-p run-t run-ocr

run: | build-scrabble parse
run-p: | build-p parse-p
run-t: | build-t parse-t
run-ocr: | build-ocr space ocr

.PHONY: space
space:
	@echo -e '\n\n======= Done Building =======\n\n'


.PHONY: build build-p build-t build-scrabble build-ocr build-logger

build:
	$(BUILD) $(args)
build-p:
	$(BUILD) --profile
build-t:
	$(BUILD) --trace
build-scrabble:  ##args?
	$(BUILD) --scrabble
build-ocr:
	$(BUILD) --ocr
build-logger:
	$(BUILD) --logger


.PHONY: parse parse-p parse-t

parse:
	$(PARSE) $(args)
parse-p: ## remove
	$(PARSE) -p $(args)
parse-t:
	$(PARSE) -x $(args)


.PHONY: ocr ocr-p ocr-t

ocr:
	$(OCR) $(args)
ocr-p:
	$(OCR) -p $(args)
ocr-t:
	$(OCR) -x $(args)


.PHONY: clean
clean:
	rm -rf ./build ./scrabble/__pycache__ ./scrabble/*.c ./scrabble/*.so ./scrabble/*.html
