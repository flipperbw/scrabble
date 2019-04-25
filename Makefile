BUILD_CMD=python3 ./setup.py build_ext -i
PARSE_CMD=time -p ./scrabble/e.py
OCR_CMD=time -p ./scrabble/o.py


.PHONY: run run-p run-t

run: build parse
run-p: build-p parse-p
run-t: build-t parse-t


.PHONY: build build-p build-t

build:
	$(BUILD_CMD)
build-p:
	$(BUILD_CMD) --profile
build-t:
	$(BUILD_CMD) --trace


.PHONY: parse parse-p parse-t

parse:
	$(PARSE_CMD) $(args)
parse-p:
	$(PARSE_CMD) -p $(args)
parse-t:
	$(PARSE_CMD) -x $(args)


.PHONY: ocr ocr-p ocr-t

ocr:
	$(OCR_CMD) $(args)
ocr-p:
	$(OCR_CMD) -p $(args)
ocr-t:
	$(OCR_CMD) -x $(args)
