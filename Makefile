# Common developer tasks. DATA points at the folder holding train/ and public_test/
# (create the link once: ln -s /path/to/train+public_test data).
DATA ?= data
PY ?= python3

.PHONY: install dev test lint stats oracle score-oracle

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,yaml]"

test:
	LIFEMATE_DATA=$(DATA) $(PY) -m unittest discover -s tests -t . -v

lint:
	ruff check src tests && ruff format --check src tests

stats:
	PYTHONPATH=src $(PY) -m lifemate stats --data $(DATA) --split train

# End-to-end sanity check: replay the gold actions through the full agent loop (must score 100).
oracle:
	PYTHONPATH=src $(PY) -m lifemate infer --data $(DATA) --split train --backend oracle --asr oracle --out outputs/oracle_train.jsonl
	PYTHONPATH=src $(PY) -m lifemate score --data $(DATA) --split train --submission outputs/oracle_train.jsonl
