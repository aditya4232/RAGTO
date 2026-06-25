PY ?= python
PIP ?= $(PY) -m pip

.DEFAULT_GOAL := help

help:
	@$(PY) -m devrag.cli --help

install:
	$(PIP) install -e ".[dev,docs]"

install-cpu:
	$(PIP) install -e ".[dev,docs]"

install-gpu:
	$(PIP) install -e ".[dev,docs,gpu]"

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests

test:
	$(PY) -m pytest -q

ingest:
	$(PY) -m devrag.cli ingest --config configs/corpus.yaml

index:
	$(PY) -m devrag.cli index build --config configs/retrieval.yaml

train:
	$(PY) -m devrag.cli train embedding --config configs/training.yaml

eval:
	$(PY) -m devrag.cli eval run --config configs/eval.yaml

eval-embedding:
	$(PY) -m devrag.cli eval embedding-ab --config configs/eval.yaml

charts:
	$(PY) scripts/make_charts.py

serve:
	$(PY) -m devrag.cli serve api --config configs/serving.yaml

ui:
	$(PY) app/gradio_app.py

docs:
	$(PY) -m mkdocs build --strict

docs-serve:
	$(PY) -m mkdocs serve -a 127.0.0.1:8000

publish-hf:
	$(PY) scripts/publish_to_hf.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} +