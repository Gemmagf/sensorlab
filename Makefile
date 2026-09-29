.PHONY: help install test lint format clean train evaluate score serve site docker notebook notebooks download-tep

VENV     := .venv
PYTHON   := python3.11
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip
PY       := $(BIN)/python
TORCH_CPU := --extra-index-url https://download.pytorch.org/whl/cpu

help:
	@echo "sensorlab — common commands"
	@echo ""
	@echo "  make install        Create .venv and install package + serve/dev extras (CPU torch)"
	@echo "  make test           Run pytest"
	@echo "  make lint           ruff check + format check"
	@echo "  make format         Apply ruff format & autofixes"
	@echo "  make train          Fit one pipeline on synthetic data, save models/pipeline.joblib"
	@echo "  make evaluate       Multi-seed experiment -> artifacts/results.json (README numbers)"
	@echo "  make score IN=f.csv Score a CSV/parquet of sensor rows with the saved pipeline"
	@echo "  make download-tep   Fetch the real Tennessee Eastman dataset"
	@echo "  make notebook       Launch Jupyter notebook server"
	@echo "  make notebooks      Regenerate and execute the narrative notebooks"
	@echo "  make site           Export the governance dashboard as static files to site/"
	@echo "  make serve          Same dashboard with the live scoring API on http://127.0.0.1:8000"
	@echo "  make docker         Build the container image (model trained at build time)"
	@echo "  make clean          Remove build/test caches and virtualenv"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel

install: $(BIN)/python
	$(PIP) install $(TORCH_CPU) -e ".[serve,survival,dev]"

test:
	$(BIN)/pytest -v

lint:
	$(BIN)/ruff check src tests scripts app
	$(BIN)/ruff format --check src tests scripts app

format:
	$(BIN)/ruff check --fix src tests scripts app
	$(BIN)/ruff format src tests scripts app

train:
	$(BIN)/sensorlab train --data synthetic

evaluate:
	$(BIN)/sensorlab evaluate --seeds 0 1 2

score:
	$(BIN)/sensorlab score --input $(IN) --output artifacts/scored.csv --drift

download-tep:
	$(PY) scripts/download_tep.py

notebook:
	$(BIN)/jupyter notebook notebooks/

notebooks:
	$(PY) scripts/build_notebooks.py
	for nb in notebooks/0*.ipynb; do $(BIN)/jupyter nbconvert --to notebook --execute --inplace $$nb; done

site:
	$(BIN)/sensorlab export-site --train-if-missing

serve:
	$(BIN)/sensorlab serve --train-if-missing

docker:
	docker build -t sensorlab .

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
