.PHONY: help install test lint format clean train app notebook download-tep

VENV     := .venv
PYTHON   := python3.11
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip
PY       := $(BIN)/python

help:
	@echo "sensorlab — common commands"
	@echo ""
	@echo "  make install        Create .venv and install package + dev extras"
	@echo "  make test           Run pytest"
	@echo "  make lint           ruff check + format check"
	@echo "  make format         Apply ruff format & autofixes"
	@echo "  make train          Train all pipelines on synthetic data"
	@echo "  make download-tep   Fetch the real Tennessee Eastman dataset"
	@echo "  make notebook       Launch Jupyter notebook server"
	@echo "  make app            Launch the Streamlit dashboard"
	@echo "  make clean          Remove build/test caches and virtualenv"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel

install: $(BIN)/python
	$(PIP) install -e ".[app,survival,dev]"

test:
	$(BIN)/pytest -v

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

train:
	$(PY) scripts/train_all.py --data synthetic

download-tep:
	$(PY) scripts/download_tep.py

notebook:
	$(BIN)/jupyter notebook notebooks/

app:
	$(BIN)/streamlit run app/streamlit_app.py

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
