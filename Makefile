# Define variables
VENV_DIR := venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := source $(VENV_DIR)/bin/activate

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  make help          - Show this help message"
	@echo "  make install       - Create virtual environment and install dependencies"
	@echo "  make run           - Placeholder for running the application"
	@echo "  make clean         - Remove virtual environment and clean up"
	@echo "  make test          - Placeholder for running tests"

# Create virtual environment and install dependencies
.PHONY: install
install: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install numpy==1.24.3 --only-binary=:all:
	$(PIP) install -r requirements.txt --no-deps
	touch $(VENV_DIR)/bin/activate

# Placeholder for running the application
.PHONY: run
run:
	@echo "Run the application (placeholder)"
	# $(ACTIVATE) && $(PYTHON) main.py

# Remove virtual environment and clean up
.PHONY: clean
clean:
	@echo "Cleaning the environment..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Placeholder for running tests
.PHONY: test
test:
	@echo "Run tests (placeholder)"
	# $(ACTIVATE) && pytest
