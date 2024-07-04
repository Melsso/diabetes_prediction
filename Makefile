# Define variables
PYTHON = python3
PIP = pip3

# Default target
.DEFAULT_GOAL := help

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Clean generated files
clean:
	rm -rf ./data/processed/*.csv
	rm -rf ./models/*.pkl

# Clean the data
data_clean:
	$(PYTHON) scripts/data_cleaning.py

# Train the model
train:
	$(PYTHON) scripts/train_model.py

# Evaluate the model
evaluate:
	$(PYTHON) scripts/evaluate_model.py

# Run everything
run: clean data_clean train evaluate

# Help target
help:
	@echo "Available targets:"
	@echo "  install    : Install dependencies"
	@echo "  clean      : Clean generated files"
	@echo "  data_clean : Clean the data"
	@echo "  train      : Train the model"
	@echo "  evaluate   : Evaluate the model"
	@echo "  run        : Clean, clean data, train, and evaluate"
	@echo "  help       : Show this help message"
