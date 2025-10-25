install:
	pip install -r requirements.txt

train:
	cd src && python train.py

test:
	python -m pytest tests/ -v

lint:
	black src/ tests/ --check

format:
	black src/ tests/

run-pipeline: install train

.PHONY: install train test lint format run-pipeline