install:
	pip install -r requirements.txt

train:
	cd src && python train.py

test:
	python -m pytest tests/ -v

lint:
	black src/ tests/ --check --diff || echo "⚠️  Some files need formatting"

format:
	black src/ tests/

run-pipeline: install train

# Nueva tarea para CI que no falle con formato
ci-lint:
	black src/ tests/ --check

.PHONY: install train test lint format run-pipeline ci-lint