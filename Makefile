.PHONY: dev install-uv install-deps install-hooks install-nextflow install-nf-test test test-nf lint format

dev: install-deps install-hooks
	@echo "Dev environment ready."

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

install-deps:
	uv sync --group dev

install-hooks:
	uv run pre-commit install

install-nextflow:
	curl -s https://get.nextflow.io | bash
	chmod +x nextflow
	mv nextflow ~/.local/bin/

install-nf-test:
	curl -fsSL https://code.askimed.com/install/nf-test | bash
	mv nf-test ~/.local/bin/

test:
	uv run pytest

test-nf:
	cd nextflow && nf-test test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .
