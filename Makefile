GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type style clean run-benchmarks
format: ## Running code formatter: black and isort
	@echo "(isort) Ordering imports..."
	@isort .
	@echo "(black) Formatting codebase..."
	@black --config pyproject.toml src tests docs
	@echo "(black) Formatting stubs..."
	@find src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config pyproject.toml {} \;
	@echo "(ruff) Running fix only..."
	@ruff check src docs tests --fix-only
format-check:
	@echo "(isort) Checking import order..."
	@isort --check .
	@echo "(black) Checking code formatting..."
	@black --config pyproject.toml --check src tests docs
	@echo "(ruff) Linting development project..."
	@ruff check src docs tests --fix-only
lint: ## Running lint checker: ruff
	@echo "(ruff) Linting development project..."
	@ruff check src docs tests
type: ## Running type checker: pyright
	@echo "(pyright) Typechecking codebase..."
	PYRIGHT_PYTHON_FORCE_VERSION=latest pyright src
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	@cd $(GIT_ROOT)/docs && make clean
	@cd $(GIT_ROOT) || exit 1
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
run-ci: format lint type ## Running all CI checks
test: ## Run tests
	@echo "Running tests..."
	@pytest tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)
test-e2e: ## Run end2end tests
	echo "running end2end tests..."
	@pytest tests/e2e -s
	
# Docs
docs-site: ## Build and serve documentation
	@sphinx-build -nW --keep-going -j 4 -b html $(GIT_ROOT)/docs/ $(GIT_ROOT)/docs/_build/html
	@python -m http.server --directory $(GIT_ROOT)/docs/_build/html
watch-docs: ## Build and watch documentation
	rm -rf $(GIT_ROOT)/docs/_build/{html, jupyter_execute}
	sphinx-autobuild docs docs/_build/html --watch $(GIT_ROOT)/src/ --ignore "_build" --open-browser
rewrite-docs: ## Use GPT4 to rewrite the documentation
	@echo "Rewriting the documentation in directory $(DIR)..."
	@python $(GIT_ROOT)/docs/python alphred.py --directory $(DIR)

# Benchmarks
run-benchmarks-eval: ## Run benchmarks for Evaluation
	@echo "Running benchmarks for Evaluation..."
	@cd $(GIT_ROOT)/tests/benchmarks && python benchmark_eval.py
run-benchmarks-testset: ## Run benchmarks for TestSet Generation
	@echo "Running benchmarks for TestSet Generation..."
	@cd $(GIT_ROOT)/tests/benchmarks && python benchmark_testsetgen.py
run-benchmarks-in-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	@cd $(GIT_ROOT)
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/tests/benchmarks/Dockerfile . 
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si
