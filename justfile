# Run all checks (lint, format, typecheck, test, coverage)
check: lint format-check typecheck test coverage

# Run tests
test:
    uv run -m pytest

# Run tests with coverage
coverage:
    uv run -m pytest --cov=ytch --cov=yjax --cov=ymlx --cov-report=term-missing

# Run a demo
demo name:
    uv run python -m demo.{{name}}

# Lint with ruff
lint:
    uv run ruff check --fix .

# Format with ruff
format:
    uv run ruff format .

# Check formatting without modifying
format-check:
    uv run ruff format --check .

# Type check with ty
typecheck:
    uv run ty check .
