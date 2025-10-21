# Run all checks (lint, format, typecheck, test)
check: lint format-check typecheck test

# Run tests
test:
    uv run -m pytest

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
