.DEFAULT_GOAL := quality
.PHONY: quality format clean

CHECK_DIRS := segment

# Check that source code meets quality standards

quality:
	black --check $(CHECK_DIRS)
	isort --check-only $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)

# Format source code automatically and check is there are any problems left that need manual fixing

format:
	black $(CHECK_DIRS)
	isort $(CHECK_DIRS)

clean:
	find . | grep -E '(\.mypy_cache|.DS_Store|__pycache__|.ipynb_checkpoints|\.pyc|.pyc|\.pyo$$)' | xargs rm -rf
