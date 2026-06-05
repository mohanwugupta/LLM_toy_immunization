.PHONY: test install clean

test:
	pytest tests/ -v

install:
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
