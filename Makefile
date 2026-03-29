.PHONY: install run dev clean

install:
	pip install -r requirements.txt

run:
	python -m uvicorn server:app --host 0.0.0.0 --port 7000

dev:
	python -m uvicorn server:app --host 0.0.0.0 --port 7000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
