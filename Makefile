.PHONY: preprocess train api docker-build docker-run clean

# Variables
PYTHON = python
ConfigFile = configs/default.yaml

preprocess:
	$(PYTHON) src/preprocess_data.py --config $(ConfigFile)

train:
	$(PYTHON) src/train.py --config $(ConfigFile)

infer:
	$(PYTHON) src/infer.py --image $(IMAGE)

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t et-disease-api .

docker-run:
	docker run -p 8000:8000 --gpus all et-disease-api

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
