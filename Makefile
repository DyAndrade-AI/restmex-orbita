PYTHON ?= .venv/bin/python

.PHONY: setup build build-semantic app web web-api docker-up docker-down docker-logs test evaluate evaluate-full evaluate-semantic evaluate-futures benchmark-api smoke-e2e experiments audit-multiverse demo

# Python 3.9 de Command Line Tools: numpy 2.0.2 (versión de los artefactos) no tiene wheel para 3.13.
BOOT_PYTHON ?= /usr/bin/python3

setup:
	$(BOOT_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -c docker/requirements-api.txt -e '.[dev]'

build:
	$(PYTHON) scripts/build_index.py

build-semantic:
	$(PYTHON) scripts/build_index.py --semantic

app:
	$(PYTHON) -m streamlit run app.py

web:
	cd web && npm run dev

web-api:
	$(PYTHON) scripts/serve_api.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

test:
	$(PYTHON) -m pytest -q

evaluate:
	$(PYTHON) scripts/evaluate.py

evaluate-full:
	$(PYTHON) scripts/evaluate.py --holdout

evaluate-semantic:
	$(PYTHON) scripts/evaluate.py --semantic --holdout

benchmark-api:
	$(PYTHON) scripts/benchmark_api.py

smoke-e2e:
	$(PYTHON) scripts/smoke_e2e.py

experiments:
	$(PYTHON) scripts/experiments.py

audit-multiverse:
	$(PYTHON) scripts/audit_multiverse.py

evaluate-futures:
	$(PYTHON) scripts/evaluate_futures.py

demo:
	$(PYTHON) scripts/smoke_demo.py

