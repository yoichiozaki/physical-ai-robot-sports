# Common tasks (run inside the Dev Container).
PY ?= python
SEEDS ?= 0 1 2

.PHONY: help doctor lint format test test-fast smoke train-hybrid train-direct evaluate figures video report all-artifacts

help:
	@echo "doctor | lint | format | test | test-fast | smoke | train-hybrid | train-direct | evaluate | figures | video | report"

doctor:
	$(PY) -m hockey.infrastructure.doctor

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow and not render"

smoke:
	$(PY) scripts/train.py --config configs/training/smoke.yaml --out results/tmp/smoke
	$(PY) scripts/render.py --scene strike --out videos/clips/smoke_strike.mp4 --duration 3

train-hybrid:
	for s in $(SEEDS); do $(PY) scripts/train.py --config configs/training/ppo_hybrid.yaml --seed $$s --run-name ppo_hybrid_s$$s; done

train-direct:
	for s in $(SEEDS); do $(PY) scripts/train.py --config configs/training/ppo_direct.yaml --seed $$s --run-name ppo_direct_s$$s; done

evaluate:
	$(PY) scripts/evaluate.py --config configs/evaluation/main.yaml

figures:
	$(PY) scripts/make_figures.py

video:
	$(PY) scripts/make_video.py

report:
	$(PY) scripts/make_report.py
