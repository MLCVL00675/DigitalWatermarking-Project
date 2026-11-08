# Makefile for Frequency-Domain Adaptive Watermarking Benchmark (v4.0)
.PHONY: help install data quick full test audit summary figures clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install   Install exact pinned dependencies via pip"
	@echo "  make data      Download and verify standardized benchmark dataset images"
	@echo "  make quick     Execute quick verification smoke test (5 images, ~2 min)"
	@echo "  make full      Execute full experimental journal replication (208 images)"
	@echo "  make test      Run unit and integration test suite via pytest"
	@echo "  make audit     Verify independent numerical consistency assertions"
	@echo "  make summary   Rebuild RESULTS_SUMMARY.md purely from CSV results"
	@echo "  make figures   Regenerate publication figures and LaTeX tables"
	@echo "  make clean     Remove temporary build artifacts and cache files"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_datasets.py --workers 6

quick:
	$(PYTHON) run_all_v4.py --quick --results-dir results_v4

full:
	$(PYTHON) run_all_v4.py --full --results-dir results_v4 --workers 8

test:
	$(PYTHON) -m pytest experiments_v2/ -v

audit:
	$(PYTHON) check_consistency.py --results-dir results_v4

summary:
	$(PYTHON) -c "from experiments_v2.make_summary import generate_results_summary; generate_results_summary('results_v4', 'results_v4/RESULTS_SUMMARY.md')"

figures:
	$(PYTHON) -m experiments_v2.make_figures --results-dir results_v4

clean:
	$(PYTHON) -c "import shutil, os, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('.pytest_cache')]"

