SHELL := /bin/bash
UV := uv run

CWD := $(shell pwd)
SCRIPTS := $(CWD)/scripts
DATA := $(CWD)/data

.PHONY: download-data
download-data:
	@echo "attempting to download kaggle dataset ..."
	$(UV) $(SCRIPTS)/download_data.sh

.PHONY: build-project
build-project:
	@echo "building project"
	uv build

.PHONY: run-train
run-train:
	@echo "training car object detection model"
	$(UV) car-object-detection

train: download-data build-project run-train

clean:
	@echo "cleaning up data directory"
	rm -rf $(DATA)
