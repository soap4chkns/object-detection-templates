SHELL := /bin/bash
UV := uv run

PROJECT_DIR := $(shell pwd)
SCRIPTS := $(PROJECT_DIR)/scripts
DATA := $(PROJECT_DIR)/data

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
	$(UV) car-object-detection train \
		--manifest-input-path "$(DATA)/train_solution_bounding_boxes (1).csv" \
		--img-input-path $(DATA)/training_images \
		--model-output-path $(PROJECT_DIR)/models/faster_rcnn_$(shell date +"%Y_%m_%d_%H_%m")_test_run.pth \
		--test-run

train-dev: download-data build-project run-train

clean:
	@echo "cleaning up build"
	rm -rf $(PROJECT_DIR)/*.egg-info
	rm -rf $(PROJECT_DIR)/dist

	@echo "cleaning offline wandb test logging"
	rm -rf $(PROJECT_DIR)/wandb

	@echo "cleaning up data directory"
	rm -rf $(DATA)
