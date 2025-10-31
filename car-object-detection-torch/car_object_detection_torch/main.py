import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from car_object_detection_torch.dataset import CarDetectionDataset, collate_fn
from car_object_detection_torch.logging_config import get_logger, setup_logging
from car_object_detection_torch.model import CarObjectDetectionModel, get_optimizer
from car_object_detection_torch.train import Trainer


def get_config() -> Namespace:
    parser = ArgumentParser(
        prog="car-object-detection", description="Car object detection CLI"
    )

    # Global arguments for logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--seed", type=int, default=4, help="seed for reproduceability"
    )
    train_parser.add_argument(
        "--n-classes", type=int, default=2, help="Number of objects"
    )
    train_parser.add_argument(
        "--img-dims-width",
        type=int,
        default=224,
        help="expected image width dimension for training and inference",
    )
    train_parser.add_argument(
        "--img-dims-height",
        type=int,
        default=224,
        help="expected image height dimension for training and inference",
    )
    train_parser.add_argument(
        "--n-epochs", type=int, default=30, help="number of training epochs"
    )
    train_parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="validation size when splitting training",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.005, help="Learning rate"
    )
    train_parser.add_argument(
        "--manifest-input-path",
        type=str,
        required=True,
        help="metadata surrounding each image",
    )
    train_parser.add_argument(
        "--img-input-path", type=str, required=True, help="Path to training data"
    )
    train_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="whether to use cpu or cuda",
    )
    train_parser.add_argument(
        "--wandb-secret",
        type=str,
        required=True,
        help="wandb secret in order to log model output",
    )

    config = parser.parse_args()

    return config


def bb_split(path: Path, val_size: float, seed) -> tuple[DataFrame, DataFrame]:
    # rows of bounding boxes to img.
    # each image may have multiple bounding boxes.
    manifest_df = read_csv(path)
    train_ids, val_ids = train_test_split(
        manifest_df["image"].unique().tolist(),
        test_size=val_size,
        random_state=seed,
    )

    # split up the data according to the image files. since there's potentially
    # multiple bounding boxes per image this should be split according to image id
    train_df = manifest_df[manifest_df["image"].isin(train_ids)]
    val_df = manifest_df[manifest_df["image"].isin(val_ids)]

    assert len(train_df) + len(val_df) == len(manifest_df), (
        f"Data split mismatch {len(train_df)} + {len(val_df)} /= {len(manifest_df)}"
    )

    return train_df, val_df


def get_dataloader(
    df: DataFrame,
    batch_size: int,
    img_path: Path,
    img_dims: tuple[int, int],
    img_mapping: list[str],
    device: str,
) -> DataLoader:
    loader = DataLoader(
        CarDetectionDataset(
            df=df,
            img_dir=img_path,
            img_dims=img_dims,
            img_mapping=img_mapping,
            device=device,
        ),
        batch_size=batch_size,
        # TODO: add the appropriate type later
        collate_fn=lambda batch: collate_fn(batch, device=device),  # type: ignore
        drop_last=True,
    )
    return loader


def run() -> None:
    # collect application tunable configuration
    config = get_config()

    # setup logging
    setup_logging(config.log_level)
    logger = get_logger(__name__)

    # configure reproduceability
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # setup configuration for cuda-specific
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Starting car object detection application")
    logger.info(f"Command: {config.command}")
    logger.debug(f"Full config: {config}")

    if config.command == "train":
        try:
            # step 1: load train split manifest containing bounding boxes
            train_df, val_df = bb_split(
                config.manifest_input_path, config.val_size, config.seed
            )
            logger.info(f"Train Size: {len(train_df)}")
            logger.info(f"validation size: {len(val_df)}")

            # step 2: create dataloaders for images
            logger.info("building data loaders")
            train_loader = get_dataloader(
                train_df,
                config.batch_size,
                config.img_input_path,
                (config.img_dims_width, config.img_dims_height),
                train_df["image"].unique().tolist(),
                config.device,
            )
            val_loader = get_dataloader(
                val_df,
                config.batch_size,
                config.img_input_path,
                (config.img_dims_width, config.img_dims_height),
                val_df["image"].unique().tolist(),
                config.device,
            )

            # step 3: create model and optimizer for bounding box
            logger.info("setting up pre-loaded model")
            model = CarObjectDetectionModel(2, config.device)
            optimizer = get_optimizer(model, config.learning_rate)

            # step 4: model training loop
            trainer = Trainer(
                train_loader,
                val_loader,
                model,
                optimizer,
            )
            trainer.train(config.n_epochs, config.device)
            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    else:
        logger.warning(f"Command '{config.command}' not yet implemented")

    logger.info("Application finished")
