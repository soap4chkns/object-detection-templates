from pathlib import Path

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import wandb
from car_object_detection_torch.logging_config import get_logger
from car_object_detection_torch.model import CarObjectDetectionModel

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: CarObjectDetectionModel,
        optimizer: SGD,
        wandb_config: dict[str, float],
        model_output_path: Path,
        test_run: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer
        self.wandb_config = wandb_config

        self.model_output_path = model_output_path
        self.test_run = test_run

    def train(self, n_epochs: int, device: str) -> CarObjectDetectionModel:
        best_valid_loss = float("inf")
        patience_counter = 0

        with wandb.init(
            project="car-object-detection-torch", name="", config=self.wandb_config
        ) as run:
            for epoch in range(n_epochs):
                logger.info(f"Epoch: {epoch + 1}")
                # primary training loop
                for i, (imgs, targets) in enumerate(self.train_dataloader):
                    # ensure model is in training mode
                    self.model.train()

                    self.optimizer.zero_grad()
                    loss, losses = self.model(imgs, targets)
                    run.log(
                        {
                            "train-loss": loss,
                            "train-location-loss": losses["loss_classifier"],
                            "train-regr-loss": losses["loss_box_reg"],
                            "train-objectness-loss": losses["loss_objectness"],
                            "train-rpn-box-reg-loss": losses["loss_rpn_box_reg"],
                        }
                    )

                    # python lsp interprets as int for some reason
                    loss.backward()  # type: ignore
                    self.optimizer.step()

                    if self.test_run:
                        torch.save(self.model.state_dict(), self.model_output_path)
                        return self.model

                # validation loop
                for i, (imgs, targets) in enumerate(self.val_dataloader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    loss, losses = self.model(imgs, targets)
                    run.log(
                        {
                            "val-loss": loss,
                            "val-location-loss": losses["loss_classifier"],
                            "val-regr-loss": losses["loss_box_reg"],
                            "val-objectness-loss": losses["loss_objectness"],
                            "val-rpn-box-reg-loss": losses["loss_rpn_box_reg"],
                        }
                    )
                    if loss < best_valid_loss:
                        best_valid_loss = loss
                        torch.save(self.model.state_dict(), self.model_output_path)
                        run.log_model(
                            path=self.model_output_path,
                            name=str(self.model_output_path),
                            aliases=["best"],
                        )
                        patience_counter = 0
                    else:
                        patience_counter += 1

        self.model.load_state_dict(torch.load(self.model_output_path))
        return self.model
