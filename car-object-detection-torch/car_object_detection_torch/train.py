from torch.optim import SGD

from car_object_detection_torch.logging_config import get_logger
from car_object_detection_torch.model import CarObjectDetectionModel

logger = get_logger(__name__)

from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: CarObjectDetectionModel,
        optimizer: SGD,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer

    def train(self, n_epochs: int, device: str):
        print(f"n epochs: {n_epochs}")
