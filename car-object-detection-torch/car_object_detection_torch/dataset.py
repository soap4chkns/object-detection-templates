from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from car_object_detection_torch.logging_config import get_logger

logger = get_logger(__name__)


class CarDetectionDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        img_dir: Path,
        img_dims: tuple[int, int],
        img_mapping: list[str],
        device: str,
    ):
        self.df = df
        self.img_dir = img_dir
        self.img_dims = img_dims
        self.img_mapping = img_mapping
        self.device = device

    def __len__(self) -> int:
        return len(self.img_mapping)

    def __getitem__(self, i: int) -> tuple[Tensor, dict[str, Tensor]]:
        # load image and corresponding bounding boxes
        filename = self.img_mapping[i]
        img_path = self.img_dir.joinpath(filename)
        img = Image.open(img_path).convert("RGB")
        boxes = self.df[self.df["image"] == filename]
        h, w, _ = np.array(img).shape

        # normalize the value of the bounding boxes
        boxes.loc[:, ["xmin", "ymin", "xmax", "ymax"]] /= [w, h, w, h]

        # normalize the value of the images from 0 to 1
        img = np.array(img.resize(self.img_dims, resample=Image.BILINEAR)) / 255.0
        labels = torch.ones(len(boxes)).long()
        boxes = boxes[["xmin", "ymin", "xmax", "ymax"]].values

        # regenerate the dimension of the bouding boxes of the new size
        # they were previously normalized, now re-expanding to the expected dimension
        boxes[:, [0, 2]] *= self.img_dims[0]  # xdim
        boxes[:, [1, 3]] *= self.img_dims[1]  # ydim
        boxes = boxes.astype(np.uint32).tolist()

        # for each img, create boxes and corresponding label
        target = {}
        target["boxes"] = Tensor(boxes).float()
        target["labels"] = labels

        # preprocess img
        img = (
            Tensor(img).permute(2, 0, 1).to(self.device).float()
        )  # swap the channel filter, so it's first

        return img, target


def collate_fn(
    batch: Tensor, device: str
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    imgs, targets = zip(*batch)
    imgs = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return imgs, targets
