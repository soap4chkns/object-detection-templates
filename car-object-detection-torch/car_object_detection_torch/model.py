import torchvision
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CarObjectDetectionModel(Module):
    def __init__(self, n_classes: int, device: str, pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.device = device

        # base model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=self.pretrained
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_classes)
        self.model = model.to(self.device)

    def forward(self, input_pack: Tensor) -> int:
        inputs, targets = input_pack
        losses = self.model(inputs, targets)
        loss = sum(loss for loss in losses.values())
        return loss


def get_optimizer(
    model: Module,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
):
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return optimizer
