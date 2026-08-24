"""Fixture classifier head for static inspection only."""

from torch import nn


class DefectClassifier(nn.Module):
    """Small plain-PyTorch classifier signature."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(16, num_classes)
