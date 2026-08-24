"""Synthetic classifier signature used for static PLAIN_PYTORCH profiling only."""

from torch import nn


class DefectClassifier(nn.Module):
    """Tiny classifier head; adaptation never imports this module without Torch."""

    def __init__(self, num_classes: int, channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.classifier = nn.Linear(16, num_classes)
