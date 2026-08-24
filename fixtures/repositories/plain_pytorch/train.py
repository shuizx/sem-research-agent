"""Synthetic PyTorch entrypoint inspected as text and not run by adaptation smoke."""

from data import DefectDataset
from model import DefectClassifier
from torch import nn
from torch.utils.data import DataLoader


def train() -> None:
    loader = DataLoader(DefectDataset(), batch_size=4)
    model = DefectClassifier(num_classes=3)
    loss_function = nn.CrossEntropyLoss()
    for samples, labels in loader:
        loss_function(model(samples), labels)


if __name__ == "__main__":
    train()
