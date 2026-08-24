"""Synthetic dataset signature used for static PLAIN_PYTORCH profiling only."""

from torch.utils.data import Dataset


class DefectDataset(Dataset):
    """Placeholder classification dataset; adaptation never imports this module."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample = [float(index % 2), float((index + 1) % 2), 0.5]
        return sample, index % 3
