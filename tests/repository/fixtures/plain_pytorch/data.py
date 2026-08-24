"""Fixture dataset contract for static inspection only."""

from torch.utils.data import Dataset


class DefectDataset(Dataset):
    """A placeholder dataset shape; repository never instantiates it."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        raise NotImplementedError(index)
