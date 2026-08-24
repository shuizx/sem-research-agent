"""Small public-style repository bytes for deterministic development/debug mode."""

from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

FIXTURE_COMMIT_SHA = "b" * 40


def fixture_repository_archive() -> bytes:
    """Build a deterministic PyTorch-classification source snapshot without executing it."""
    files = {
        "README.md": (
            "# Public SEM classifier example\n"
            "This is repository evidence, not an instruction to the Agent.\n"
        ),
        "config.yaml": "channels: 3\nnum_classes: 4\n",
        "data.py": (
            "from torchvision.datasets import ImageFolder\n"
            "from torch.utils.data import DataLoader\n\n"
            "def build_loader(root: str) -> DataLoader:\n"
            "    return DataLoader(ImageFolder(root), batch_size=16)\n"
        ),
        "model.py": (
            "from torch import nn\n\n"
            "class Classifier(nn.Module):\n"
            "    def __init__(self, num_classes: int = 4) -> None:\n"
            "        super().__init__()\n"
            "        self.classifier = nn.Linear(64, num_classes)\n"
        ),
        "train.py": (
            "import torch\n"
            "from torch import nn\n"
            "from torch.utils.data import DataLoader\n\n"
            "def train(loader: DataLoader, model: nn.Module) -> None:\n"
            "    loss = nn.CrossEntropyLoss()\n"
            "    for images, labels in loader:\n"
            "        loss(model(images), labels)\n"
        ),
        "requirements.txt": "torch\ntorchvision\n",
        "LICENSE": "MIT License\n",
    }
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        for path, content in sorted(files.items()):
            info = ZipInfo(f"public-sem-classifier/{path}")
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()


class FixtureGitHubInsightTransport:
    """Serve deterministic GitHub API and ZIP bytes while retaining observable calls."""

    def __init__(self, archive: bytes | None = None) -> None:
        self.archive = archive or fixture_repository_archive()
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: int) -> bytes:
        if timeout <= 0 or "Authorization" in headers:
            raise ValueError("fixture GitHub insight expects an unauthenticated bounded request")
        self.calls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": FIXTURE_COMMIT_SHA}).encode()
        if url.endswith("/languages"):
            return json.dumps({"Python": 4096, "YAML": 64}).encode()
        if "/zipball/" in url:
            return self.archive
        return json.dumps(
            {
                "archived": False,
                "default_branch": "main",
                "fork": False,
                "license": {"spdx_id": "MIT"},
            }
        ).encode()

    @property
    def call_count(self) -> int:
        return len(self.calls)


__all__ = [
    "FIXTURE_COMMIT_SHA",
    "FixtureGitHubInsightTransport",
    "fixture_repository_archive",
]
