"""Local DatasetProfile validation and identifier-free Code target snapshots."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from vision_research_ops.domain import DatasetProfile, JsonObject


class TargetProfileError(ValueError):
    """Safe failure without source paths or validation payloads."""


class SafeTargetProfile(BaseModel):
    """Allowlisted numeric/enum facts; never dataset or label identifiers."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    provenance: Literal["configured_local_profile"] = "configured_local_profile"
    task: Literal["IMAGE_CLASSIFICATION"] = "IMAGE_CLASSIFICATION"
    modality: Literal["RGB", "GRAYSCALE", "PSEUDOCOLOR", "OTHER"]
    input_channels: int = Field(ge=1, le=4)
    height: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1)
    class_count: int = Field(ge=1)
    class_counts: list[int] = Field(default_factory=list, max_length=1000)
    total_samples: int | None = Field(default=None, ge=0)
    train_samples: int | None = Field(default=None, ge=0)
    validation_samples: int | None = Field(default=None, ge=0)
    test_samples: int | None = Field(default=None, ge=0)
    split_strategy: Literal[
        "GROUP_HOLDOUT", "TIME_EXTRAPOLATION", "DOMAIN_HOLDOUT", "SAMPLE_STRATIFIED"
    ]
    resize: int | None = Field(default=None, ge=1)
    normalization: Literal["unit_interval", "none", "standardize"] | None = None
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)
    test_fraction: float | None = Field(default=None, gt=0, lt=1)

    @property
    def snapshot_ref(self) -> str:
        """Content-address only the projected facts, not private source metadata."""
        digest = sha256(self.model_dump_json().encode()).hexdigest()
        return f"target-profiles/{digest}.json"

    def tool_facts(self) -> JsonObject:
        """Return a fresh JSON-safe projection for a read-only Code tool."""
        return dict(self.model_dump(mode="json"))


def load_target_profile(path: Path) -> SafeTargetProfile:
    """Validate a generated profile locally; never read images or discover a root."""
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("oversized profile")
        profile = DatasetProfile.model_validate_json(path.read_bytes())
        if profile.task_type.value != "IMAGE_CLASSIFICATION":
            raise ValueError("unsupported task")

        def count(value: object) -> int | None:
            return value if type(value) is int and value >= 0 else None

        counts = profile.sample_counts
        splits = counts.get("by_split")
        splits = splits if isinstance(splits, dict) else {}
        labels = counts.get("by_label")
        class_counts = (
            sorted(value for value in labels.values() if type(value) is int and value >= 0)
            if isinstance(labels, dict)
            else []
        )
        return SafeTargetProfile(
            modality=profile.modality,
            input_channels=profile.channels,
            height=count(profile.image_shape_policy.get("height")),
            width=count(profile.image_shape_policy.get("width")),
            class_count=len(profile.label_schema),
            class_counts=class_counts,
            total_samples=count(counts.get("total")),
            train_samples=count(splits.get("train")),
            validation_samples=count(splits.get("validation")),
            test_samples=count(splits.get("test")),
            split_strategy=profile.split_policy.strategy.value,
            validation_fraction=profile.split_policy.validation_fraction,
            test_fraction=profile.split_policy.test_fraction,
            resize=count(profile.preprocessing_contract.get("resize")),
            normalization=(
                cast(
                    Literal["unit_interval", "none", "standardize"],
                    profile.preprocessing_contract.get("normalization"),
                )
                if profile.preprocessing_contract.get("normalization")
                in ("unit_interval", "none", "standardize")
                else None
            ),
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        raise TargetProfileError("DATASET_PROFILE_INVALID_OR_MISSING") from None
