"""Strict local inspection for an explicitly supplied image-classification dataset."""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vision_research_ops.domain import (
    DatasetProfile,
    JsonObject,
    LabelSpec,
    SplitPolicy,
    SplitStrategy,
    TaskType,
)

_TOKEN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
_CSV_HEADERS = ["relative_path", "source_label", "split", "group_id"]
_ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


class DatasetProfileError(ValueError):
    """Stable, path-free error raised for an invalid local dataset contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ManifestLabel(_ManifestModel):
    source_key: str = Field(min_length=1, max_length=64)
    label_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)

    @field_validator("source_key", "label_id", "name")
    @classmethod
    def _safe_label_token(cls, value: str) -> str:
        if value != value.strip() or any(character not in _TOKEN + " " for character in value):
            raise ValueError("label mapping must use a bounded token")
        return value


class _ManifestSplitPolicy(_ManifestModel):
    strategy: Literal["GROUP_HOLDOUT"]
    group_key: str = Field(min_length=1, max_length=64)
    seed: int = Field(ge=0)
    test_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)

    @field_validator("group_key")
    @classmethod
    def _safe_group_key(cls, value: str) -> str:
        if value != value.strip() or any(character not in _TOKEN for character in value):
            raise ValueError("group key must use a bounded token")
        return value


class _ManifestPreprocessing(_ManifestModel):
    normalization: Literal["unit_interval"]
    resize: int = Field(ge=1, le=8192)


class DatasetManifest(_ManifestModel):
    """Small, strict user-supplied manifest. Source keys never leave local inspection."""

    schema_version: Literal["1"]
    dataset_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    created_at: datetime
    source_kind: Literal["SYNTHETIC", "USER_AUTHORIZED"]
    task_type: Literal["IMAGE_CLASSIFICATION"]
    modality: Literal["GRAYSCALE", "RGB"]
    channels: Literal[1, 3]
    labels: list[_ManifestLabel] = Field(min_length=2, max_length=64)
    split_policy: _ManifestSplitPolicy
    preprocessing_contract: _ManifestPreprocessing

    @field_validator("dataset_id", "version")
    @classmethod
    def _safe_storage_token(cls, value: str) -> str:
        if value != value.strip() or any(character not in _TOKEN for character in value):
            raise ValueError("dataset storage identity must use a bounded token")
        return value

    @field_validator("display_name")
    @classmethod
    def _safe_display_name(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("display name must be plain text")
        return value


@dataclass(frozen=True, slots=True)
class DatasetProfileResult:
    """A profile plus only its safe local-artifact reference and aggregate count."""

    profile: DatasetProfile
    profile_ref: str
    output_path: Path
    sample_count: int


@dataclass(frozen=True, slots=True)
class _Sample:
    relative_path: str
    label_id: str
    split: str
    group_digest: str
    image_hash: str
    width: int
    height: int
    channels: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _fail_if_symlink(path: Path) -> None:
    if path.is_symlink():
        raise DatasetProfileError("DATASET_SYMLINK_REJECTED")


def _safe_relative_path(value: str) -> tuple[str, ...]:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or ":" in value
        or "%" in value
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise DatasetProfileError("DATASET_PATH_INVALID")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} or part != part.strip() for part in parts):
        raise DatasetProfileError("DATASET_PATH_INVALID")
    return parts


def _read_manifest(root: Path) -> DatasetManifest:
    path = root / "dataset.json"
    _fail_if_symlink(path)
    if not path.is_file():
        raise DatasetProfileError("DATASET_MANIFEST_MISSING")
    try:
        manifest = DatasetManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError):
        raise DatasetProfileError("DATASET_MANIFEST_INVALID") from None
    if manifest.created_at.tzinfo is None or manifest.created_at.utcoffset() is None:
        raise DatasetProfileError("DATASET_MANIFEST_INVALID")
    if manifest.channels != (1 if manifest.modality == "GRAYSCALE" else 3):
        raise DatasetProfileError("DATASET_MANIFEST_INVALID")
    source_keys = [item.source_key for item in manifest.labels]
    label_ids = [item.label_id for item in manifest.labels]
    names = [item.name for item in manifest.labels]
    if len(set(source_keys)) != len(source_keys) or len(set(label_ids)) != len(label_ids):
        raise DatasetProfileError("DATASET_MANIFEST_INVALID")
    if len(set(names)) != len(names):
        raise DatasetProfileError("DATASET_MANIFEST_INVALID")
    if manifest.split_policy.test_fraction + manifest.split_policy.validation_fraction >= 1:
        raise DatasetProfileError("DATASET_MANIFEST_INVALID")
    return manifest


def _parse_pgm(payload: bytes) -> tuple[int, int, int]:
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(payload) and payload[index] in b" \t\r\n":
            index += 1
        if index < len(payload) and payload[index] == ord("#"):
            while index < len(payload) and payload[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(payload) and payload[index] not in b" \t\r\n#":
            index += 1
        if start == index:
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        tokens.append(payload[start:index])
    if tokens[0] not in {b"P2", b"P5"}:
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    try:
        width, height, maximum = (int(item) for item in tokens[1:])
    except ValueError:
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED") from None
    if width < 1 or height < 1 or maximum < 1 or maximum > 255:
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    if tokens[0] == b"P5":
        if index >= len(payload) or payload[index] not in b" \t\r\n":
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        if payload[index : index + 2] == b"\r\n":
            index += 2
        else:
            index += 1
        if len(payload) - index != width * height:
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    else:
        body = payload[index:].split()
        if len(body) != width * height:
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        try:
            if any(int(item) < 0 or int(item) > maximum for item in body):
                raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        except ValueError:
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED") from None
    return width, height, 1


def _parse_png(payload: bytes) -> tuple[int, int, int]:
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13 or payload[12:16] != b"IHDR":
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", payload[16:29]
    )
    if (
        width < 1
        or height < 1
        or bit_depth != 8
        or color_type not in {0, 2}
        or compression != 0
        or filter_method != 0
        or interlace != 0
    ):
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    position = 8
    saw_iend = False
    saw_idat = False
    while position < len(payload):
        if position + 12 > len(payload):
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        chunk_length = struct.unpack(">I", payload[position : position + 4])[0]
        chunk_type = payload[position + 4 : position + 8]
        position += 12 + chunk_length
        if position > len(payload):
            raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
        if chunk_type == b"IEND":
            if chunk_length != 0 or position != len(payload):
                raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
            saw_iend = True
            break
        if chunk_type == b"IDAT":
            saw_idat = True
    if not saw_iend or not saw_idat:
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    return width, height, 1 if color_type == 0 else 3


def _read_image(path: Path) -> tuple[str, int, int, int]:
    _fail_if_symlink(path)
    try:
        payload = path.read_bytes()
    except OSError:
        raise DatasetProfileError("DATASET_IMAGE_MISSING") from None
    suffix = path.suffix.casefold()
    if suffix == ".pgm":
        width, height, channels = _parse_pgm(payload)
    elif suffix == ".png":
        width, height, channels = _parse_png(payload)
    else:
        raise DatasetProfileError("DATASET_IMAGE_UNSUPPORTED")
    return _sha256(payload), width, height, channels


def _read_samples(root: Path, manifest: DatasetManifest) -> list[_Sample]:
    path = root / "samples.csv"
    _fail_if_symlink(path)
    if not path.is_file():
        raise DatasetProfileError("DATASET_SAMPLES_MISSING")
    labels = {item.source_key: item.label_id for item in manifest.labels}
    samples: list[_Sample] = []
    seen_paths: set[str] = set()
    groups: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != _CSV_HEADERS:
                raise DatasetProfileError("DATASET_SAMPLES_HEADER_INVALID")
            for row in reader:
                if set(row) != set(_CSV_HEADERS) or any(value is None for value in row.values()):
                    raise DatasetProfileError("DATASET_SAMPLES_INVALID")
                relative_path = row["relative_path"]
                source_label = row["source_label"]
                split = row["split"]
                group_id = row["group_id"]
                if relative_path in seen_paths or not source_label or not group_id:
                    raise DatasetProfileError("DATASET_SAMPLES_INVALID")
                seen_paths.add(relative_path)
                parts = _safe_relative_path(relative_path)
                if parts[0] != "images" or Path(parts[-1]).suffix.casefold() not in {
                    ".pgm",
                    ".png",
                }:
                    raise DatasetProfileError("DATASET_PATH_INVALID")
                if source_label not in labels:
                    raise DatasetProfileError("DATASET_LABEL_UNKNOWN")
                if split not in _ALLOWED_SPLITS:
                    raise DatasetProfileError("DATASET_SPLIT_INVALID")
                prior = groups.setdefault(group_id, split)
                if prior != split:
                    raise DatasetProfileError("DATASET_GROUP_CROSSES_SPLIT")
                image_path = root.joinpath(*parts)
                _fail_if_symlink(image_path)
                resolved = image_path.resolve()
                if not resolved.is_relative_to(root):
                    raise DatasetProfileError("DATASET_PATH_INVALID")
                image_hash, width, height, channels = _read_image(image_path)
                samples.append(
                    _Sample(
                        relative_path=relative_path,
                        label_id=labels[source_label],
                        split=split,
                        group_digest=_sha256(group_id.encode("utf-8")),
                        image_hash=image_hash,
                        width=width,
                        height=height,
                        channels=channels,
                    )
                )
    except (OSError, UnicodeError, csv.Error):
        raise DatasetProfileError("DATASET_SAMPLES_INVALID") from None
    if not samples:
        raise DatasetProfileError("DATASET_SAMPLES_INVALID")
    return samples


def _validate_tree(root: Path, samples: list[_Sample]) -> None:
    allowed_top_level = {"dataset.json", "samples.csv", "images"}
    observed_paths: set[str] = set()
    for entry in root.iterdir():
        _fail_if_symlink(entry)
        if entry.name not in allowed_top_level:
            raise DatasetProfileError("DATASET_LAYOUT_INVALID")
    images = root / "images"
    _fail_if_symlink(images)
    if not images.is_dir():
        raise DatasetProfileError("DATASET_IMAGES_MISSING")
    for entry in images.rglob("*"):
        _fail_if_symlink(entry)
        if entry.is_file():
            relative = entry.relative_to(root).as_posix()
            _safe_relative_path(relative)
            observed_paths.add(relative)
        elif not entry.is_dir():
            raise DatasetProfileError("DATASET_LAYOUT_INVALID")
    expected_paths = {sample.relative_path for sample in samples}
    if observed_paths != expected_paths:
        raise DatasetProfileError("DATASET_IMAGE_SET_MISMATCH")


def _profile_payload(
    manifest: DatasetManifest, samples: list[_Sample]
) -> tuple[DatasetProfile, int]:
    shapes = {(sample.width, sample.height) for sample in samples}
    channels = {sample.channels for sample in samples}
    if len(shapes) != 1 or channels != {manifest.channels}:
        raise DatasetProfileError("DATASET_IMAGE_CONTRACT_MISMATCH")
    width, height = next(iter(shapes))
    by_label = {label.label_id: 0 for label in manifest.labels}
    by_split = {split: 0 for split in sorted(_ALLOWED_SPLITS)}
    for sample in samples:
        by_label[sample.label_id] += 1
        by_split[sample.split] += 1
    if any(count == 0 for count in by_label.values()):
        raise DatasetProfileError("DATASET_LABEL_EMPTY")
    canonical_metadata = {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "display_name": manifest.display_name,
        "created_at": manifest.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source_kind": manifest.source_kind,
        "task_type": manifest.task_type,
        "modality": manifest.modality,
        "channels": manifest.channels,
        "labels": [{"label_id": label.label_id, "name": label.name} for label in manifest.labels],
        "split_policy": manifest.split_policy.model_dump(mode="json"),
        "preprocessing_contract": manifest.preprocessing_contract.model_dump(mode="json"),
        "image_shape_policy": {"height": height, "width": width},
        "sample_counts": {
            "by_label": by_label,
            "by_split": by_split,
            "groups": len({s.group_digest for s in samples}),
            "total": len(samples),
        },
    }
    image_records = sorted(
        [
            {
                "image_hash": sample.image_hash,
                "label_id": sample.label_id,
                "split": sample.split,
                "group_digest": sample.group_digest,
            }
            for sample in samples
        ],
        key=_canonical_bytes,
    )
    digest = _sha256(_canonical_bytes({"metadata": canonical_metadata, "images": image_records}))
    location_digest = _sha256(_canonical_bytes(canonical_metadata))[:16]
    try:
        profile = DatasetProfile(
            schema_version="1",
            dataset_id=manifest.dataset_id,
            version=manifest.version,
            display_name=manifest.display_name,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            modality=manifest.modality,
            channels=manifest.channels,
            image_shape_policy={"height": height, "width": width},
            label_schema=[
                LabelSpec(
                    schema_version="1",
                    label_id=label.label_id,
                    name=label.name,
                    is_unknown=False,
                )
                for label in manifest.labels
            ],
            sample_counts=cast(JsonObject, canonical_metadata["sample_counts"]),
            group_keys=[manifest.split_policy.group_key],
            split_policy=SplitPolicy(
                schema_version="1",
                strategy=SplitStrategy.GROUP_HOLDOUT,
                group_keys=[manifest.split_policy.group_key],
                holdout_values={},
                test_fraction=manifest.split_policy.test_fraction,
                validation_fraction=manifest.split_policy.validation_fraction,
                seed=manifest.split_policy.seed,
            ),
            location_ref=f"dataset-handle-{location_digest}",
            content_hash=f"sha256:{digest}",
            authorization={"profile_use_allowed": True, "source_kind": manifest.source_kind},
            preprocessing_contract=manifest.preprocessing_contract.model_dump(mode="json"),
            created_at=manifest.created_at,
        )
    except (ValidationError, ValueError, TypeError):
        raise DatasetProfileError("DATASET_MANIFEST_INVALID") from None
    return profile, len(samples)


def profile_dataset(
    dataset_root: Path | None,
    *,
    output_root: Path = Path("var/dataset-profiles"),
) -> DatasetProfileResult:
    """Profile one explicit local root without serializing private paths or sample identifiers."""
    if dataset_root is None or not str(dataset_root).strip():
        raise DatasetProfileError("DATASET_ROOT_REQUIRED")
    root = Path(dataset_root)
    _fail_if_symlink(root)
    if not root.is_dir():
        raise DatasetProfileError("DATASET_ROOT_INVALID")
    try:
        root = root.resolve()
    except OSError:
        raise DatasetProfileError("DATASET_ROOT_INVALID") from None
    manifest = _read_manifest(root)
    samples = _read_samples(root, manifest)
    _validate_tree(root, samples)
    profile, sample_count = _profile_payload(manifest, samples)
    profile_ref = f"dataset-profiles/{profile.dataset_id}/{profile.version}/profile.json"
    target = Path(output_root) / profile.dataset_id / profile.version / "profile.json"
    payload = profile.model_dump_json(indent=2).encode("utf-8") + b"\n"
    if target.exists():
        _fail_if_symlink(target)
        try:
            existing = DatasetProfile.model_validate_json(target.read_bytes())
        except (OSError, ValidationError, ValueError):
            raise DatasetProfileError("DATASET_PROFILE_CONFLICT") from None
        if existing.content_hash != profile.content_hash or target.read_bytes() != payload:
            raise DatasetProfileError("DATASET_PROFILE_CONFLICT")
    else:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        except OSError:
            raise DatasetProfileError("DATASET_PROFILE_WRITE_FAILED") from None
    return DatasetProfileResult(profile, profile_ref, target, sample_count)


__all__ = ["DatasetManifest", "DatasetProfileError", "DatasetProfileResult", "profile_dataset"]
