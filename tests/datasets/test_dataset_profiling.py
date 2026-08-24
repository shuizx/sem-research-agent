"""Ordinary acceptance tests for strict local dataset profiling."""

from __future__ import annotations

import json
import shutil
import struct
import zlib
from importlib import import_module
from pathlib import Path

import pytest

from vision_research_ops.application.services.adaptation_planning import adaptation_prompt_facts
from vision_research_ops.application.services.dataset_profiling import (
    DatasetProfileError,
    profile_dataset,
)
from vision_research_ops.cli.dataset_profile import run
from vision_research_ops.settings import Settings


def _sample_root(tmp_path: Path) -> Path:
    source = Path(__file__).parents[2] / "fixtures" / "datasets" / "synthetic_sem_images"
    destination = tmp_path / "authorized-dataset"
    shutil.copytree(source, destination)
    return destination


def _profile(root: Path, output_root: Path):
    return profile_dataset(root, output_root=output_root)


def test_missing_root_and_empty_environment_fail_closed_without_path_leakage(
    monkeypatch, capsys
) -> None:
    """A root is opt-in; empty configuration cannot trigger discovery or diagnostics leaks."""
    assert Settings.from_env({"VRO_DATASET_ROOT": "  "}).dataset_root is None
    monkeypatch.delenv("VRO_DATASET_ROOT", raising=False)
    assert run([]) == 2
    output = capsys.readouterr().out
    assert output == '{"error": "DATASET_ROOT_REQUIRED", "status": "FAILED"}\n'
    assert "C:" not in output


def test_cli_dataset_root_overrides_environment(monkeypatch) -> None:
    """The command line is the explicit highest-priority root selection boundary."""
    module = import_module("vision_research_ops.cli.dataset_profile")
    observed: list[Path | None] = []

    def fail_after_observing(root: Path | None):
        observed.append(root)
        raise DatasetProfileError("DATASET_ROOT_REQUIRED")

    monkeypatch.setenv("VRO_DATASET_ROOT", "environment-root")
    monkeypatch.setattr(module, "profile_dataset", fail_after_observing)
    assert module.run(["--dataset-root", "command-root"]) == 2
    assert observed == [Path("command-root")]


def test_synthetic_images_generate_a_deterministic_profile_and_snapshot(tmp_path: Path) -> None:
    """Counts, shape, channels, labels, and split evidence come from actual fixture files."""
    root = _sample_root(tmp_path)
    first = _profile(root, tmp_path / "profiles")
    first_bytes = first.output_path.read_bytes()
    second = _profile(root, tmp_path / "profiles")
    expected = json.loads(
        (
            Path(__file__).parents[2] / "fixtures" / "datasets" / "synthetic_sem_profile.json"
        ).read_text(encoding="utf-8")
    )
    assert first.profile.model_dump(mode="json") == expected
    assert second.output_path.read_bytes() == first_bytes
    assert first.profile.sample_counts == {
        "by_label": {
            "label-bridge": 2,
            "label-particle": 2,
            "label-scratch": 2,
            "label-void": 2,
        },
        "by_split": {"test": 2, "train": 4, "validation": 2},
        "groups": 8,
        "total": 8,
    }
    assert first.profile.image_shape_policy == {"height": 2, "width": 2}
    assert first.profile.channels == 1
    assert first.profile.location_ref.startswith("dataset-handle-")
    assert (
        first.profile_ref
        == "dataset-profiles/dataset-synthetic-sem-1/synthetic-sem-v1/profile.json"
    )


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        ("images/0001.pgm", "0 32 64 96", "1 32 64 96"),
        (
            "samples.csv",
            "images/0001.pgm,source-a,train,group-01",
            "images/0001.pgm,source-b,train,group-01",
        ),
        (
            "samples.csv",
            "images/0001.pgm,source-a,train,group-01",
            "images/0001.pgm,source-a,test,group-01",
        ),
    ],
)
def test_image_label_or_split_change_changes_content_hash(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    """The fingerprint binds image bytes and sanitized label/split metadata."""
    root = _sample_root(tmp_path)
    baseline = _profile(root, tmp_path / "profiles-before")
    target = root / relative
    target.write_text(target.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    changed = _profile(root, tmp_path / "profiles-after")
    assert changed.profile.content_hash != baseline.profile.content_hash


def test_changed_same_identity_refuses_to_overwrite_existing_profile(tmp_path: Path) -> None:
    """A dataset ID/version is immutable once its profile bytes have been recorded."""
    root = _sample_root(tmp_path)
    output_root = tmp_path / "profiles"
    _profile(root, output_root)
    image = root / "images" / "0001.pgm"
    image.write_text(image.read_text(encoding="utf-8").replace("0 32", "1 32"), encoding="utf-8")
    with pytest.raises(DatasetProfileError, match="DATASET_PROFILE_CONFLICT"):
        _profile(root, output_root)


def test_png_image_header_is_profiled_with_the_declared_channel_count(tmp_path: Path) -> None:
    """PNG is accepted alongside PGM without adding an image-processing dependency."""
    root = _sample_root(tmp_path)
    png = root / "images" / "0001.png"
    header = struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x20\x00\x40\x60"))
        + chunk(b"IEND", b"")
    )
    (root / "images" / "0001.pgm").unlink()
    samples = root / "samples.csv"
    samples.write_text(
        samples.read_text(encoding="utf-8").replace("images/0001.pgm", "images/0001.png"),
        encoding="utf-8",
    )
    profile = _profile(root, tmp_path / "profiles").profile
    assert profile.channels == 1
    assert profile.image_shape_policy == {"height": 2, "width": 2}


def test_symlink_is_rejected_before_a_profile_is_written(tmp_path: Path) -> None:
    """The local file boundary does not follow a sample symlink."""
    root = _sample_root(tmp_path)
    link = root / "images" / "linked.pgm"
    try:
        link.symlink_to(root / "images" / "0001.pgm")
    except OSError:
        pytest.skip("symlink creation is unavailable in this Windows test environment")
    samples = root / "samples.csv"
    samples.write_text(
        samples.read_text(encoding="utf-8").replace("images/0001.pgm", "images/linked.pgm"),
        encoding="utf-8",
    )
    output_root = tmp_path / "profiles"
    with pytest.raises(DatasetProfileError, match="DATASET_SYMLINK_REJECTED"):
        _profile(root, output_root)
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (
            lambda root: (root / "samples.csv").write_text("bad\n", encoding="utf-8"),
            "DATASET_SAMPLES_HEADER_INVALID",
        ),
        (
            lambda root: (root / "samples.csv").write_text(
                (root / "samples.csv").read_text(encoding="utf-8").replace("source-a", "unknown"),
                encoding="utf-8",
            ),
            "DATASET_LABEL_UNKNOWN",
        ),
        (
            lambda root: (root / "samples.csv").write_text(
                (root / "samples.csv")
                .read_text(encoding="utf-8")
                .replace("images/0001.pgm", "images/../0001.pgm"),
                encoding="utf-8",
            ),
            "DATASET_PATH_INVALID",
        ),
        (
            lambda root: (root / "samples.csv").write_text(
                (root / "samples.csv")
                .read_text(encoding="utf-8")
                .replace(
                    "images/0005.pgm,source-a,validation,group-05",
                    "images/0005.pgm,source-a,validation,group-01",
                ),
                encoding="utf-8",
            ),
            "DATASET_GROUP_CROSSES_SPLIT",
        ),
    ],
)
def test_ordinary_invalid_dataset_inputs_fail_without_profile(
    tmp_path: Path, mutator, code: str
) -> None:
    """Malformed normal inputs stop before a success artifact is written."""
    root = _sample_root(tmp_path)
    mutator(root)
    output_root = tmp_path / "profiles"
    with pytest.raises(DatasetProfileError, match=code):
        _profile(root, output_root)
    assert not output_root.exists()


def test_profile_and_llm_facts_exclude_private_sample_identifiers(tmp_path: Path) -> None:
    """Neither persisted profile JSON nor ToolNode facts reveal local sample metadata."""
    root = _sample_root(tmp_path)
    result = _profile(root, tmp_path / "profiles")
    profile_json = result.output_path.read_text(encoding="utf-8")
    from vision_research_ops.application.services.adaptation_models import AdaptationInputFacts

    facts = AdaptationInputFacts(
        repository_id="repo-1",
        repository_url="https://github.com/example/sem-classifier",
        base_commit_sha="a" * 40,
        structure_type="PLAIN_PYTORCH",
        license_spdx="MIT",
        dataset_id=result.profile.dataset_id,
        dataset_version=result.profile.version,
        dataset_content_hash=result.profile.content_hash,
        modality="GRAYSCALE",
        channels=1,
        label_names=[item.name for item in result.profile.label_schema],
        group_keys=result.profile.group_keys,
        group_split_key=result.profile.split_policy.group_keys[0],
        dataset_kind="SYNTHETIC_SEM_FIXTURE",
        repository_kind="CONTROLLED_PLAIN_PYTORCH_FIXTURE",
    )
    prompt_json = json.dumps(adaptation_prompt_facts(facts), sort_keys=True)
    for forbidden in (str(root), "images/0001.pgm", "source-a", "group-01"):
        assert forbidden not in profile_json
        assert forbidden not in prompt_json
