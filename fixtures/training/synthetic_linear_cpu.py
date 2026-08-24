"""Trusted stdlib-only synthetic image classifier used by local training."""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

CAPABILITY = "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"


def _blocked_network(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("network is disabled for the controlled training fixture")


socket.socket = _blocked_network  # type: ignore[assignment]
socket.create_connection = _blocked_network  # type: ignore[assignment]


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


def _file_hash(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    return _hash_bytes(normalized)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fixture JSON root must be an object")
    return value


def _safe_ref(value: str, *, prefix: str | None = None) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or value.startswith("/")
        or ":" in value
        or "%" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("fixture reference is not canonical")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError("fixture reference is outside its fixed prefix")
    return value


def _resolve(root: Path, ref: str) -> Path:
    safe = _safe_ref(ref)
    path = (root / Path(*safe.split("/"))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("fixture reference escaped the project root")
    return path


def _image(label: int, seed: int, height: int, width: int, noise: int) -> list[list[int]]:
    rng = random.Random(seed)
    base = 45 if label != 3 else 185
    pixels = [[base for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for column in range(width):
            if label == 0 and column in {2, 3}:
                pixels[row][column] = 220
            elif label == 1 and row in {2, 3} and column in {4, 5}:
                pixels[row][column] = 235
            elif label == 2 and abs(row - column) <= 1:
                pixels[row][column] = 225
            elif label == 3 and 2 <= row <= 5 and 2 <= column <= 5:
                pixels[row][column] = 25
            pixels[row][column] = max(
                0,
                min(255, pixels[row][column] + rng.randint(-noise, noise)),
            )
    return pixels


def _features(image: list[list[int]], mode: str) -> list[float]:
    normalized = [[pixel / 255.0 for pixel in row] for row in image]
    flat = [pixel for row in normalized for pixel in row]
    if mode == "GLOBAL_STATS":
        mean = sum(flat) / len(flat)
        variance = sum((pixel - mean) ** 2 for pixel in flat) / len(flat)
        return [1.0, mean, variance]
    if mode != "GRID4":
        raise ValueError("unsupported fixed feature mode")
    block_means: list[float] = []
    for block_row in range(4):
        for block_column in range(4):
            values = [
                normalized[row][column]
                for row in range(block_row * 2, block_row * 2 + 2)
                for column in range(block_column * 2, block_column * 2 + 2)
            ]
            block_means.append(sum(values) / len(values))
    return [1.0, *block_means]


def _softmax(scores: list[float]) -> list[float]:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _predict(weights: list[list[float]], features: list[float]) -> list[float]:
    return _softmax(
        [
            sum(weight * value for weight, value in zip(row, features, strict=True))
            for row in weights
        ]
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_fixture(
    root: Path,
    run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    refs_and_hashes = (
        ("dataset_ref", "dataset_ref_hash"),
        ("split_ref", "split_hash"),
        ("preprocess_ref", "preprocess_hash"),
        ("method_config_ref", "method_config_hash"),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for ref_field, hash_field in refs_and_hashes:
        ref = str(run[ref_field])
        path = _resolve(root, ref)
        if not path.is_file() or path.is_symlink() or _file_hash(path) != run[hash_field]:
            raise ValueError("controlled fixture hash mismatch")
        loaded[ref_field] = _load_json(path)
    dataset = loaded["dataset_ref"]
    split = loaded["split_ref"]
    preprocess = loaded["preprocess_ref"]
    config = loaded["method_config_ref"]
    if (
        dataset.get("fixture_kind") != "SYNTHETIC_SEM_IMAGE_RECIPE"
        or dataset.get("synthetic_data_labeled") is not True
        or dataset.get("dataset_id") != run["dataset_id"]
        or dataset.get("dataset_version") != run["dataset_version"]
        or preprocess
        != {
            "channels": 1,
            "decode": "SYNTHETIC_RECIPE_TO_8X8_GRAYSCALE",
            "normalization": "UNIT_INTERVAL",
            "schema_version": "1",
        }
        or config.get("method") != run["method"]
        or config.get("class_count") != 4
        or split.get("dataset_id") != run["dataset_id"]
        or split.get("group_overlap") is not False
    ):
        raise ValueError("controlled training fixture contract mismatch")
    return dataset, split, config


def _train(
    *,
    run: dict[str, Any],
    dataset: dict[str, Any],
    split: dict[str, Any],
    config: dict[str, Any],
    started_monotonic: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], float]:
    generator = dataset["generator"]
    samples = dataset["samples"]
    if not isinstance(samples, list) or len(samples) != 24:
        raise ValueError("synthetic dataset must contain exactly 24 recipes")
    by_id = {str(item["sample_id"]): item for item in samples}
    split_ids = [*split["train"], *split["validation"], *split["test"]]
    if len(set(split_ids)) != 24 or set(split_ids) != set(by_id):
        raise ValueError("fixed split must cover every synthetic sample exactly once")
    groups = [str(by_id[sample_id]["group_id"]) for sample_id in split_ids]
    if len(groups) != len(set(groups)):
        raise ValueError("fixed group split cannot overlap")

    def sample_features(sample_id: str) -> tuple[list[float], int]:
        item = by_id[sample_id]
        label = int(item["label"])
        image = _image(
            label,
            int(item["noise_seed"]),
            int(generator["height"]),
            int(generator["width"]),
            int(generator["noise_amplitude"]),
        )
        return _features(image, str(config["feature_mode"])), label

    feature_count = len(sample_features(str(split["train"][0]))[0])
    rng = random.Random(int(run["seed"]))
    weights = [[rng.uniform(-0.01, 0.01) for _ in range(feature_count)] for _ in range(4)]
    budget = run["budget"]
    max_epochs = int(budget["max_epochs"])
    max_steps = int(budget["max_steps"])
    deadline = float(budget["max_walltime_seconds"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    step_losses: list[dict[str, object]] = []
    epoch_losses: list[dict[str, object]] = []
    total_steps = 0
    for epoch_index in range(1, max_epochs + 1):
        order = [str(value) for value in split["train"]]
        rng.shuffle(order)
        losses: list[float] = []
        for sample_id in order:
            if total_steps >= max_steps:
                break
            if time.monotonic() - started_monotonic >= deadline:
                raise TimeoutError("fixture walltime budget reached")
            features, label = sample_features(sample_id)
            probabilities = _predict(weights, features)
            loss = -math.log(max(probabilities[label], 1e-15))
            total_steps += 1
            losses.append(loss)
            step_losses.append({"epoch": epoch_index, "loss": loss, "step": total_steps})
            for class_index in range(4):
                error = probabilities[class_index] - (1.0 if class_index == label else 0.0)
                for feature_index, feature in enumerate(features):
                    gradient = error * feature + weight_decay * weights[class_index][feature_index]
                    weights[class_index][feature_index] -= learning_rate * gradient
        if losses:
            epoch_losses.append(
                {
                    "epoch": epoch_index,
                    "mean_loss": sum(losses) / len(losses),
                    "steps": len(losses),
                }
            )
        if total_steps >= max_steps:
            break
    predictions: list[dict[str, object]] = []
    correct = 0
    for sample_id in split["test"]:
        features, label = sample_features(str(sample_id))
        scores = _predict(weights, features)
        predicted = max(range(4), key=scores.__getitem__)
        correct += int(predicted == label)
        predictions.append(
            {
                "predicted_label": predicted,
                "sample_id": sample_id,
                "scores": scores,
                "true_label": label,
            }
        )
    return step_losses, epoch_losses, predictions, correct / len(predictions)


def _execute(args: argparse.Namespace) -> None:
    root = Path.cwd().resolve()
    spec_ref = _safe_ref(args.spec_ref, prefix="var/training/")
    output_ref = _safe_ref(args.output_ref, prefix="var/runs/")
    spec = _load_json(_resolve(root, spec_ref))
    claimed_hash = str(spec["spec_hash"])
    if (
        _hash_bytes(_canonical_bytes({k: v for k, v in spec.items() if k != "spec_hash"}))
        != claimed_hash
    ):
        raise ValueError("frozen spec hash mismatch")
    role_key = str(args.role).casefold()
    if role_key not in {"baseline", "candidate"}:
        raise ValueError("unsupported training role")
    run = spec[role_key]
    if (
        not isinstance(run, dict)
        or run.get("run_id") != args.run_id
        or run.get("role") != args.role
    ):
        raise ValueError("run identity does not match frozen spec")
    expected_output = f"var/{run['command']['output_ref']}"
    if output_ref != expected_output or run["command"]["argv"][2:] != sys.argv[1:]:
        raise ValueError("runtime argv does not match frozen structured command")
    dataset, split, config = _validate_fixture(root, run)

    output = _resolve(root, output_ref)
    if output.exists():
        raise ValueError("run output directory already exists")
    output.mkdir(parents=True)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    step_losses, epoch_losses, predictions, accuracy = _train(
        run=run,
        dataset=dataset,
        split=split,
        config=config,
        started_monotonic=started_monotonic,
    )
    finished_at = _utc_now()
    local_prefix = f"runs/{args.run_id}"
    metrics = {
        "budget": run["budget"],
        "capability": CAPABILITY,
        "epoch_losses": epoch_losses,
        "final_loss": step_losses[-1]["loss"],
        "initial_loss": step_losses[0]["loss"],
        "prediction_count": len(predictions),
        "real_pytorch_training": False,
        "role": args.role,
        "run_id": args.run_id,
        "schema_version": "1",
        "seed": run["seed"],
        "spec_hash": claimed_hash,
        "status": "SUCCEEDED",
        "step_losses": step_losses,
        "test_accuracy": accuracy,
    }
    prediction_artifact = {
        "capability": CAPABILITY,
        "items": predictions,
        "real_pytorch_training": False,
        "role": args.role,
        "run_id": args.run_id,
        "schema_version": "1",
        "spec_hash": claimed_hash,
        "split_hash": run["split_hash"],
        "split_ref": run["split_ref"],
    }
    manifest = {
        **{
            key: value
            for key, value in run.items()
            if key not in {"capability", "fixture_labeled", "synthetic_data_labeled"}
        },
        "capability": CAPABILITY,
        "dependency_install_used": False,
        "finished_at": finished_at,
        "fixture_labeled": True,
        "log_ref": f"{local_prefix}/train.log",
        "manifest_ref": f"{local_prefix}/manifest.json",
        "metrics_ref": f"{local_prefix}/metrics.json",
        "network_used": False,
        "predictions_ref": f"{local_prefix}/predictions.json",
        "public_or_synthetic_data_only": True,
        "real_pytorch_training": False,
        "schema_version": "1",
        "shell_used": False,
        "spec_hash": claimed_hash,
        "spec_ref": run["command"]["spec_ref"],
        "started_at": started_at,
        "status": "SUCCEEDED",
        "synthetic_data_labeled": True,
        "unknown_checkpoint_loaded": False,
    }
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "predictions.json", prediction_artifact)
    _write_json(output / "manifest.json", manifest)
    log_lines = [
        {"event": "run_started", "role": args.role, "run_id": args.run_id},
        *[
            {
                "epoch": item["epoch"],
                "event": "epoch_completed",
                "mean_loss": item["mean_loss"],
                "steps": item["steps"],
            }
            for item in epoch_losses
        ],
        {
            "capability": CAPABILITY,
            "event": "run_completed",
            "real_pytorch_training": False,
        },
    ]
    (output / "train.log").write_text(
        "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in log_lines),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-ref", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("BASELINE", "CANDIDATE"), required=True)
    parser.add_argument("--output-ref", required=True)
    try:
        _execute(parser.parse_args())
    except Exception as error:
        print(f"TRAINING_FIXTURE_FAILED:{type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
