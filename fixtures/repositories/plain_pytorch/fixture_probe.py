"""Fixed no-Torch contract probe for the controlled adaptation fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

CAPABILITY = "FIXTURE_CONTRACT_PROBE_NO_TORCH"
METRICS = ["macro_f1", "balanced_accuracy", "per_class_recall"]
STAGES = {"STATIC_POLICY", "IMPORT", "ONE_BATCH", "BOUNDED_OVERFIT"}


def _deny_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("FIXTURE_NETWORK_DISABLED")


def _install_network_guard() -> None:
    socket.socket = _deny_network  # type: ignore[assignment]
    socket.create_connection = _deny_network  # type: ignore[assignment]


def _load_config() -> dict[str, Any]:
    value = json.loads(Path("sem_adaptation.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("FIXTURE_CONFIG_NOT_OBJECT")
    return value


def _validate_output_path(value: object) -> str:
    if not isinstance(value, str) or value != value.strip() or "\\" in value:
        raise ValueError("METRICS_OUTPUT_PATH_INVALID")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", "..", ".git", ".env"} for part in path.parts):
        raise ValueError("METRICS_OUTPUT_PATH_INVALID")
    if path.suffix.casefold() != ".json":
        raise ValueError("METRICS_OUTPUT_PATH_INVALID")
    return value


def _validate_contract(config: dict[str, Any], minimum_repair_revision: int) -> dict[str, Any]:
    if config.get("schema_version") != "1":
        raise ValueError("FIXTURE_SCHEMA_VERSION_INVALID")
    if config.get("fixture_contract") != "SEM_PLAIN_PYTORCH_CONFIG_V1":
        raise ValueError("FIXTURE_TEMPLATE_INVALID")
    channels = config.get("input", {}).get("channels")
    if channels != 1 or config.get("input", {}).get("modality") != "GRAYSCALE":
        raise ValueError("GRAYSCALE_CHANNEL_CONTRACT_INVALID")
    mapping = config.get("data", {}).get("label_mapping")
    num_classes = config.get("model", {}).get("num_classes")
    if not isinstance(mapping, dict) or len(mapping) != num_classes:
        raise ValueError("LABEL_CLASS_CONTRACT_INVALID")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("LABEL_MAPPING_NOT_DENSE")
    if config.get("data", {}).get("group_split_strategy") != "GROUP_HOLDOUT":
        raise ValueError("GROUP_SPLIT_CONTRACT_INVALID")
    group_key = config.get("data", {}).get("group_split_key")
    if not isinstance(group_key, str) or not group_key:
        raise ValueError("GROUP_SPLIT_KEY_INVALID")
    if config.get("metrics", {}).get("names") != METRICS:
        raise ValueError("METRICS_CONTRACT_INVALID")
    output_file = _validate_output_path(config.get("metrics", {}).get("output_file"))
    repair_revision = config.get("repair_revision")
    if not isinstance(repair_revision, int) or repair_revision < minimum_repair_revision:
        raise ValueError("FIXTURE_REPAIR_REVISION_REQUIRED")
    return {
        "channels": channels,
        "num_classes": num_classes,
        "label_count": len(mapping),
        "group_split_key": group_key,
        "metrics": METRICS,
        "metrics_output_file": output_file,
        "repair_revision": repair_revision,
    }


def _import_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sem_fixture_adapter", "sem_adapter.py")
    if spec is None or spec.loader is None:
        raise ValueError("FIXTURE_ADAPTER_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _static_stage(config: dict[str, Any], minimum: int) -> dict[str, Any]:
    required = {
        "config.yaml",
        "data.py",
        "model.py",
        "train.py",
        "sem_adapter.py",
        "sem_adaptation.json",
    }
    if not all(Path(name).is_file() and not Path(name).is_symlink() for name in required):
        raise ValueError("FIXTURE_LAYOUT_INVALID")
    evidence = _validate_contract(config, minimum)
    evidence["dependency_changes"] = 0
    evidence["shell_commands"] = 0
    evidence["network_guard"] = "STDLIB_SOCKET_BLOCKED"
    return evidence


def _import_stage(config: dict[str, Any], minimum: int) -> dict[str, Any]:
    _validate_contract(config, minimum)
    adapter = _import_adapter()
    loaded = adapter.load_contract()
    if loaded != config:
        raise ValueError("FIXTURE_ADAPTER_CONFIG_MISMATCH")
    return {"imported_module": "sem_adapter", "torch_imported": False}


def _one_batch_stage(config: dict[str, Any], minimum: int) -> dict[str, Any]:
    contract = _validate_contract(config, minimum)
    adapter = _import_adapter()
    labels = list(config["data"]["label_mapping"])
    result = adapter.build_one_batch(
        [[0.05], [0.25], [0.75], [0.95]],
        [labels[0], labels[1], labels[-1], labels[0]],
        ["group-a", "group-a", "group-b", "group-b"],
        config,
    )
    result["num_classes"] = contract["num_classes"]
    result["synthetic_batch"] = True
    return result


def _bounded_overfit_stage(config: dict[str, Any], minimum: int) -> dict[str, Any]:
    contract = _validate_contract(config, minimum)
    features = [0.0, 0.3, 0.7, 1.0]
    targets = [0.0, 0.3, 0.7, 1.0]
    weight = 0.0

    def loss(current: float) -> float:
        return sum(
            (current * x - target) ** 2 for x, target in zip(features, targets, strict=True)
        ) / len(features)

    initial_loss = loss(weight)
    for _ in range(12):
        gradient = sum(
            2.0 * (weight * x - target) * x for x, target in zip(features, targets, strict=True)
        ) / len(features)
        weight -= 0.4 * gradient
    final_loss = loss(weight)
    if not final_loss < initial_loss:
        raise ValueError("FIXTURE_OPTIMIZATION_PROBE_DID_NOT_IMPROVE")

    output = Path(*PurePosixPath(contract["metrics_output_file"]).parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fixture_labeled": True,
                "synthetic_data_labeled": True,
                "capability_boundary": CAPABILITY,
                "real_pytorch_training": False,
                "metrics": {
                    name: {"status": "NOT_COMPUTED_FIXTURE_CONTRACT_ONLY"} for name in METRICS
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "optimization_steps": 12,
        "metrics_output": contract["metrics_output_file"],
        "real_pytorch_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--minimum-repair-revision", type=int, required=True)
    args = parser.parse_args()
    _install_network_guard()
    try:
        config = _load_config()
        handlers = {
            "STATIC_POLICY": _static_stage,
            "IMPORT": _import_stage,
            "ONE_BATCH": _one_batch_stage,
            "BOUNDED_OVERFIT": _bounded_overfit_stage,
        }
        evidence = handlers[args.stage](config, args.minimum_repair_revision)
        payload = {
            "schema_version": "1",
            "stage": args.stage,
            "passed": True,
            "capability_boundary": CAPABILITY,
            "evidence": evidence,
            "reason_code": None,
        }
        exit_code = 0
    except (
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        reason = str(error)
        if not reason or len(reason) > 128:
            reason = "FIXTURE_PROBE_FAILED"
        payload = {
            "schema_version": "1",
            "stage": args.stage,
            "passed": False,
            "capability_boundary": CAPABILITY,
            "evidence": {"failure_observed": True},
            "reason_code": reason,
        }
        exit_code = 2
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
