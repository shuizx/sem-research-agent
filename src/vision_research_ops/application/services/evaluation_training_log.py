"""Pure validation for the bounded training synthetic training JSONL contract."""

from __future__ import annotations

import json
from math import isclose, isfinite

from .training_models import (
    TRAINING_CAPABILITY,
    FrozenRunSpec,
    TrainingMetrics,
)

_MAX_TRAINING_LOG_BYTES = 65_536
_LOG_EVENT_VOCABULARY = frozenset({"run_started", "epoch_completed", "run_completed"})


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def validate_training_log(
    payload: bytes,
    *,
    run: FrozenRunSpec,
    metrics: TrainingMetrics,
) -> None:
    """Validate log integrity and identity without sourcing evaluation metrics."""
    if not payload or len(payload) > _MAX_TRAINING_LOG_BYTES:
        raise ValueError("training log is empty or exceeds the fixture bound")
    log_text = payload.decode("utf-8")
    lines = [line for line in log_text.splitlines() if line.strip()]
    if len(lines) < 3 or len(lines) > run.budget.max_epochs + 2:
        raise ValueError("training log does not have a bounded event count")

    events: list[dict[str, object]] = []
    for line in lines:
        parsed = json.loads(line, parse_constant=_reject_nonfinite_json)
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            raise ValueError("each training log line must be one JSON object")
        if parsed.get("event") not in _LOG_EVENT_VOCABULARY:
            raise ValueError("training log event is outside the fixture vocabulary")
        events.append(parsed)

    started = events[0]
    if (
        set(started) != {"event", "role", "run_id"}
        or started["event"] != "run_started"
        or started["role"] != run.role
        or started["run_id"] != run.run_id
    ):
        raise ValueError("training log must start with the exact run identity")

    completed = events[-1]
    if (
        set(completed) != {"capability", "event", "real_pytorch_training"}
        or completed["event"] != "run_completed"
        or completed["capability"] != TRAINING_CAPABILITY
        or completed["real_pytorch_training"] is not False
    ):
        raise ValueError("training log must end with the exact completion event")

    epoch_events = events[1:-1]
    if len(epoch_events) != len(metrics.epoch_losses):
        raise ValueError("training log epoch count conflicts with metrics")
    for event, epoch_loss in zip(epoch_events, metrics.epoch_losses, strict=True):
        epoch = event.get("epoch")
        steps = event.get("steps")
        mean_loss = event.get("mean_loss")
        if (
            set(event) != {"epoch", "event", "mean_loss", "steps"}
            or event.get("event") != "epoch_completed"
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch != epoch_loss.epoch
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps != epoch_loss.steps
            or isinstance(mean_loss, bool)
            or not isinstance(mean_loss, int | float)
            or not isfinite(mean_loss)
            or mean_loss < 0.0
            or not isclose(
                mean_loss,
                epoch_loss.mean_loss,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("training log epoch event conflicts with structured metrics")


__all__ = ["validate_training_log"]
