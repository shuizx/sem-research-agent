"""StructuredFailure and structured execution-entrypoint regressions."""

from __future__ import annotations

import pydantic
import pytest

from vision_research_ops.domain import StructuredFailure

pytestmark = pytest.mark.unit


def test_structured_failure_happy_path(make_structured_failure) -> None:
    failure = make_structured_failure()
    assert failure.code == "RUN_RESOURCE_POLICY_EXCEEDED"
    assert failure.retryable is False
    assert failure.message_hash.startswith("sha256:")


def test_structured_failure_json_roundtrip(make_structured_failure) -> None:
    failure = make_structured_failure(details={"expected_revision": 2})
    rebuilt = StructuredFailure.model_validate_json(failure.model_dump_json())
    assert rebuilt == failure


def test_structured_failure_requires_message_hash(make_structured_failure) -> None:
    payload = make_structured_failure().model_dump()
    del payload["message_hash"]
    with pytest.raises(pydantic.ValidationError):
        StructuredFailure(**payload)


@pytest.mark.parametrize("category", ["", "run", "1RUN", "RUN-ERROR", "A" * 65])
def test_structured_failure_category_is_extensible_but_patterned(
    make_structured_failure, category: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(category=category)


def test_structured_failure_message_is_bounded_and_desensitized(make_structured_failure) -> None:
    assert make_structured_failure(message="x" * 1024)
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(message="")
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(message="x" * 1025)


def test_structured_failure_rejects_cause_ref_and_unknown_fields(make_structured_failure) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(cause_ref="internal-cause")
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(extra="boom")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_structured_failure_details_reject_non_finite_json(
    make_structured_failure, bad: float
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(details={"nested": [bad]})


def test_structured_failure_details_enforce_size_and_depth(make_structured_failure) -> None:
    nested: object = 0
    for _ in range(9):
        nested = {"nested": nested}
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(details={"root": nested})
    with pytest.raises(pydantic.ValidationError):
        make_structured_failure(details={"payload": "x" * (16 * 1024)})


def test_command_spec_uses_structured_fields(make_command_spec) -> None:
    spec = make_command_spec(env_refs={"HF_TOKEN": "secret-ref-1"})
    assert spec.executable_id == "python"
    assert spec.argv == ["train.py", "--epochs", "1"]
    assert spec.cwd_ref == "scratch/worktree_1"
    assert spec.env_refs == {"HF_TOKEN": "secret-ref-1"}


@pytest.mark.parametrize("bad", ["", 1, True])
def test_command_spec_rejects_non_strict_executable(make_command_spec, bad: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_command_spec(executable_id=bad)


@pytest.mark.parametrize(
    ("executable_id", "argv"),
    [
        ("sh", ["-c", "python train.py"]),
        ("bash", ["-c", "python train.py"]),
        ("dash", ["-c", "python train.py"]),
        ("zsh", ["-c", "python train.py"]),
        ("ksh", ["-c", "python train.py"]),
        ("csh", ["-c", "python train.py"]),
        ("tcsh", ["-c", "python train.py"]),
        ("fish", ["-c", "python train.py"]),
        ("cmd", ["/c", "python train.py"]),
        ("cmd.exe", ["/c", "python train.py"]),
        ("powershell", ["-Command", "python train.py"]),
        ("powershell.exe", ["-Command", "python train.py"]),
        ("pwsh", ["-EncodedCommand", "cAB5AHQAaABvAG4A"]),
        ("pwsh.exe", ["-EncodedCommand", "cAB5AHQAaABvAG4A"]),
        ("wsl", ["python", "train.py"]),
        ("env", ["python", "train.py"]),
        ("xargs", ["python", "train.py"]),
        ("sudo", ["python", "train.py"]),
        ("su", ["-c", "python train.py"]),
        ("doas", ["python", "train.py"]),
        ("busybox", ["sh", "-c", "python train.py"]),
        ("/bin/bash", ["-c", "python train.py"]),
        (r"C:\Windows\System32\CMD.EXE", ["/c", "python train.py"]),
        ("PoWeRsHeLl.ExE", ["-Command", "python train.py"]),
    ],
)
def test_structured_command_models_reject_shell_and_wrapper_executables(
    make_command_spec,
    make_run_entrypoint,
    executable_id: str,
    argv: list[str],
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_command_spec(executable_id=executable_id, argv=argv)
    with pytest.raises(pydantic.ValidationError):
        make_run_entrypoint(executable_id=executable_id, argv=argv)


def test_structured_commands_allow_python_torchrun_and_literal_metacharacters(
    make_command_spec, make_run_entrypoint
) -> None:
    literal_argv = ["train.py", "--label", "a;b && $(literal)"]
    assert make_run_entrypoint(executable_id="python", argv=literal_argv).argv == literal_argv
    assert make_command_spec(executable_id="torchrun", argv=literal_argv).argv == literal_argv


def test_run_entrypoint_is_structured_and_repository_relative(make_run_entrypoint) -> None:
    entrypoint = make_run_entrypoint(env_refs={"DATA_TOKEN": "secret-ref-2"})
    assert entrypoint.cwd_subpath == "src"
    assert entrypoint.argv == ["train.py", "--epochs", "1"]
    assert entrypoint.env_refs == {"DATA_TOKEN": "secret-ref-2"}


@pytest.mark.parametrize(
    "bad_path",
    [
        "%2e%2e/outside",
        "%252e%252e/outside",
        "src%2f..%2foutside",
        "src%5c..%5coutside",
        "../outside",
        "src/../outside",
        "./src",
        "src/.",
        "src//train",
        "src/",
        "/repo",
        "//server/share",
        "C:/repo",
        r"C:\repo",
        r"\\server\share",
        "file:/tmp",
        "",
        " ",
        "\t",
        " src",
        "src ",
        "src/ train",
        "src/train ",
        "src\x00train",
        "src\ntrain",
        "NUL",
        "src/CON.txt",
    ],
)
def test_run_entrypoint_rejects_unsafe_cwd_subpaths(make_run_entrypoint, bad_path: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_run_entrypoint(cwd_subpath=bad_path)


@pytest.mark.parametrize("cwd_subpath", [".", "src", "src/train", ".config", "src/.config"])
def test_run_entrypoint_accepts_canonical_repository_relative_paths(
    make_run_entrypoint, cwd_subpath: str
) -> None:
    assert make_run_entrypoint(cwd_subpath=cwd_subpath).cwd_subpath == cwd_subpath


def test_run_entrypoint_rejects_shell_command_field(make_run_entrypoint) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_run_entrypoint(command="python train.py")
