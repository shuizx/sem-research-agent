"""Value-object tests: ArtifactRef, ProvenanceRef, ResourceRequest,
MetricDefinition."""

from __future__ import annotations

import pydantic
import pytest

from vision_research_ops.domain import (
    ArtifactKind,
    ArtifactRef,
    NetworkAllowlistRef,
    NetworkPolicy,
)

pytestmark = pytest.mark.unit

SHA256 = "sha256:" + "a" * 64


def test_artifact_ref_happy_path(make_artifact_ref) -> None:
    artifact = make_artifact_ref()
    assert artifact.kind is ArtifactKind.PATCH
    assert artifact.size_bytes == 128
    assert artifact.sha256 == SHA256


def test_artifact_ref_size_bytes_must_be_non_negative(make_artifact_ref) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_artifact_ref(size_bytes=-1)


def test_artifact_ref_metadata_default_is_isolated(make_artifact_ref) -> None:
    first = make_artifact_ref()
    second = make_artifact_ref()
    first.metadata["x"] = 1
    assert "x" not in second.metadata
    assert first.metadata == {"x": 1}
    assert second.metadata == {}


def test_artifact_ref_sensitivity_literal(make_artifact_ref) -> None:
    for value in ["PUBLIC", "INTERNAL", "RESTRICTED"]:
        assert make_artifact_ref(sensitivity=value).sensitivity == value
    with pytest.raises(pydantic.ValidationError):
        make_artifact_ref(sensitivity="TOP_SECRET")


def test_provenance_ref_happy_path(make_provenance_ref) -> None:
    prov = make_provenance_ref()
    assert prov.source_type == "provider"
    assert prov.source_url is not None


def test_provenance_ref_source_type_literal(make_provenance_ref) -> None:
    for value in ["provider", "api", "user", "generated"]:
        assert make_provenance_ref(source_type=value).source_type == value
    with pytest.raises(pydantic.ValidationError):
        make_provenance_ref(source_type="PROVIDER")


def test_provenance_ref_content_hash_constraint(make_provenance_ref) -> None:
    prov = make_provenance_ref(content_hash="sha256:" + "b" * 64)
    assert prov.content_hash == "sha256:" + "b" * 64
    with pytest.raises(pydantic.ValidationError):
        make_provenance_ref(content_hash="not-a-hash")


def test_resource_request_happy_path(make_resource_request) -> None:
    res = make_resource_request()
    assert res.cpu_cores == 2.0
    assert res.gpu_count == 0


def test_resource_request_cpu_cores_must_be_positive(make_resource_request) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(cpu_cores=0.0)
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(cpu_cores=-1.0)
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(cpu_cores=float("nan"))


def test_resource_request_memory_positive(make_resource_request) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(memory_mb=0)
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(memory_mb=-100)


def test_resource_request_gpu_count_non_negative(make_resource_request) -> None:
    assert make_resource_request(gpu_count=0).gpu_count == 0
    assert make_resource_request(gpu_count=4).gpu_count == 4
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(gpu_count=-1)


def test_resource_request_walltime_positive(make_resource_request) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(walltime_seconds=0)
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(walltime_seconds=-60)


def test_resource_request_scratch_non_negative(make_resource_request) -> None:
    assert make_resource_request(scratch_mb=0).scratch_mb == 0
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(scratch_mb=-1)


def test_resource_request_network_policy(make_resource_request) -> None:
    assert (
        make_resource_request(network_policy=NetworkPolicy.NONE).network_policy
        is NetworkPolicy.NONE
    )
    assert (
        make_resource_request(
            network_policy=NetworkPolicy.ALLOWLIST,
            network_allowlist_refs=["netref_pypi_readonly"],
        ).network_policy
        is NetworkPolicy.ALLOWLIST
    )
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(network_policy="UNRESTRICTED")


def test_resource_request_network_policy_combinations_and_deduplication(
    make_resource_request,
) -> None:
    assert make_resource_request(network_policy=NetworkPolicy.NONE, network_allowlist_refs=[])
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(
            network_policy=NetworkPolicy.NONE,
            network_allowlist_refs=["netref_pypi_readonly"],
        )
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(network_policy=NetworkPolicy.ALLOWLIST, network_allowlist_refs=[])
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(
            network_policy=NetworkPolicy.ALLOWLIST,
            network_allowlist_refs=["netref_pypi_readonly", 1],
        )
    allowlisted = make_resource_request(
        network_policy=NetworkPolicy.ALLOWLIST,
        network_allowlist_refs=[
            "netref_pypi_readonly",
            "netref_pypi_readonly",
            "netref_model_mirror_v1",
        ],
    )
    assert allowlisted.network_allowlist_refs == [
        "netref_pypi_readonly",
        "netref_model_mirror_v1",
    ]


@pytest.mark.parametrize(
    "bad_ref",
    [
        "https://evil.example",
        "evil.example",
        "localhost",
        "127.0.0.1",
        "host:443",
        "443",
        "--proxy=http://evil.example",
        "$(curl evil.example)",
        "`whoami`",
        "netref_pypi readonly",
        "netref_pypi/readonly",
        r"netref_pypi\readonly",
        "netref_pypi:readonly",
        "netref_pypi%5freadonly",
        "NETREF_PYPI_READONLY",
        " netref_pypi_readonly",
        "netref_pypi_readonly ",
        "netref_pypi_readonly\n",
        "netref_pypi\treadonly",
        "netref_",
    ],
)
def test_resource_request_rejects_noncanonical_network_allowlist_refs(
    make_resource_request, bad_ref: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(
            network_policy=NetworkPolicy.ALLOWLIST,
            network_allowlist_refs=[bad_ref],
        )


def test_network_allowlist_ref_is_a_public_controlled_type() -> None:
    adapter = pydantic.TypeAdapter(NetworkAllowlistRef)
    assert adapter.validate_python("netref_pypi_readonly") == "netref_pypi_readonly"
    assert adapter.validate_python("netref_model_mirror_v1") == "netref_model_mirror_v1"
    assert adapter.validate_python("netref_" + "a" * 64) == "netref_" + "a" * 64
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python("netref_" + "a" * 65)
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_json('"https://evil.example"')


@pytest.mark.parametrize("bad", ["2.0", True, False])
def test_resource_request_cpu_cores_is_strict(make_resource_request, bad: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(cpu_cores=bad)


@pytest.mark.parametrize("bad", ["4096", 4096.0, True, False])
def test_resource_request_integer_fields_are_strict(make_resource_request, bad: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_resource_request(memory_mb=bad)


def test_metric_definition_happy_path(make_metric_definition) -> None:
    metric = make_metric_definition()
    assert metric.primary is True
    assert metric.direction == "MAXIMIZE"


def test_metric_definition_direction(make_metric_definition) -> None:
    assert make_metric_definition(direction="MINIMIZE").direction == "MINIMIZE"
    with pytest.raises(pydantic.ValidationError):
        make_metric_definition(direction="maximize")


def test_metric_definition_delta_finite(make_metric_definition) -> None:
    metric = make_metric_definition(minimum_practical_delta=0.01)
    assert metric.minimum_practical_delta == 0.01
    with pytest.raises(pydantic.ValidationError):
        make_metric_definition(minimum_practical_delta=float("nan"))
    with pytest.raises(pydantic.ValidationError):
        make_metric_definition(minimum_practical_delta=float("inf"))


def test_value_objects_json_roundtrip(make_artifact_ref) -> None:
    original = make_artifact_ref()
    rebuilt = ArtifactRef.model_validate_json(original.model_dump_json())
    assert rebuilt == original
