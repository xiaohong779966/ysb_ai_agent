"""Tests for the local Sentence Transformers embedding provider."""

from collections.abc import Sequence
from typing import Any

import pytest

from backend.app.config.settings import Settings
from backend.app.rag.embeddings.base import EmbeddingProvider
from backend.app.rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDependencyError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
)
from backend.app.rag.embeddings.sentence_transformers_provider import (
    SentenceTransformersProvider,
)
from backend.app.rag.embeddings.service import EmbeddingService


class ArrayResult:
    """Minimal NumPy-like return value used without importing NumPy."""

    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class FakeSentenceTransformer:
    """Records model calls while returning deterministic vectors."""

    def __init__(self, dimensions: int | None = 3) -> None:
        self._dimensions = dimensions
        self.document_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.query_calls: list[tuple[str, dict[str, Any]]] = []
        self.document_result: object = ArrayResult([[1.0, 2.0, 3.0]])
        self.query_result: object = ArrayResult([4.0, 5.0, 6.0])
        self.document_error: Exception | None = None
        self.query_error: Exception | None = None

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._dimensions

    def encode_document(self, texts: Sequence[str], **kwargs: object) -> object:
        self.document_calls.append((tuple(texts), dict(kwargs)))
        if self.document_error is not None:
            raise self.document_error
        return self.document_result

    def encode_query(self, text: str, **kwargs: object) -> object:
        self.query_calls.append((text, dict(kwargs)))
        if self.query_error is not None:
            raise self.query_error
        return self.query_result


class RecordingLoader:
    """Injectable model loader that proves lazy and single-load behavior."""

    def __init__(self, model: FakeSentenceTransformer) -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model_name: str, device: str) -> FakeSentenceTransformer:
        self.calls.append((model_name, device))
        return self.model


def make_provider(
    *,
    model: FakeSentenceTransformer | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    normalize: bool = True,
    cuda_available: bool = False,
) -> tuple[SentenceTransformersProvider, FakeSentenceTransformer, RecordingLoader]:
    fake_model = model or FakeSentenceTransformer()
    loader = RecordingLoader(fake_model)
    provider = SentenceTransformersProvider(
        model_name="test/erp-embedding",
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        model_loader=loader,
        cuda_available=lambda: cuda_available,
    )
    return provider, fake_model, loader


def test_provider_construction_is_lazy_and_implements_contract() -> None:
    provider, _, loader = make_provider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.provider_name == "sentence_transformers"
    assert provider.model_name == "test/erp-embedding"
    assert loader.calls == []


def test_dimensions_loads_model_once_and_is_cached() -> None:
    provider, _, loader = make_provider()

    assert provider.dimensions == 3
    assert provider.dimensions == 3
    assert loader.calls == [("test/erp-embedding", "cpu")]


def test_from_settings_maps_embedding_configuration_without_loading_model() -> None:
    settings = Settings(
        embedding_model="local/model",
        embedding_device="cpu",
        embedding_batch_size=16,
        embedding_normalize=False,
    )

    provider = SentenceTransformersProvider.from_settings(settings)

    assert provider.model_name == "local/model"
    assert provider.configured_device == "cpu"
    assert provider.batch_size == 16
    assert provider.normalize is False


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(False, "cpu"), (True, "cuda")],
)
def test_auto_device_resolves_from_cuda_availability(
    cuda_available: bool,
    expected_device: str,
) -> None:
    provider, _, loader = make_provider(
        device="auto",
        cuda_available=cuda_available,
    )

    assert provider.resolved_device == expected_device
    assert provider.dimensions == 3
    assert loader.calls == [("test/erp-embedding", expected_device)]


def test_explicit_cpu_does_not_probe_cuda() -> None:
    probe_calls = 0

    def probe_cuda() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return True

    model = FakeSentenceTransformer()
    loader = RecordingLoader(model)
    provider = SentenceTransformersProvider(
        model_name="test/model",
        device="cpu",
        model_loader=loader,
        cuda_available=probe_cuda,
    )

    assert provider.resolved_device == "cpu"
    assert probe_calls == 0


def test_explicit_cuda_fails_clearly_when_cuda_is_unavailable() -> None:
    provider, _, loader = make_provider(device="cuda", cuda_available=False)

    with pytest.raises(EmbeddingConfigurationError, match="CUDA.*not available"):
        _ = provider.dimensions

    assert loader.calls == []


def test_document_encoding_uses_batch_normalization_and_numpy_conversion_options() -> None:
    provider, model, _ = make_provider(batch_size=12, normalize=False)
    model.document_result = ArrayResult([[1, 2, 3], [4, 5, 6]])

    result = provider.embed_documents(("问题一", "问题二"))

    assert result == [[1, 2, 3], [4, 5, 6]]
    assert model.document_calls == [
        (
            ("问题一", "问题二"),
            {
                "batch_size": 12,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "device": "cpu",
            },
        )
    ]


def test_query_encoding_uses_query_specific_method() -> None:
    provider, model, _ = make_provider(normalize=True)

    result = provider.embed_query("如何新增商品？")

    assert result == [4.0, 5.0, 6.0]
    assert model.query_calls == [
        (
            "如何新增商品？",
            {
                "batch_size": 8,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "device": "cpu",
            },
        )
    ]


def test_empty_document_batch_does_not_load_model() -> None:
    provider, _, loader = make_provider(device="auto")

    assert provider.embed_documents(()) == ()
    assert loader.calls == []


def test_provider_integrates_with_validation_service() -> None:
    provider, model, _ = make_provider()
    model.document_result = ArrayResult([[1.0, 0.0, 0.0]])
    service = EmbeddingService(provider)

    vectors = service.embed_documents(["  ERP 库存问题  "])

    assert vectors == ((1.0, 0.0, 0.0),)
    assert model.document_calls[0][0] == ("ERP 库存问题",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": "  "}, "model name"),
        ({"device": "mps"}, "device"),
        ({"batch_size": 0}, "batch size"),
        ({"batch_size": True}, "batch size"),
        ({"normalize": "yes"}, "normalize"),
    ],
)
def test_invalid_provider_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    defaults: dict[str, object] = {
        "model_name": "test/model",
        "device": "cpu",
        "batch_size": 32,
        "normalize": True,
        "model_loader": RecordingLoader(FakeSentenceTransformer()),
        "cuda_available": lambda: False,
    }
    defaults.update(kwargs)

    with pytest.raises(EmbeddingConfigurationError, match=message):
        SentenceTransformersProvider(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize("dimensions", [None, 0, True, "3"])
def test_invalid_model_dimension_is_wrapped_as_model_load_error(
    dimensions: object,
) -> None:
    model = FakeSentenceTransformer()
    model._dimensions = dimensions  # type: ignore[assignment]
    provider, _, _ = make_provider(model=model)

    with pytest.raises(EmbeddingModelLoadError, match="embedding dimension"):
        _ = provider.dimensions


def test_model_loader_failure_is_wrapped_without_exposing_library_exception() -> None:
    def failing_loader(model_name: str, device: str) -> FakeSentenceTransformer:
        raise OSError(f"cannot load {model_name} on {device}")

    provider = SentenceTransformersProvider(
        model_name="test/model",
        device="cpu",
        model_loader=failing_loader,
    )

    with pytest.raises(EmbeddingModelLoadError, match="Failed to load") as error:
        _ = provider.dimensions

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize("method", ["documents", "query"])
def test_encoding_failure_is_wrapped_with_operation_context(method: str) -> None:
    provider, model, _ = make_provider()
    inference_error = RuntimeError("backend inference failure")

    if method == "documents":
        model.document_error = inference_error

        def run_operation() -> object:
            return provider.embed_documents(["问题"])

    else:
        model.query_error = inference_error

        def run_operation() -> object:
            return provider.embed_query("问题")

    with pytest.raises(EmbeddingInferenceError, match=method) as error:
        run_operation()

    assert error.value.__cause__ is inference_error


def test_cuda_probe_dependency_failure_is_preserved() -> None:
    def missing_torch() -> bool:
        raise EmbeddingDependencyError("PyTorch is unavailable")

    provider = SentenceTransformersProvider(
        model_name="test/model",
        device="auto",
        model_loader=RecordingLoader(FakeSentenceTransformer()),
        cuda_available=missing_torch,
    )

    with pytest.raises(EmbeddingDependencyError, match="PyTorch"):
        _ = provider.resolved_device
