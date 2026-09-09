"""Local Sentence Transformers implementation of the embedding provider contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import RLock
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

from backend.app.rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDependencyError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
)

if TYPE_CHECKING:
    from backend.app.config.settings import Settings

EmbeddingDeviceOption = Literal["auto", "cpu", "cuda"]


class SentenceTransformerModel(Protocol):
    """Narrow model surface used by the adapter and its unit tests."""

    def get_sentence_embedding_dimension(self) -> int | None:
        """Return the output vector size reported by the loaded model."""
        ...

    def encode_document(self, texts: Sequence[str], **kwargs: object) -> object:
        """Encode document text using model-specific document behavior."""
        ...

    def encode_query(self, text: str, **kwargs: object) -> object:
        """Encode query text using model-specific query behavior."""
        ...


ModelLoader = Callable[[str, str], SentenceTransformerModel]
CudaAvailabilityProbe = Callable[[], bool]


class SentenceTransformersProvider:
    """Generate local embeddings with one lazily loaded Sentence Transformer model."""

    _PROVIDER_NAME = "sentence_transformers"
    _SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda"})

    def __init__(
        self,
        *,
        model_name: str,
        device: EmbeddingDeviceOption = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        model_loader: ModelLoader | None = None,
        cuda_available: CudaAvailabilityProbe | None = None,
    ) -> None:
        self._model_name = self._validate_model_name(model_name)
        self._configured_device = self._validate_device(device)
        self._batch_size = self._validate_batch_size(batch_size)
        self._normalize = self._validate_normalize(normalize)
        self._model_loader = model_loader or _default_model_loader
        self._cuda_available = cuda_available or _default_cuda_available
        self._model: SentenceTransformerModel | None = None
        self._dimensions: int | None = None
        self._resolved_device: str | None = None
        self._lock = RLock()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a provider from validated application embedding settings."""
        return cls(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
        )

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used by application settings."""
        return self._PROVIDER_NAME

    @property
    def model_name(self) -> str:
        """Return the configured local model identifier or filesystem path."""
        return self._model_name

    @property
    def configured_device(self) -> EmbeddingDeviceOption:
        """Return the configured device before automatic device selection."""
        return self._configured_device

    @property
    def resolved_device(self) -> str:
        """Resolve auto to CUDA when available and otherwise use CPU."""
        with self._lock:
            if self._resolved_device is None:
                self._resolved_device = self._resolve_device()
            return self._resolved_device

    @property
    def batch_size(self) -> int:
        """Return the maximum batch size passed to model encoding."""
        return self._batch_size

    @property
    def normalize(self) -> bool:
        """Return whether the model should normalize generated vectors."""
        return self._normalize

    @property
    def dimensions(self) -> int:
        """Load the model when first needed and cache its vector dimensions."""
        with self._lock:
            if self._dimensions is None:
                model = self._get_model()
                try:
                    dimensions = model.get_sentence_embedding_dimension()
                except Exception as exc:
                    raise EmbeddingModelLoadError(
                        f"Failed to inspect embedding dimension for model '{self.model_name}'"
                    ) from exc
                if (
                    isinstance(dimensions, bool)
                    or not isinstance(dimensions, int)
                    or dimensions <= 0
                ):
                    raise EmbeddingModelLoadError(
                        f"Model '{self.model_name}' returned an invalid embedding dimension"
                    )
                self._dimensions = dimensions
            return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode documents in order without loading the model for an empty batch."""
        if not texts:
            return ()
        try:
            result = self._get_model().encode_document(
                list(texts),
                **self._encoding_options(),
            )
            return cast("Sequence[Sequence[float]]", _convert_array_result(result))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Failed to encode documents with model '{self.model_name}'"
            ) from exc

    def embed_query(self, text: str) -> Sequence[float]:
        """Encode one query with the model's query-specific encoding method."""
        try:
            result = self._get_model().encode_query(
                text,
                **self._encoding_options(),
            )
            return cast("Sequence[float]", _convert_array_result(result))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Failed to encode query with model '{self.model_name}'"
            ) from exc

    def _get_model(self) -> SentenceTransformerModel:
        with self._lock:
            if self._model is None:
                device = self.resolved_device
                try:
                    self._model = self._model_loader(self.model_name, device)
                except EmbeddingError:
                    raise
                except Exception as exc:
                    raise EmbeddingModelLoadError(
                        f"Failed to load embedding model '{self.model_name}' on {device}"
                    ) from exc
            return self._model

    def _resolve_device(self) -> str:
        if self.configured_device == "cpu":
            return "cpu"

        cuda_available = self._cuda_available()
        if self.configured_device == "cuda":
            if not cuda_available:
                raise EmbeddingConfigurationError(
                    "CUDA was requested for embeddings but is not available"
                )
            return "cuda"
        return "cuda" if cuda_available else "cpu"

    def _encoding_options(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize,
            "device": self.resolved_device,
        }

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        if not isinstance(model_name, str) or not model_name.strip():
            raise EmbeddingConfigurationError("Embedding model name must not be blank")
        return model_name.strip()

    @classmethod
    def _validate_device(cls, device: str) -> EmbeddingDeviceOption:
        if not isinstance(device, str) or device.lower() not in cls._SUPPORTED_DEVICES:
            raise EmbeddingConfigurationError(
                "Embedding device must be one of: auto, cpu, cuda"
            )
        return cast("EmbeddingDeviceOption", device.lower())

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 512
        ):
            raise EmbeddingConfigurationError(
                "Embedding batch size must be an integer between 1 and 512"
            )
        return batch_size

    @staticmethod
    def _validate_normalize(normalize: bool) -> bool:
        if not isinstance(normalize, bool):
            raise EmbeddingConfigurationError("Embedding normalize must be a boolean")
        return normalize


def _default_cuda_available() -> bool:
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "PyTorch is required to resolve the embedding device"
        ) from exc

    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        raise EmbeddingDependencyError("Unable to inspect PyTorch CUDA availability") from exc


def _default_model_loader(model_name: str, device: str) -> SentenceTransformerModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "sentence-transformers is required to load the configured embedding model"
        ) from exc

    return cast(
        "SentenceTransformerModel",
        SentenceTransformer(model_name, device=device),
    )


def _convert_array_result(result: object) -> object:
    to_list = getattr(result, "tolist", None)
    return to_list() if callable(to_list) else result
