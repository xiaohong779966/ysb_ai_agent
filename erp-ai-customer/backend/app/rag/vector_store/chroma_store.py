"""Persistent Chroma implementation of the provider-neutral vector-store contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from ipaddress import AddressValueError, IPv4Address
from math import isfinite
from pathlib import Path
from re import compile as compile_pattern
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

from backend.app.rag.vector_store.base import (
    VectorMetadata,
    VectorMetadataValue,
    VectorRecord,
    VectorSearchResult,
)
from backend.app.rag.vector_store.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreDependencyError,
    VectorStoreError,
    VectorStoreInputError,
    VectorStoreOperationError,
    VectorStoreResultError,
)

if TYPE_CHECKING:
    from backend.app.config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
COLLECTION_NAME_PATTERN = compile_pattern(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])$")
ChromaDistanceMetric = Literal["cosine", "l2", "ip"]


class ChromaCollection(Protocol):
    """Narrow Chroma collection API consumed by this adapter."""

    def upsert(self, **kwargs: object) -> None:
        """Insert or replace vectors."""
        ...

    def query(self, **kwargs: object) -> Mapping[str, object]:
        """Find nearest vectors."""
        ...

    def delete(self, **kwargs: object) -> None:
        """Delete vectors by identifier."""
        ...

    def count(self) -> int:
        """Return the collection size."""
        ...


class ChromaClient(Protocol):
    """Narrow persistent client API used to open a collection."""

    def get_or_create_collection(self, **kwargs: object) -> ChromaCollection:
        """Open the configured collection or create it when missing."""
        ...


ChromaClientFactory = Callable[[Path], ChromaClient]
DirectoryPreparer = Callable[[Path], None]


class ChromaVectorStore:
    """Persist precomputed ERP knowledge vectors in one local Chroma collection."""

    _PROVIDER_NAME = "chroma"
    _SUPPORTED_DISTANCE_METRICS = frozenset({"cosine", "l2", "ip"})

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str,
        distance_metric: ChromaDistanceMetric = "cosine",
        default_query_limit: int = 5,
        client_factory: ChromaClientFactory | None = None,
        directory_preparer: DirectoryPreparer | None = None,
    ) -> None:
        self._persist_directory = self._validate_persist_directory(persist_directory)
        self._collection_name = self._validate_collection_name(collection_name)
        self._distance_metric = self._validate_distance_metric(distance_metric)
        self._default_query_limit = self._validate_query_limit(
            default_query_limit,
            error_type=VectorStoreConfigurationError,
        )
        self._client_factory = client_factory or _default_client_factory
        self._directory_preparer = directory_preparer or _prepare_directory
        self._client: ChromaClient | None = None
        self._collection: ChromaCollection | None = None
        self._lock = RLock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client_factory: ChromaClientFactory | None = None,
        directory_preparer: DirectoryPreparer | None = None,
    ) -> Self:
        """Create a Chroma store from validated application settings."""
        return cls(
            persist_directory=settings.chroma_persist_path,
            collection_name=settings.chroma_collection_name,
            distance_metric=settings.chroma_distance_metric,
            default_query_limit=settings.chroma_query_limit,
            client_factory=client_factory,
            directory_preparer=directory_preparer,
        )

    @property
    def provider_name(self) -> str:
        """Return the configured vector database provider name."""
        return self._PROVIDER_NAME

    @property
    def persist_directory(self) -> Path:
        """Return the absolute Chroma persistence directory."""
        return self._persist_directory

    @property
    def collection_name(self) -> str:
        """Return the normalized ERP vector collection name."""
        return self._collection_name

    @property
    def distance_metric(self) -> ChromaDistanceMetric:
        """Return the collection's HNSW distance metric."""
        return self._distance_metric

    @property
    def default_query_limit(self) -> int:
        """Return the result limit used when a query does not override it."""
        return self._default_query_limit

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Validate and persist precomputed vectors in one atomic Chroma call."""
        normalized_records = _normalize_records(records)
        if not normalized_records:
            return 0

        try:
            self._get_collection().upsert(
                ids=[record.id for record in normalized_records],
                documents=[record.document for record in normalized_records],
                embeddings=[list(record.embedding) for record in normalized_records],
                metadatas=[dict(record.metadata) for record in normalized_records],
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreOperationError("Chroma upsert operation failed") from exc
        return len(normalized_records)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int | None = None,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Query Chroma with a precomputed embedding and parse stable results."""
        normalized_embedding = _normalize_embedding(
            query_embedding,
            input_name="query embedding",
        )
        query_limit = self.default_query_limit
        if limit is not None:
            query_limit = self._validate_query_limit(
                limit,
                error_type=VectorStoreInputError,
            )
        normalized_where = _normalize_where_filter(where)
        query_arguments: dict[str, object] = {
            "query_embeddings": [list(normalized_embedding)],
            "n_results": query_limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if normalized_where is not None:
            query_arguments["where"] = normalized_where

        try:
            raw_result = self._get_collection().query(**query_arguments)
            return _parse_query_result(raw_result)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreOperationError("Chroma query operation failed") from exc

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records by validated unique identifiers."""
        normalized_ids = _normalize_ids(ids)
        if not normalized_ids:
            return
        try:
            self._get_collection().delete(ids=list(normalized_ids))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreOperationError("Chroma delete operation failed") from exc

    def count(self) -> int:
        """Return the number of records stored in the configured collection."""
        try:
            result = self._get_collection().count()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreOperationError("Chroma count operation failed") from exc
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise VectorStoreResultError("Chroma returned an invalid collection count")
        return result

    def _get_collection(self) -> ChromaCollection:
        with self._lock:
            if self._collection is None:
                client = self._get_client()
                try:
                    self._collection = client.get_or_create_collection(
                        name=self.collection_name,
                        configuration={"hnsw": {"space": self.distance_metric}},
                        embedding_function=None,
                    )
                except VectorStoreError:
                    raise
                except Exception as exc:
                    raise VectorStoreOperationError(
                        f"Failed to open Chroma collection '{self.collection_name}'"
                    ) from exc
            return self._collection

    def _get_client(self) -> ChromaClient:
        with self._lock:
            if self._client is None:
                try:
                    self._directory_preparer(self.persist_directory)
                    self._client = self._client_factory(self.persist_directory)
                except VectorStoreError:
                    raise
                except Exception as exc:
                    raise VectorStoreOperationError(
                        f"Failed to initialize Chroma at '{self.persist_directory}'"
                    ) from exc
            return self._client

    @staticmethod
    def _validate_persist_directory(value: str | Path) -> Path:
        if isinstance(value, str) and not value.strip():
            raise VectorStoreConfigurationError(
                "Chroma persist directory must not be blank"
            )
        try:
            path = Path(value).expanduser()
        except TypeError as exc:
            raise VectorStoreConfigurationError(
                "Chroma persist directory must be a filesystem path"
            ) from exc
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _validate_collection_name(value: str) -> str:
        if not isinstance(value, str):
            raise VectorStoreConfigurationError("Chroma collection name must be a string")
        normalized = value.strip().lower()
        if not 3 <= len(normalized) <= 512:
            raise VectorStoreConfigurationError(
                "Chroma collection name must contain 3 to 512 characters"
            )
        if ".." in normalized or not COLLECTION_NAME_PATTERN.fullmatch(normalized):
            raise VectorStoreConfigurationError(
                "Chroma collection name contains unsupported characters"
            )
        try:
            IPv4Address(normalized)
        except AddressValueError:
            return normalized
        raise VectorStoreConfigurationError(
            "Chroma collection name must not be an IPv4 address"
        )

    @classmethod
    def _validate_distance_metric(cls, value: str) -> ChromaDistanceMetric:
        if not isinstance(value, str) or value.lower() not in cls._SUPPORTED_DISTANCE_METRICS:
            raise VectorStoreConfigurationError(
                "Chroma distance metric must be one of: cosine, l2, ip"
            )
        return cast("ChromaDistanceMetric", value.lower())

    @staticmethod
    def _validate_query_limit(
        value: int,
        *,
        error_type: type[VectorStoreConfigurationError] | type[VectorStoreInputError],
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
            raise error_type("Chroma query limit must be an integer between 1 and 100")
        return value


def _prepare_directory(persist_directory: Path) -> None:
    persist_directory.mkdir(parents=True, exist_ok=True)

def _default_client_factory(persist_directory: Path) -> ChromaClient:
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreDependencyError(
            "chromadb is required to use the persistent Chroma vector store"
        ) from exc
    return cast("ChromaClient", chromadb.PersistentClient(path=str(persist_directory)))


def _normalize_records(records: Sequence[VectorRecord]) -> tuple[VectorRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise VectorStoreInputError("records must be a sequence of VectorRecord values")
    try:
        raw_records = tuple(records)
    except TypeError as exc:
        raise VectorStoreInputError("records must be iterable") from exc
    if not raw_records:
        return ()

    normalized_records: list[VectorRecord] = []
    seen_ids: set[str] = set()
    expected_dimensions: int | None = None
    for index, record in enumerate(raw_records):
        if not isinstance(record, VectorRecord):
            raise VectorStoreInputError(f"records[{index}] must be a VectorRecord")
        record_id = _normalize_id(record.id, input_name=f"records[{index}].id")
        if record_id in seen_ids:
            raise VectorStoreInputError(f"Duplicate vector record id: {record_id}")
        seen_ids.add(record_id)
        document = _normalize_document(
            record.document,
            input_name=f"records[{index}].document",
        )
        embedding = _normalize_embedding(
            record.embedding,
            input_name=f"records[{index}].embedding",
        )
        if expected_dimensions is None:
            expected_dimensions = len(embedding)
        elif len(embedding) != expected_dimensions:
            raise VectorStoreInputError(
                "All embeddings in one upsert must have the same dimensions"
            )
        metadata = _normalize_metadata(
            record.metadata,
            input_name=f"records[{index}].metadata",
        )
        normalized_records.append(
            VectorRecord(
                id=record_id,
                document=document,
                embedding=embedding,
                metadata=metadata,
            )
        )
    return tuple(normalized_records)


def _normalize_ids(ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)):
        raise VectorStoreInputError("ids must be a sequence of strings")
    try:
        raw_ids = tuple(ids)
    except TypeError as exc:
        raise VectorStoreInputError("ids must be iterable") from exc
    normalized_ids = tuple(
        _normalize_id(value, input_name=f"ids[{index}]")
        for index, value in enumerate(raw_ids)
    )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise VectorStoreInputError("ids must not contain duplicates")
    return normalized_ids


def _normalize_id(value: str, *, input_name: str) -> str:
    if not isinstance(value, str):
        raise VectorStoreInputError(f"{input_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise VectorStoreInputError(f"{input_name} must not be blank")
    return normalized


def _normalize_document(value: str, *, input_name: str) -> str:
    if not isinstance(value, str):
        raise VectorStoreInputError(f"{input_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise VectorStoreInputError(f"{input_name} must not be blank")
    return normalized


def _normalize_embedding(
    embedding: Sequence[float],
    *,
    input_name: str,
) -> tuple[float, ...]:
    if isinstance(embedding, (str, bytes)):
        raise VectorStoreInputError(f"{input_name} must be a numeric sequence")
    try:
        raw_values = tuple(embedding)
    except TypeError as exc:
        raise VectorStoreInputError(f"{input_name} must be iterable") from exc
    if not raw_values:
        raise VectorStoreInputError(f"{input_name} must not be empty")

    values: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool):
            raise VectorStoreInputError(f"{input_name}[{index}] must be numeric")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise VectorStoreInputError(f"{input_name}[{index}] must be numeric") from exc
        if not isfinite(numeric_value):
            raise VectorStoreInputError(f"{input_name}[{index}] must be finite")
        values.append(numeric_value)
    return tuple(values)


def _normalize_metadata(
    metadata: Mapping[str, VectorMetadataValue],
    *,
    input_name: str,
) -> VectorMetadata:
    if not isinstance(metadata, Mapping) or not metadata:
        raise VectorStoreInputError(f"{input_name} must be a non-empty mapping")
    normalized: dict[str, VectorMetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise VectorStoreInputError(f"{input_name} keys must be non-blank strings")
        if isinstance(value, (str, int)) or (
            isinstance(value, float) and isfinite(value)
        ):
            normalized[key.strip()] = value
        else:
            raise VectorStoreInputError(
                f"{input_name}['{key}'] must be a finite string, integer, float, or boolean"
            )
    return MappingProxyType(normalized)


def _normalize_where_filter(
    where: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if where is None:
        return None
    if not isinstance(where, Mapping) or not where:
        raise VectorStoreInputError("where must be a non-empty mapping when provided")
    normalized: dict[str, object] = {}
    for key, value in where.items():
        if not isinstance(key, str) or not key.strip():
            raise VectorStoreInputError("where keys must be non-blank strings")
        normalized[key.strip()] = value
    return normalized


def _parse_query_result(raw_result: Mapping[str, object]) -> tuple[VectorSearchResult, ...]:
    if not isinstance(raw_result, Mapping):
        raise VectorStoreResultError("Chroma query result must be a mapping")

    ids = _extract_single_query_group(raw_result, "ids")
    documents = _extract_single_query_group(raw_result, "documents")
    metadatas = _extract_single_query_group(raw_result, "metadatas")
    distances = _extract_single_query_group(raw_result, "distances")
    lengths = {len(ids), len(documents), len(metadatas), len(distances)}
    if len(lengths) != 1:
        raise VectorStoreResultError("Chroma query result fields have different lengths")

    results: list[VectorSearchResult] = []
    for index, (record_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True)
    ):
        if not isinstance(record_id, str) or not record_id.strip():
            raise VectorStoreResultError(f"Chroma result ids[{index}] is invalid")
        if not isinstance(document, str) or not document.strip():
            raise VectorStoreResultError(f"Chroma result documents[{index}] is invalid")
        try:
            normalized_metadata = _normalize_metadata(
                metadata,
                input_name=f"Chroma result metadatas[{index}]",
            )
        except VectorStoreInputError as exc:
            raise VectorStoreResultError(str(exc)) from exc
        if isinstance(distance, bool):
            raise VectorStoreResultError(f"Chroma result distances[{index}] is invalid")
        try:
            normalized_distance = float(distance)
        except (TypeError, ValueError) as exc:
            raise VectorStoreResultError(
                f"Chroma result distances[{index}] is invalid"
            ) from exc
        if not isfinite(normalized_distance):
            raise VectorStoreResultError(f"Chroma result distances[{index}] is invalid")
        results.append(
            VectorSearchResult(
                id=record_id.strip(),
                document=document.strip(),
                metadata=normalized_metadata,
                distance=normalized_distance,
            )
        )
    return tuple(results)


def _extract_single_query_group(
    raw_result: Mapping[str, object],
    field: str,
) -> tuple[object, ...]:
    value = raw_result.get(field)
    if isinstance(value, (str, bytes)):
        raise VectorStoreResultError(f"Chroma query result field '{field}' is invalid")
    try:
        query_groups = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise VectorStoreResultError(
            f"Chroma query result field '{field}' is missing or invalid"
        ) from exc
    if len(query_groups) != 1:
        raise VectorStoreResultError(
            f"Chroma query result field '{field}' must contain one query group"
        )
    group = query_groups[0]
    if isinstance(group, (str, bytes)):
        raise VectorStoreResultError(f"Chroma query result field '{field}' is invalid")
    try:
        return tuple(group)  # type: ignore[arg-type]
    except TypeError as exc:
        raise VectorStoreResultError(
            f"Chroma query result field '{field}' contains an invalid group"
        ) from exc
