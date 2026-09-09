"""Unit tests for the persistent Chroma vector-store adapter."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from backend.app.config.settings import Settings
from backend.app.rag.vector_store.base import VectorRecord, VectorStore
from backend.app.rag.vector_store.chroma_store import ChromaVectorStore
from backend.app.rag.vector_store.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreDependencyError,
    VectorStoreInputError,
    VectorStoreOperationError,
    VectorStoreResultError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeCollection:
    """Small Chroma collection replacement that records every operation."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.count_calls = 0
        self.count_result: object = 2
        self.query_result: Mapping[str, object] = {
            "ids": [["faq-1", "faq-2"]],
            "documents": [["商品新增步骤", "库存查询步骤"]],
            "metadatas": [[{"row_number": 2}, {"row_number": 3}]],
            "distances": [[0.1, 0.25]],
        }
        self.errors: dict[str, Exception] = {}

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(dict(kwargs))
        if error := self.errors.get("upsert"):
            raise error

    def query(self, **kwargs: object) -> Mapping[str, object]:
        self.query_calls.append(dict(kwargs))
        if error := self.errors.get("query"):
            raise error
        return self.query_result

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(dict(kwargs))
        if error := self.errors.get("delete"):
            raise error

    def count(self) -> object:
        self.count_calls += 1
        if error := self.errors.get("count"):
            raise error
        return self.count_result


class FakeClient:
    """Records collection creation and returns one fake collection."""

    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.collection_calls: list[dict[str, object]] = []
        self.collection_error: Exception | None = None

    def get_or_create_collection(self, **kwargs: object) -> FakeCollection:
        self.collection_calls.append(dict(kwargs))
        if self.collection_error is not None:
            raise self.collection_error
        return self.collection


class RecordingClientFactory:
    """Records the path supplied to the injected persistent-client factory."""

    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.calls: list[Path] = []
        self.error: Exception | None = None

    def __call__(self, persist_directory: Path) -> FakeClient:
        self.calls.append(persist_directory)
        if self.error is not None:
            raise self.error
        return self.client


class RecordingDirectoryPreparer:
    """Records directory preparation without touching the real filesystem."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.error: Exception | None = None

    def __call__(self, persist_directory: Path) -> None:
        self.calls.append(persist_directory)
        if self.error is not None:
            raise self.error


type Components = tuple[
    ChromaVectorStore,
    FakeCollection,
    FakeClient,
    RecordingClientFactory,
    RecordingDirectoryPreparer,
]


@pytest.fixture
def persist_directory() -> Path:
    return PROJECT_ROOT / "data/test-chroma"


@pytest.fixture
def chroma_components(persist_directory: Path) -> Components:
    collection = FakeCollection()
    client = FakeClient(collection)
    factory = RecordingClientFactory(client)
    preparer = RecordingDirectoryPreparer()
    store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name="erp_faq_test",
        distance_metric="cosine",
        default_query_limit=5,
        client_factory=factory,
        directory_preparer=preparer,
    )
    return store, collection, client, factory, preparer


def make_record(
    record_id: str = "faq-1",
    *,
    document: str = "商品新增步骤",
    embedding: Sequence[float] = (0.1, 0.2, 0.3),
    metadata: Mapping[str, object] | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        document=document,
        embedding=tuple(embedding),
        metadata=(
            metadata
            if metadata is not None
            else {"row_number": 2, "source": "erp_faq.xlsx"}
        ),
    )  # type: ignore[arg-type]


def test_store_construction_is_lazy_and_implements_contract(
    chroma_components: Components,
) -> None:
    store, _, client, factory, preparer = chroma_components

    assert isinstance(store, VectorStore)
    assert store.provider_name == "chroma"
    assert store.collection_name == "erp_faq_test"
    assert preparer.calls == []
    assert factory.calls == []
    assert client.collection_calls == []


def test_from_settings_maps_configuration_without_opening_chroma(
    persist_directory: Path,
) -> None:
    collection = FakeCollection()
    client = FakeClient(collection)
    factory = RecordingClientFactory(client)
    preparer = RecordingDirectoryPreparer()
    settings = Settings(
        chroma_persist_directory=persist_directory,
        chroma_collection_name="erp_vectors",
        chroma_distance_metric="ip",
        chroma_query_limit=7,
        _env_file=None,
    )

    store = ChromaVectorStore.from_settings(
        settings,
        client_factory=factory,
        directory_preparer=preparer,
    )

    assert store.persist_directory == persist_directory.resolve()
    assert store.collection_name == "erp_vectors"
    assert store.distance_metric == "ip"
    assert store.default_query_limit == 7
    assert preparer.calls == []
    assert factory.calls == []


def test_first_operation_prepares_and_opens_persistent_collection_once(
    chroma_components: Components,
) -> None:
    store, _, client, factory, preparer = chroma_components

    assert store.count() == 2
    assert store.count() == 2

    assert preparer.calls == [store.persist_directory]
    assert factory.calls == [store.persist_directory]
    assert client.collection_calls == [
        {
            "name": "erp_faq_test",
            "configuration": {"hnsw": {"space": "cosine"}},
            "embedding_function": None,
        }
    ]


def test_empty_upsert_and_delete_do_not_open_chroma(
    chroma_components: Components,
) -> None:
    store, collection, _, factory, preparer = chroma_components

    assert store.upsert(()) == 0
    store.delete(())

    assert preparer.calls == []
    assert factory.calls == []
    assert collection.upsert_calls == []
    assert collection.delete_calls == []


def test_upsert_maps_records_to_chroma_payload(chroma_components: Components) -> None:
    store, collection, _, _, _ = chroma_components
    records = [
        make_record(" faq-1 ", document=" 商品新增步骤 "),
        make_record(
            "faq-2",
            document="库存查询步骤",
            embedding=(0.4, 0.5, 0.6),
            metadata={"row_number": 3, "active": True},
        ),
    ]

    assert store.upsert(records) == 2
    assert collection.upsert_calls == [
        {
            "ids": ["faq-1", "faq-2"],
            "documents": ["商品新增步骤", "库存查询步骤"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "metadatas": [
                {"row_number": 2, "source": "erp_faq.xlsx"},
                {"row_number": 3, "active": True},
            ],
        }
    ]


@pytest.mark.parametrize(
    "records",
    [
        [make_record(" ")],
        [make_record("faq-1"), make_record("faq-1")],
        [make_record(document="  ")],
        [make_record(embedding=())],
        [make_record(embedding=(0.1, float("nan")))],
        [make_record(embedding=(0.1, True))],
        [make_record(metadata={})],
        [make_record(metadata={" ": "value"})],
        [make_record(metadata={"nested": {"bad": "value"}})],
        [make_record("faq-1"), make_record("faq-2", embedding=(0.1, 0.2))],
    ],
)
def test_invalid_upsert_records_are_rejected_before_opening_chroma(
    chroma_components: Components,
    records: list[VectorRecord],
) -> None:
    store, collection, _, factory, preparer = chroma_components

    with pytest.raises(VectorStoreInputError):
        store.upsert(records)

    assert preparer.calls == []
    assert factory.calls == []
    assert collection.upsert_calls == []


def test_query_uses_default_limit_filter_and_required_result_fields(
    chroma_components: Components,
) -> None:
    store, collection, _, _, _ = chroma_components

    results = store.query(
        (0.1, 0.2, 0.3),
        where={"module": "inventory"},
    )

    assert collection.query_calls == [
        {
            "query_embeddings": [[0.1, 0.2, 0.3]],
            "n_results": 5,
            "include": ["documents", "metadatas", "distances"],
            "where": {"module": "inventory"},
        }
    ]
    assert [(result.id, result.document, result.distance) for result in results] == [
        ("faq-1", "商品新增步骤", 0.1),
        ("faq-2", "库存查询步骤", 0.25),
    ]
    assert dict(results[0].metadata) == {"row_number": 2}


def test_query_limit_can_be_overridden(chroma_components: Components) -> None:
    store, collection, _, _, _ = chroma_components

    store.query((1.0, 0.0), limit=1)

    assert collection.query_calls[0]["n_results"] == 1
    assert "where" not in collection.query_calls[0]


@pytest.mark.parametrize(
    ("embedding", "limit", "where"),
    [
        ((), None, None),
        ((True, 0.2), None, None),
        ((float("inf"), 0.2), None, None),
        ((0.1, 0.2), 0, None),
        ((0.1, 0.2), 101, None),
        ((0.1, 0.2), None, {}),
        ((0.1, 0.2), None, {" ": "value"}),
    ],
)
def test_invalid_query_input_is_rejected_before_opening_chroma(
    chroma_components: Components,
    embedding: Sequence[float],
    limit: int | None,
    where: Mapping[str, object] | None,
) -> None:
    store, collection, _, factory, preparer = chroma_components

    with pytest.raises(VectorStoreInputError):
        store.query(embedding, limit=limit, where=where)

    assert preparer.calls == []
    assert factory.calls == []
    assert collection.query_calls == []


@pytest.mark.parametrize(
    "query_result",
    [
        {},
        {
            "ids": [["faq-1"]],
            "documents": [[]],
            "metadatas": [[{"row_number": 2}]],
            "distances": [[0.1]],
        },
        {
            "ids": [["faq-1"]],
            "documents": [[None]],
            "metadatas": [[{"row_number": 2}]],
            "distances": [[0.1]],
        },
        {
            "ids": [["faq-1"]],
            "documents": [["document"]],
            "metadatas": [[None]],
            "distances": [[0.1]],
        },
        {
            "ids": [["faq-1"]],
            "documents": [["document"]],
            "metadatas": [[{"row_number": 2}]],
            "distances": [[float("nan")]],
        },
    ],
)
def test_malformed_query_results_are_rejected(
    chroma_components: Components,
    query_result: Mapping[str, object],
) -> None:
    store, collection, _, _, _ = chroma_components
    collection.query_result = query_result

    with pytest.raises(VectorStoreResultError):
        store.query((0.1, 0.2, 0.3))


def test_delete_normalizes_ids_and_calls_collection(
    chroma_components: Components,
) -> None:
    store, collection, _, _, _ = chroma_components

    store.delete([" faq-1 ", "faq-2"])

    assert collection.delete_calls == [{"ids": ["faq-1", "faq-2"]}]


@pytest.mark.parametrize("ids", [[""], ["faq-1", "faq-1"]])
def test_invalid_delete_ids_are_rejected(
    chroma_components: Components,
    ids: list[str],
) -> None:
    store, collection, _, factory, preparer = chroma_components

    with pytest.raises(VectorStoreInputError):
        store.delete(ids)

    assert preparer.calls == []
    assert factory.calls == []
    assert collection.delete_calls == []


@pytest.mark.parametrize("count_result", [-1, True, "2"])
def test_invalid_count_result_is_rejected(
    chroma_components: Components,
    count_result: object,
) -> None:
    store, collection, _, _, _ = chroma_components
    collection.count_result = count_result

    with pytest.raises(VectorStoreResultError):
        store.count()


@pytest.mark.parametrize("operation", ["upsert", "query", "delete", "count"])
def test_collection_operation_failures_are_wrapped(
    chroma_components: Components,
    operation: str,
) -> None:
    store, collection, _, _, _ = chroma_components
    backend_error = RuntimeError("Chroma backend failure")
    collection.errors[operation] = backend_error

    with pytest.raises(VectorStoreOperationError, match=operation) as error:
        if operation == "upsert":
            store.upsert([make_record()])
        elif operation == "query":
            store.query((0.1, 0.2, 0.3))
        elif operation == "delete":
            store.delete(["faq-1"])
        else:
            store.count()

    assert error.value.__cause__ is backend_error


def test_client_initialization_failure_is_wrapped(
    chroma_components: Components,
) -> None:
    store, _, _, factory, _ = chroma_components
    factory.error = OSError("path cannot be opened")

    with pytest.raises(VectorStoreOperationError, match="initialize") as error:
        store.count()

    assert error.value.__cause__ is factory.error


def test_directory_preparation_failure_is_wrapped(
    chroma_components: Components,
) -> None:
    store, _, _, factory, preparer = chroma_components
    preparer.error = PermissionError("directory is read-only")

    with pytest.raises(VectorStoreOperationError, match="initialize") as error:
        store.count()

    assert error.value.__cause__ is preparer.error
    assert factory.calls == []


def test_collection_initialization_failure_is_wrapped(
    chroma_components: Components,
) -> None:
    store, _, client, _, _ = chroma_components
    client.collection_error = RuntimeError("collection configuration conflict")

    with pytest.raises(VectorStoreOperationError, match="collection") as error:
        store.count()

    assert error.value.__cause__ is client.collection_error


def test_dependency_failure_is_preserved(persist_directory: Path) -> None:
    def missing_chroma(path: Path) -> FakeClient:
        raise VectorStoreDependencyError(f"Chroma unavailable at {path}")

    store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name="erp_faq_test",
        client_factory=missing_chroma,
        directory_preparer=RecordingDirectoryPreparer(),
    )

    with pytest.raises(VectorStoreDependencyError, match="Chroma unavailable"):
        store.count()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persist_directory": "  "},
        {"collection_name": "ab"},
        {"collection_name": "127.0.0.1"},
        {"distance_metric": "euclidean"},
        {"default_query_limit": 0},
    ],
)
def test_invalid_store_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "persist_directory": "data/chroma",
        "collection_name": "erp_faq",
        "distance_metric": "cosine",
        "default_query_limit": 5,
    }
    defaults.update(kwargs)

    with pytest.raises(VectorStoreConfigurationError):
        ChromaVectorStore(**defaults)
