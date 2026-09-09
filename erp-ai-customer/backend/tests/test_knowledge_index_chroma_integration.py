"""Integration test for ERP knowledge indexing with a real persistent Chroma client."""

from __future__ import annotations

import gc
import shutil
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import cast
from uuid import uuid4

import chromadb

from backend.app.knowledge.models import KnowledgeParseResult, KnowledgeRecord
from backend.app.rag.embeddings.service import EmbeddingService
from backend.app.rag.indexing.service import KnowledgeIndexService
from backend.app.rag.vector_store.chroma_store import ChromaClient, ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeterministicEmbeddingProvider:
    """Small offline embedding provider for the Chroma persistence boundary."""

    provider_name = "integration-test"
    model_name = "deterministic-3d"
    dimensions = 3

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> tuple[float, float, float]:
        if "库存" in text:
            return (0.0, 1.0, 0.0)
        return (1.0, 0.0, 0.0)


def test_index_service_persists_records_in_real_chroma() -> None:
    persist_directory = PROJECT_ROOT / "data" / f"chroma-integration-{uuid4().hex}"
    collection_name = f"erp_faq_integration_{uuid4().hex}"
    provider = DeterministicEmbeddingProvider()
    clients: list[ChromaClient] = []

    def create_client(path: Path) -> ChromaClient:
        client = cast(ChromaClient, chromadb.PersistentClient(path=str(path)))
        clients.append(client)
        return client

    try:
        first_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric="cosine",
            client_factory=create_client,
        )
        service = KnowledgeIndexService(
            embedding_service=EmbeddingService(provider),
            vector_store=first_store,
        )
        parse_result = KnowledgeParseResult(
            sheet_name="ERP FAQ",
            total_rows=2,
            records=(
                KnowledgeRecord(2, "如何新增客户？", "进入客户管理后点击新增。"),
                KnowledgeRecord(3, "如何查询库存？", "进入库存查询页面选择仓库。"),
            ),
            issues=(),
        )

        index_result = service.index_parse_result(
            parse_result,
            source_name="integration_faq.xlsx",
        )

        assert index_result.indexed_records == 2
        assert first_store.count() == 2
        first_hits = first_store.query(provider.embed_query("如何新增客户？"), limit=1)
        assert len(first_hits) == 1
        assert first_hits[0].id == index_result.record_ids[0]
        assert first_hits[0].metadata["question"] == "如何新增客户？"

        second_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric="cosine",
            client_factory=create_client,
        )
        assert second_store.count() == 2
        persisted_hits = second_store.query(
            provider.embed_query("如何查询库存？"),
            limit=1,
        )
        assert len(persisted_hits) == 1
        assert persisted_hits[0].id == index_result.record_ids[1]
        assert persisted_hits[0].document.endswith("进入库存查询页面选择仓库。")
    finally:
        for client in reversed(clients):
            close = getattr(client, "close", None)
            if callable(close):
                close()
        gc.collect()
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
        with suppress(OSError):
            persist_directory.parent.rmdir()
