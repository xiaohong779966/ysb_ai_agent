"""Integration test for ERP similarity retrieval with persistent Chroma."""

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
from backend.app.rag.retrieval.service import KnowledgeRetrievalService
from backend.app.rag.vector_store.chroma_store import ChromaClient, ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeterministicRetrievalProvider:
    """Offline vectors that separate customer questions from inventory questions."""

    provider_name = "retrieval-integration"
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


def test_retrieval_service_returns_nearest_indexed_faq_from_real_chroma() -> None:
    persist_directory = PROJECT_ROOT / "data" / f"chroma-retrieval-{uuid4().hex}"
    collection_name = f"erp_faq_retrieval_{uuid4().hex}"
    clients: list[ChromaClient] = []

    def create_client(path: Path) -> ChromaClient:
        client = cast(ChromaClient, chromadb.PersistentClient(path=str(path)))
        clients.append(client)
        return client

    try:
        provider = DeterministicRetrievalProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric="cosine",
            client_factory=create_client,
        )
        index_service = KnowledgeIndexService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        index_service.index_parse_result(
            KnowledgeParseResult(
                sheet_name="ERP FAQ",
                total_rows=2,
                records=(
                    KnowledgeRecord(2, "如何新增客户？", "进入客户管理后点击新增。"),
                    KnowledgeRecord(3, "如何查询库存？", "进入库存查询页面选择仓库。"),
                ),
                issues=(),
            ),
            source_name="retrieval_faq.xlsx",
        )
        retrieval_service = KnowledgeRetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_limit=1,
        )

        result = retrieval_service.search("现在还有多少库存？")

        assert result.total_hits == 1
        assert result.limit == 1
        assert result.collection_name == collection_name
        assert result.hits[0].question == "如何查询库存？"
        assert result.hits[0].answer == "进入库存查询页面选择仓库。"
        assert result.hits[0].source_file == "retrieval_faq.xlsx"
        assert result.hits[0].distance == 0.0
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
