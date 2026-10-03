"""Provider-neutral knowledge ingestion and indexing orchestration."""

from collections.abc import Callable
from dataclasses import dataclass

from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    KnowledgeDocument,
)
from ai_engineering_agent_platform.domain.ingestion import (
    KnowledgeSource,
)
from ai_engineering_agent_platform.services.indexing import (
    IndexingResult,
    IndexingService,
)
from ai_engineering_agent_platform.services.ingestion_mapping import (
    parse_knowledge_source,
)

type DocumentChunker = Callable[
    [KnowledgeDocument],
    tuple[DocumentChunk, ...],
]


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionResult:
    """Validated stages preserved from one ingestion execution."""

    document: KnowledgeDocument
    chunks: tuple[DocumentChunk, ...]
    indexing: IndexingResult


def _validate_chunker_output(
    document: KnowledgeDocument,
    chunks: object,
) -> tuple[DocumentChunk, ...]:
    """Require chunk output to remain bound to the parsed document."""
    if not isinstance(chunks, tuple):
        raise ValueError("chunker must return a tuple of DocumentChunk values")

    if not chunks:
        raise ValueError("chunker must return at least one chunk")

    if any(not isinstance(chunk, DocumentChunk) for chunk in chunks):
        raise ValueError("chunker must return only DocumentChunk values")

    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunker returned duplicate chunk identifiers")

    indexes = tuple(chunk.index for chunk in chunks)

    if indexes != tuple(range(len(chunks))):
        raise ValueError("chunker must return contiguous chunk indexes")

    for chunk in chunks:
        if chunk.document_id != document.document_id:
            raise ValueError("chunk document identity does not match source document")

        if chunk.source_ref != document.source_ref:
            raise ValueError("chunk source reference does not match source document")

        if chunk.title != document.title:
            raise ValueError("chunk title does not match source document")

        if chunk.metadata != document.metadata:
            raise ValueError("chunk metadata does not match source document")

        expected_text = document.text[chunk.start_char : chunk.end_char]

        if chunk.text != expected_text:
            raise ValueError("chunk text does not match source offsets")

    return chunks


class KnowledgeIngestionService:
    """Parse bounded source bytes, validate chunks, and index them.

    Source acquisition is intentionally outside this service. It does not read
    filesystem paths, fetch URLs, perform retries, manage document replacement,
    or own provider lifecycle.
    """

    def __init__(
        self,
        indexing_service: IndexingService,
        *,
        chunker: DocumentChunker,
        max_source_bytes: int,
    ) -> None:
        """Store injected indexing/chunking capabilities and source bound."""
        if not callable(chunker):
            raise ValueError("chunker must be callable")

        if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int):
            raise ValueError("max_source_bytes must be an integer")

        if max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be positive")

        self._indexing_service = indexing_service
        self._chunker = chunker
        self._max_source_bytes = max_source_bytes

    async def ingest(
        self,
        source: KnowledgeSource,
        *,
        namespace: str | None = None,
    ) -> KnowledgeIngestionResult:
        """Execute one bounded source-to-vector indexing operation."""
        document = parse_knowledge_source(
            source,
            max_source_bytes=self._max_source_bytes,
        )

        chunks = _validate_chunker_output(
            document,
            self._chunker(document),
        )

        indexing = await self._indexing_service.index_chunks(
            chunks,
            namespace=namespace,
        )

        return KnowledgeIngestionResult(
            document=document,
            chunks=chunks,
            indexing=indexing,
        )
