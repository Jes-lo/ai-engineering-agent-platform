"""Infrastructure-independent grounded answer and citation primitives."""

from dataclasses import dataclass
from enum import StrEnum

from ai_engineering_agent_platform.domain.retrieval import RetrievedEvidence


class GroundedAnswerStatus(StrEnum):
    """Normalized outcome of one grounded generation operation."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class GroundedCitation:
    """One validated citation pointing only to retrieved evidence."""

    citation_id: str
    evidence: RetrievedEvidence

    def __post_init__(self) -> None:
        """Validate the platform-owned citation identifier."""
        if not isinstance(self.citation_id, str):
            raise ValueError("citation_id must be a string")

        if not self.citation_id.startswith("C"):
            raise ValueError("citation_id must start with C")

        suffix = self.citation_id[1:]

        if (
            not suffix
            or not suffix.isascii()
            or not suffix.isdigit()
            or suffix.startswith("0")
        ):
            raise ValueError("citation_id must contain a canonical positive integer")


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Validated grounded answer or explicit evidence abstention."""

    query: str
    status: GroundedAnswerStatus
    answer: str | None
    citations: tuple[GroundedCitation, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate grounded answer state and citation invariants."""
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must not be empty")

        if self.namespace is not None and (
            not isinstance(self.namespace, str) or not self.namespace.strip()
        ):
            raise ValueError("namespace must not be empty")

        citation_ids = tuple(citation.citation_id for citation in self.citations)

        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("grounded citations must use unique identifiers")

        if self.status is GroundedAnswerStatus.ANSWERED:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("answered result must contain answer text")

            if not self.citations:
                raise ValueError("answered result must contain citations")

            return

        if self.status is GroundedAnswerStatus.ABSTAINED:
            if self.answer is not None:
                raise ValueError("abstained result must not contain answer text")

            if self.citations:
                raise ValueError("abstained result must not contain citations")

            return

        raise ValueError("unsupported grounded answer status")
