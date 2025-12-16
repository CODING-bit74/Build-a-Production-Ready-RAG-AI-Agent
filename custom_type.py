from typing import List
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# Chunk + Source (INTERNAL USE ONLY)
# --------------------------------------------------
class RAGChunkAndSrc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: List[str] = Field(..., description="Text chunks extracted from source")
    source_id: str = Field(..., description="Unique source identifier")

    def to_dict(self) -> dict:
        """Explicit JSON-safe conversion"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "RAGChunkAndSrc":
        return cls(**data)


# --------------------------------------------------
# Ingestion Result (INTERNAL USE ONLY)
# --------------------------------------------------
class RAGUpsertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingested: int = Field(..., ge=0, description="Number of chunks ingested")

    def to_dict(self) -> dict:
        return self.model_dump()


# --------------------------------------------------
# Search Result (API OUTPUT)
# --------------------------------------------------
class RAGSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contexts: List[str] = Field(..., description="Retrieved text contexts")
    sources: List[str] = Field(..., description="Source IDs for contexts")


# --------------------------------------------------
# Query Result (API OUTPUT)
# --------------------------------------------------
class RAGQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="Final generated answer")
    sources: List[str] = Field(..., description="Sources used in answer")
    num_contexts: int = Field(..., ge=0, description="Number of contexts used")
