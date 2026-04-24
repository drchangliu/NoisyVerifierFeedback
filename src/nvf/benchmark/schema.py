from __future__ import annotations

from pydantic import BaseModel


class BenchmarkItem(BaseModel):
    """A single benchmark task with code generation prompt and ground truth."""

    item_id: str
    cwe_id: str
    prompt: str
    insecure_code: str
    secure_code: str | None = None
    test_code: str | None = None  # Full pytest file content (functional + security)
    source: str = ""  # "cweval" or "securityeval"
