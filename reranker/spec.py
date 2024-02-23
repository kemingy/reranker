from __future__ import annotations

import sys
from datetime import datetime

import msgspec
import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class Record(msgspec.Struct, kw_only=True):
    id: int | str = 0
    text: str
    title: str | None = None
    summary: str | None = None
    vector: list[float] | np.ndarray | None = None
    title_vector: list[float] | np.ndarray | None = None
    score: float = 1.0
    vector_sim: float = 1.0
    title_sim: float = 1.0
    content_bm25: float = 1.0
    title_bm25: float = 1.0
    updated_at: datetime | None = None
    author: str | None = None
    tags: list[str] | None = None
    hidden: bool = False
    boost: float = 1.0

    def use_np(self) -> Self:
        if isinstance(self.vector, list):
            self.vector = np.array(self.vector)
        if isinstance(self.title_vector, list):
            self.title_vector = np.array(self.title_vector)
        return self

    def to_json(self) -> bytes:
        return msgspec.json.encode(self)

    @classmethod
    def from_json(cls, buf: bytes) -> Self:
        return msgspec.json.decode(buf, type=cls)

    @classmethod
    def decode_batch(cls, buf: bytes) -> list[Self]:
        return msgspec.json.decode(buf, type=list[cls])

    @staticmethod
    def encode_batch(records: list[Record]) -> bytes:
        return msgspec.json.encode(records)
