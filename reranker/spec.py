from __future__ import annotations

from datetime import datetime

import msgspec
import numpy as np


class Record(msgspec.Struct, kw_only=True):
    id: int | str = 0
    text: str
    title: str | None = None
    summary: str | None = None
    vector: list[float] | np.ndarray | None = None
    score: float = 1.0
    updated_at: datetime | None = None
    author: str | None = None
    tags: list[str] | None = None
    hidden: bool = False
    boost: float = 0
