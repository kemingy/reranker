from __future__ import annotations

from datetime import datetime

import msgspec
import numpy as np


class Record(msgspec.Struct, kw_only=True):
    id: int | str = 0
    text: str
    vector: list[float] | np.ndarray | None = None
    created_at: datetime | None = None
