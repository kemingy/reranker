# ReRanker

ReRanker for hybrid retrieval.

## Installation

```bash
pip install reranker
```

## Usage

```python
from datetime import datetime, timedelta
from reranker import ReRanker, CrossEncoderClient, TimeDecayRanker


reranker = ReRanker(
    steps=[
        CrossEncoderClient(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            addr="http://localhost:8000",
        ),
        TimeDecayRanker(decay_rate=0.5),
    ]
)

reranker.rank(
    query="What is the capital of France?",
    docs=[
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
        "The Louvre is in Paris.",
    ],
)
reranker.rank(
    query=Record(
        text="What is the capital of France?",
        timestamp=datetime.now(),
    ),
    docs=[
        Record(
            text="Paris is the capital of France.",
            timestamp=datetime.now() - timedelta(days=1),
        ),
        Record(
            text="The Eiffel Tower is in Paris.",
            timestamp=datetime.now() - timedelta(days=2),
        ),
        Record(
            text="The Louvre is in Paris.",
            timestamp=datetime.now() - timedelta(days=3),
        ),
    ]
)
```
