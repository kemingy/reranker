from datetime import datetime, timedelta

import numpy as np
import pytest

from reranker.ranker import (
    DiverseRanker,
    HybridRanker,
    KeywordBoost,
    ReRanker,
    TimeDecayRanker,
    VectorBoost,
)
from reranker.spec import Record


@pytest.fixture
def records(request):
    num, dim = request.param
    return [
        Record(
            id=i,
            text=f"pseudo text generated for testing with id {i}",
            vector=np.random.rand(dim),
            updated_at=datetime.now() - timedelta(days=i),
        )
        for i in range(num)
    ]


@pytest.mark.parametrize(
    "records", [pytest.param((10, 64), id="time-decay")], indirect=True
)
def test_time_decay_ranker(records):
    ranker = TimeDecayRanker()
    query = Record(text="test query")
    ranked = ranker.rank(query, records)
    assert ranked[0].id == 0
    assert ranked[-1].id == len(records) - 1


@pytest.mark.parametrize(
    "records", [pytest.param((10, 64), id="diversity")], indirect=True
)
def test_diversity_ranker(records):
    ranker = DiverseRanker()
    query = Record(text="test query", vector=np.random.rand(64).tolist())
    ranked = ranker.rank(query, records)
    assert ranked

    # disable threshold
    ranker.threshold = -float("inf")
    ranked = ranker.rank(query, records)
    assert len(ranked) == len(records)


@pytest.mark.parametrize(
    "records, ranker",
    [
        pytest.param((10, 64), VectorBoost, id="vector-boost"),
        pytest.param((10, 64), KeywordBoost, id="keyword-boost"),
        pytest.param((10, 64), HybridRanker, id="hybrid-ranker"),
    ],
    indirect=["records"],
)
def test_ranker(records, ranker):
    query = Record(text="test query")
    ranked = ranker().rank(query, records)
    assert len(ranked) == len(records)


@pytest.mark.parametrize(
    "records", [pytest.param((10, 64), id="re-ranker")], indirect=True
)
def test_reranker(records):
    reranker = ReRanker([TimeDecayRanker(), VectorBoost()])
    query = Record(text="test query")
    ranked = reranker.rank(query, records)
    assert len(ranked) == len(records)
