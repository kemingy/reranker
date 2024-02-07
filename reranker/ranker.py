from __future__ import annotations

import abc
from typing import overload

import cohere
import httpx

from reranker.spec import Record


class Ranker(abc.ABC):
    @abc.abstractmethod
    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        pass


class CrossEncoderClient(Ranker):
    def __init__(self, model_name: str, addr: str, top_k: int = 0):
        self.model_name = model_name
        self.client = httpx.Client(base_url=addr)
        self.top_k = top_k

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        top_k = len(docs) if self.top_k == 0 else self.top_k
        resp = self.client.post(
            "/inference",
            json={
                "query": query.text,
                "docs": [doc.text for doc in docs],
                "top_k": top_k,
            },
        )
        scores = resp.json()
        return docs.sort(key=scores.__getitem__, reverse=True)[:top_k]


class CohereClient(Ranker):
    def __init__(self, model_name: str, key: str, top_k: int = 0):
        self.model_name = model_name
        self.client = cohere.Client(api_key=key)
        self.top_k = top_k

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        top_k = len(docs) if self.top_k == 0 else self.top_k
        ranks = self.client.rerank(
            query=query.text,
            documents=[doc.text for doc in docs],
            top_k=top_k,
            model=self.model_name,
        )
        scores = [rank.relevance_score for rank in ranks.results]
        return docs.sort(key=scores.__getitem__, reverse=True)[:top_k]


class DiverseRanker(Ranker):
    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        if not docs or docs[0].vector is None:
            raise ValueError("`vector` is required for diverse ranking")
        raise NotImplementedError


class TimeDecayRanker(Ranker):
    def __init__(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        if not docs or docs[0].created_at is None:
            raise ValueError("doc `created_at` is required for time decay ranking")
        raise NotImplementedError


class ReRanker:
    def __init__(self, steps: list[Ranker]) -> None:
        self.steps = steps

    def rank_records(self, query: Record, docs: list[Record]) -> list[Record]:
        for step in self.steps:
            docs = step.rank(query, docs)
        return docs

    @overload
    def rank(self, query: str, docs: list[str]) -> list[str]:
        pass

    @overload
    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        pass

    def rank(self, query, docs):
        if isinstance(query, Record) and docs and isinstance(docs[0], Record):
            return self.rank_records(query, docs)
        results = self.rank_records(
            Record(text=query), [Record(id=i, text=doc) for (i, doc) in enumerate(docs)]
        )
        return [result.text for result in results]
