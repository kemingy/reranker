from __future__ import annotations

import abc
from datetime import datetime
from typing import overload

import cohere
import httpx

from reranker.distance import Distance
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

    def score(self, query: Record, docs: list[Record], top_k: int) -> list[float]:
        resp = self.client.post(
            "/inference",
            json={
                "query": query.text,
                "docs": [doc.text for doc in docs],
                "top_k": top_k,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        top_k = len(docs) if self.top_k == 0 else self.top_k
        scores = self.score(query, docs, top_k)
        return docs.sort(key=scores.__getitem__, reverse=True)[:top_k]


class CohereClient(Ranker):
    def __init__(self, model_name: str, key: str, top_k: int = 0):
        self.model_name = model_name
        self.client = cohere.Client(api_key=key)
        self.top_k = top_k

    def score(self, query: Record, docs: list[Record], top_k: int) -> list[float]:
        ranks = self.client.rerank(
            query=query.text,
            documents=[doc.text for doc in docs],
            top_k=top_k,
            model=self.model_name,
        )
        scores = [rank.relevance_score for rank in ranks.results]
        return scores

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        top_k = len(docs) if self.top_k == 0 else self.top_k
        scores = self.score(query, docs, top_k)
        return docs.sort(key=scores.__getitem__, reverse=True)[:top_k]


class DiverseRanker(Ranker):
    def __init__(
        self,
        lambda_const: float = 0.3,
        threshold: float = 0,
        distance: Distance = Distance.COSINE,
    ):
        """
        Rank documents by diversity computed by maximal marginal relevance (MMR).

        Link: https://www.cs.bilkent.edu.tr/~canf/CS533/hwSpring14/eightMinPresentations/handoutMMR.pdf

        Args:
            lambda_const: The parameter $\lambda$ in MMR. The value should be
                between 0 and 1. A lower value will result in more diverse ranking.
            threshold: The threshold to filter out similar documents.
            distance: The distance metric used to compute the similarity score.
        """
        self.lambda_const = lambda_const
        self.distance = distance
        self.threshold = threshold

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        if not docs or docs[0].vector is None:
            raise ValueError("`vector` is required for diverse ranking")

        # pre-compute similarity scores
        similarity: dict[str, float] = {}
        length = len(docs)
        for i in range(length):
            for j in range(i + 1, length):
                similarity[f"{i}_{j}"] = similarity[f"{j}_{i}"] = self.distance(
                    docs[i].vector, docs[j].vector
                )
            similarity[f"query_{i}"] = self.distance(query.vector, docs[i].vector)

        candidates, selected = list(range(len(docs))), []
        while candidates:
            scores = []
            for c in candidates:
                sim_q = similarity[f"query_{c}"]
                sim_d = max(similarity[f"{c}_{s}"] for s in selected) if selected else 0
                scores.append(
                    self.lambda_const * sim_q - (1 - self.lambda_const) * sim_d
                )
            mmr = max(scores)
            if mmr < self.threshold:
                break
            max_idx = scores.index(max(scores))
            selected.append(candidates.pop(max_idx))
        return [docs[i] for i in selected]


class TimeDecayRanker(Ranker):
    def __init__(self, decay_rate: float = 1.8) -> None:
        """
        Rank documents by time decay.

        The equation for time decay is derived from the HackerNews algorithm.

        $rank_score = score / ((hours_since_updated + 2) ** decay_rate)$
        """
        self.decay_rate = decay_rate

    def score(self, query: Record, docs: list[Record]) -> list[float]:
        if not docs or docs[0].updated_at is None:
            raise ValueError("doc `created_at` is required for time decay ranking")
        scores = [
            record.score
            / (
                (2 + ((datetime.now() - record.updated_at).seconds / 3600.0))
                ** self.decay_rate
            )
            for record in docs
        ]
        return scores

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        scores = self.score(query, docs)
        return docs.sort(key=scores.__getitem__, reverse=True)


class KeywordBoost(Ranker):
    def __init__(self, title_content_ratio: float = 0.7) -> None:
        self.title_content_ratio = title_content_ratio

    def score(self, query: Record, docs: list[Record]) -> list[float]:
        if docs and docs[0].title_bm25:
            scores = [
                (
                    doc.title_bm25 * self.title_content_ratio
                    + doc.content_bm25 * (1 - self.title_content_ratio)
                )
                * doc.boost
                for doc in docs
            ]
        else:
            scores = [doc.content_bm25 * doc.boost for doc in docs]
        return scores

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        scores = self.score(query, docs)
        return docs.sort(key=scores.__getitem__, reverse=True)


class VectorBoost(Ranker):
    def __init__(self, title_content_ratio: float = 0.7) -> None:
        self.title_content_ratio = title_content_ratio

    def score(self, query: Record, docs: list[Record]) -> list[float]:
        if docs and docs[0].title_sim:
            scores = [
                (
                    doc.title_sim * self.title_content_ratio
                    + doc.vector_sim * (1 - self.title_content_ratio)
                )
                * doc.boost
                for doc in docs
            ]
        else:
            scores = [doc.vector_sim * doc.boost for doc in docs]
        return scores

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        scores = self.score(query, docs)
        return docs.sort(key=scores.__getitem__, reverse=True)


class HybridRanker(Ranker):
    def __init__(self, decay_rate: float = 1.8, title_content_ratio: float = 0.7):
        self.decay_ranker = TimeDecayRanker(decay_rate)
        self.vector_ranker = VectorBoost(title_content_ratio)

    def score(self, query: Record, docs: list[Record]) -> list[float]:
        decay_score = self.decay_ranker.score(query, docs)
        vector_score = self.vector_ranker.score(query, docs)
        return [decay * vector for decay, vector in zip(decay_score, vector_score)]

    def rank(self, query: Record, docs: list[Record]) -> list[Record]:
        scores = self.score(query, docs)
        return docs.sort(key=scores.__getitem__, reverse=True)


class ReRanker:
    def __init__(self, steps: list[Ranker]) -> None:
        if not steps:
            raise ValueError("At least one ranker is required")
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