from __future__ import annotations

import numpy as np
import pytest

from ragas.embeddings.base import InfinityEmbeddings

try:
    import infinity_emb  # noqa
    import torch  # noqa

    INFINITY_AVAILABLE = True
except ImportError:
    INFINITY_AVAILABLE = False


@pytest.mark.skipif(not INFINITY_AVAILABLE, reason="infinity_emb is not installed.")
@pytest.mark.asyncio
async def test_basic_embedding():
    embedding_engine = InfinityEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    async with embedding_engine:
        embeddings = await embedding_engine.aembed_documents(
            [
                "Paris is in France",
                "The capital of France is Paris",
                "Infintiy batches embeddings on the fly",
            ]
            * 20
        )
    assert isinstance(embeddings, list)
    array = np.array(embeddings)
    assert array.shape == (60, 384)
    assert array[0] @ array[1] > array[0] @ array[2]


@pytest.mark.skipif(not INFINITY_AVAILABLE, reason="infinity_emb is not installed.")
@pytest.mark.asyncio
async def test_rerank():
    rerank_engine = InfinityEmbeddings(model_name="BAAI/bge-reranker-base")

    async with rerank_engine:
        rankings = await rerank_engine.arerank(
            "Where is Paris?",
            [
                "Paris is in France",
                "I don't know the capital of Paris.",
                "Dummy sentence",
            ],
        )
    assert len(rankings) == 3
    assert rankings[0] > rankings[1]
    assert rankings[0] > rankings[2]
