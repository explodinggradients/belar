from __future__ import annotations

import asyncio
import typing as t
from abc import ABC, abstractmethod
from dataclasses import field
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.dataclasses import dataclass

from ragas.run_config import RunConfig, add_async_retry, add_retry

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

if t.TYPE_CHECKING:
    from torch import Tensor


class BaseRagasEmbeddings(Embeddings, ABC):
    run_config: RunConfig

    async def embed_text(self, text: str, is_async=True) -> List[float]:
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(
        self, texts: List[str], is_async: bool = True
    ) -> t.List[t.List[float]]:
        if is_async and hasattr(self, "aembed_documents"):
            aembed_documents_with_retry = add_async_retry(
                self.aembed_documents, self.run_config
            )
            return await aembed_documents_with_retry(texts)
        else:
            loop = asyncio.get_event_loop()
            embed_documents_with_retry = add_retry(
                self.embed_documents, self.run_config
            )
            return await loop.run_in_executor(None, embed_documents_with_retry, texts)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...


class LangchainEmbeddingsWrapper(BaseRagasEmbeddings):
    def __init__(
        self, embeddings: Embeddings, run_config: t.Optional[RunConfig] = None
    ):
        self.embeddings = embeddings
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)

    async def aembed_documents(self, texts: List[str]):
        return await self.embeddings.aembed_documents(texts)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # run configurations specially for OpenAI
        if isinstance(self.embeddings, OpenAIEmbeddings):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.embeddings.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError


@dataclass
class HuggingfaceEmbeddings(BaseRagasEmbeddings):
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: t.Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import sentence_transformers
            from transformers import AutoConfig
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        config = AutoConfig.from_pretrained(self.model_name)
        self.is_cross_encoder = bool(
            np.intersect1d(
                list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                config.architectures,
            )
        )

        if self.is_cross_encoder:
            self._ce = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
            self.model = self._ce
            self.is_cross_encoder = True
        else:
            self._st = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )
            self.model = self._st
            self.is_cross_encoder = False

        # ensure outputs are tensors
        self.encode_kwargs["convert_to_tensor"] = True

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        assert not self.is_cross_encoder, "Model is not of the type Bi-encoder"
        embeddings: Tensor = self._st.encode(  # type: ignore
            texts, normalize_embeddings=True, **self.encode_kwargs
        )
        return embeddings.tolist()

    def predict(self, texts: List[List[str]]) -> List[List[float]]:
        assert self.is_cross_encoder, "Model is not of the type CrossEncoder"
        predictions: Tensor = self.model.predict(texts, **self.encode_kwargs)  # type: ignore
        return predictions.tolist()


@dataclass
class InfinityEmbeddings(BaseRagasEmbeddings):
    """Infinity embeddings using infinity_emb package.

    usage:
        ```python
        embedding_engine = InfinityEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        async with embedding_engine:
            embeddings = await embedding_engine.aembed_documents(
                ["Paris is in France", "The capital of France is Paris", "Infintiy batches embeddings on the fly"]
            )

        reranking_engine = InfinityEmbeddings(model_name="BAAI/bge-reranker-base")
        async with reranking_engine:
            rankings = await reranking_engine.arerank("Where is Paris?", ["Paris is in France", "I don't know the capital of Paris.", "Dummy sentence"])
        ```
    """

    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    infinity_engine_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    """infinity engine keyword arguments.
        {
            batch_size: int = 64
            revision: str | None = None,
            trust_remote_code: bool = True, 
            engine: str = torch | optimum | ctranslate2
            model_warmup: bool = False
            vector_disk_cache_path: str = ""
            device: Device | str = "auto"
            lengths_via_tokenize: bool = False
        }
    """

    def __post_init__(self):
        try:
            import infinity_emb
        except ImportError as exc:
            raise ImportError(
                "Could not import infinity_emb python package. "
                "Please install it with `pip install infinity-emb[torch,optimum]>=0.0.32`."
            ) from exc
        self.engine = infinity_emb.AsyncEmbeddingEngine(
            model_name_or_path=self.model_name, **self.infinity_engine_kwargs
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Infinity embeddings does not support sync embeddings"
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> t.List[t.List[float]]:
        """vectorize documents using an embedding model and return embeddings"""
        await self.__aenter__()
        if "embed" not in self.engine.capabilities:
            raise ValueError(
                f"Model={self.model_name} does not have `embed` capability, but only {self.engine.capabilities}. "
                "Try a different model, e.g. `model_name=BAAI/bge-small-en-v1.5`"
            )
        # return embeddings
        embeddings, _ = await self.engine.embed(sentences=texts)
        return np.array(embeddings).tolist()

    async def aembed_query(self, text: str) -> t.List[float]:
        """vectorize a query using an embedding model and return embeddings"""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    async def arerank(self, query: str, docs: List[str]) -> List[float]:
        """rerank documents against a single query and return scores for each document"""
        await self.__aenter__()
        if "rerank" not in self.engine.capabilities:
            raise ValueError(
                f"Model={self.model_name} does not have `rerank` capability, but only {self.engine.capabilities}. "
                "Try a different model, e.g. `model_name=mixedbread-ai/mxbai-rerank-base-v1`"
            )
        # return predictions
        rankings, _ = await self.engine.rerank(query=query, docs=docs)
        return rankings

    async def __aenter__(self, *args, **kwargs):
        if not self.engine.running:
            await self.engine.astart()

    async def __aexit__(self, *args, **kwargs):
        if self.engine.running:
            await self.engine.astop()

    def __del__(self, *args, **kwargs):
        if self.engine.running:
            if not hasattr(self.engine, "stop"):
                raise AttributeError("Engine does not have a stop method")
            self.engine.stop()


def embedding_factory(run_config: t.Optional[RunConfig] = None) -> BaseRagasEmbeddings:
    openai_embeddings = OpenAIEmbeddings()
    if run_config is not None:
        openai_embeddings.request_timeout = run_config.timeout
    else:
        run_config = RunConfig()
    return LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)
