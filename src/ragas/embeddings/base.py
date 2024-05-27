from __future__ import annotations

import asyncio
import typing as t
from abc import ABC
from dataclasses import field
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.dataclasses import dataclass

from ragas.run_config import RunConfig, add_async_retry, add_retry
import logging

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class BaseRagasEmbeddings(Embeddings, ABC):
    run_config: RunConfig

    async def embed_text(self, text: str, is_async=True) -> List[float]:
        logging.debug(f"Embedding single text: {text[0:6]}")
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(
        self, texts: List[str], is_async: bool = True
    ) -> t.List[t.List[float]]:
        logging.debug(f"Starting embedding for texts")
        if is_async:
            aembed_documents_with_retry = add_async_retry(
                self.aembed_documents, self.run_config
            )
            logging.debug(f"Async embedding result")
            return await aembed_documents_with_retry(texts)
        else:
            loop = asyncio.get_event_loop()
            embed_documents_with_retry = add_retry(
                self.embed_documents, self.run_config
            )
            return await loop.run_in_executor(None, embed_documents_with_retry, texts)
            logging.debug(f"Sync embedding result")

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config


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

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
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
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
        else:
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )

        # ensure outputs are tensors
        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(
            self.model, SentenceTransformer
        ), "Model is not of the type Bi-encoder"
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: List[List[str]]) -> List[List[float]]:
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch import Tensor

        assert isinstance(
            self.model, CrossEncoder
        ), "Model is not of the type CrossEncoder"

        predictions = self.model.predict(texts, **self.encode_kwargs)

        assert isinstance(predictions, Tensor)
        return predictions.tolist()


def embedding_factory(
    model: str = "text-embedding-ada-002", run_config: t.Optional[RunConfig] = None
) -> BaseRagasEmbeddings:
    openai_embeddings = OpenAIEmbeddings(model=model)
    if run_config is not None:
        openai_embeddings.request_timeout = run_config.timeout
    else:
        run_config = RunConfig()
    return LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)
