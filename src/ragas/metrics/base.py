"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import floor

from datasets import Dataset
from langchain.base_language import BaseLanguageModel


def make_batches(total_size: int, batch_size: int) -> list[range]:
    """
    Take a total size and batch size and return a list of ranges for the batches
    """
    tail = total_size % batch_size
    num_batches = floor(total_size / batch_size)
    batches = [
        range(i, i + batch_size) for i in range(0, batch_size * num_batches, batch_size)
    ]
    if tail != 0:
        batches.append(range(batch_size * num_batches, batch_size * num_batches + tail))

    return batches


@dataclass
class Metric(ABC):
    batch_size: int
    llm: t.Optional[BaseLanguageModel] = None

    def __post_init__(self: t.Self):
        if self.llm is None:
            from langchain.chat_models import ChatOpenAI

            self.llm = ChatOpenAI()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def init_model():
        """
        This method will lazy initialize the model.
        """
        ...

    @abstractmethod
    def score(self: t.Self, dataset: Dataset) -> Dataset:
        ...

    def get_batches(self, dataset_size: int) -> list[range]:
        return make_batches(dataset_size, self.batch_size)
