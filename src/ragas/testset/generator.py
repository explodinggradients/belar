from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass
from random import choices, sample

import pandas as pd
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from ragas._analytics import TestsetGenerationEvent, track
from ragas.embeddings.base import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper, LLMConfig
from ragas.run_config import RunConfig
from ragas.testset.docstore import Document, DocumentStore, InMemoryDocumentStore
from ragas.testset.evolutions import (
    ComplexEvolution,
    CurrentNodes,
    DataRow,
    Evolution,
    multi_context,
    reasoning,
    simple,
)
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter
from ragas.utils import check_if_sum_is_close, deprecated, get_feature_language, is_nan


if t.TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument
    from llama_index.core.schema import Document as LlamaindexDocument

logger = logging.getLogger(__name__)

Distributions = t.Dict[t.Any, float]
DEFAULT_DISTRIBUTION = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def _to_records(self) -> t.List[t.Dict]:
        data_samples = []
        for data in self.test_data:
            data_dict = dict(data)
            data_dict["episode_done"] = True
            data_samples.append(data_dict)
        return data_samples

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._to_records())

    def to_dataset(self) -> Dataset:
        return Dataset.from_list(self._to_records())


@dataclass
class TestsetGenerator:
    generator_llm: BaseRagasLLM
    critic_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings
    docstore: DocumentStore

    @classmethod
    def from_langchain(
        cls,
        generator_llm: BaseLanguageModel,
        critic_llm: BaseLanguageModel,
        embeddings: Embeddings,
        docstore: t.Optional[DocumentStore] = None,
        run_config: t.Optional[RunConfig] = None,
        chunk_size: int = 1024,
        generator_llm_config: t.Optional[LLMConfig] = None,
        critic_llm_config: t.Optional[LLMConfig] = None,
    ) -> "TestsetGenerator":
        generator_llm_model = LangchainLLMWrapper(generator_llm, llm_config=generator_llm_config)
        critic_llm_model = LangchainLLMWrapper(critic_llm, llm_config=critic_llm_config)
        embeddings_model = LangchainEmbeddingsWrapper(embeddings)

        keyphrase_extractor = KeyphraseExtractor(llm=generator_llm_model)
        if docstore is None:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            docstore = InMemoryDocumentStore(
                splitter=splitter,
                embeddings=embeddings_model,
                extractor=keyphrase_extractor,
                run_config=run_config,
            )
        return cls(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
            docstore=docstore,
        )

    @classmethod
    @deprecated("0.1.4", removal="0.2.0", alternative="from_langchain")
    def with_openai(
        cls,
        generator_llm: str = "gpt-3.5-turbo-16k",
        critic_llm: str = "gpt-4",
        embeddings: str = "text-embedding-ada-002",
        docstore: t.Optional[DocumentStore] = None,
        chunk_size: int = 1024,
    ) -> "TestsetGenerator":
        generator_llm_model = ChatOpenAI(model=generator_llm)
        critic_llm_model = ChatOpenAI(model=critic_llm)
        embeddings_model = OpenAIEmbeddings(model=embeddings)

        return cls.from_langchain(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
            docstore=docstore,
            chunk_size=chunk_size,
        )

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaindexDocument],
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        distributions = distributions or {}
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
            is_async=is_async,
            run_config=run_config,
            raise_exceptions=raise_exceptions,
        )

    # if you add any arguments to this function, make sure to add them to
    # generate_with_langchain_docs as well
    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        distributions = distributions or {}
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_langchain_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
            is_async=is_async,
            raise_exceptions=raise_exceptions,
            run_config=run_config,
        )

    def init_evolution(self, evolution: Evolution) -> None:
        if evolution.generator_llm is None:
            evolution.generator_llm = self.generator_llm
            if evolution.docstore is None:
                evolution.docstore = self.docstore

            if evolution.question_filter is None:
                evolution.question_filter = QuestionFilter(llm=self.critic_llm)
            if evolution.node_filter is None:
                evolution.node_filter = NodeFilter(llm=self.critic_llm)

            if isinstance(evolution, ComplexEvolution):
                if evolution.evolution_filter is None:
                    evolution.evolution_filter = EvolutionFilter(llm=self.critic_llm)

    def generate(
        self,
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        distributions = distributions or DEFAULT_DISTRIBUTION
        # validate distributions
        if not check_if_sum_is_close(list(distributions.values()), 1.0, 3):
            raise ValueError(
                f"distributions passed do not sum to 1.0 [got {sum(list(distributions.values()))}]. Please check the "
                f"distributions."
            )

        # configure run_config for docstore
        if run_config is None:
            run_config = RunConfig(max_retries=15, max_wait=90)
        self.docstore.set_run_config(run_config)

        # init filters and evolutions
        for evolution in distributions:
            self.init_evolution(evolution)
            evolution.init(is_async=is_async, run_config=run_config)

        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.testset.evolutions", logging.DEBUG)
            patch_logger("ragas.testset.extractor", logging.DEBUG)
            patch_logger("ragas.testset.filters", logging.DEBUG)
            patch_logger("ragas.testset.docstore", logging.DEBUG)
            patch_logger("ragas.llms.prompt", logging.DEBUG)

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=raise_exceptions,
            run_config=run_config,
        )

        current_nodes = [
            CurrentNodes(root_node=n, nodes=[n])
            for n in self.docstore.get_random_nodes(k=test_size)
        ]
        total_evolutions = 0
        for evolution, probability in distributions.items():
            for i in sample(range(test_size), round(probability * test_size)):
                exec.submit(
                    evolution.evolve,
                    current_nodes[i],
                    name=f"{evolution.__class__.__name__}-{i}",
                )
                total_evolutions += 1
        if total_evolutions <= test_size:
            filler_evolutions = choices(
                list(distributions), k=test_size - total_evolutions
            )
            for evolution in filler_evolutions:
                exec.submit(
                    evolution.evolve,
                    current_nodes[total_evolutions],
                    name=f"{evolution.__class__.__name__}-{total_evolutions}",
                )
                total_evolutions += 1

        try:
            test_data_rows = exec.results()
            if not test_data_rows:
                raise ExceptionInRunner()

        except ValueError as e:
            raise e
        # make sure to ignore any NaNs that might have been returned
        # due to failed evolutions. MaxRetriesExceeded is a common reason
        test_data_rows = [r for r in test_data_rows if not is_nan(r)]
        test_dataset = TestDataset(test_data=test_data_rows)
        evol_lang = [get_feature_language(e) for e in distributions]
        evol_lang = [e for e in evol_lang if e is not None]
        track(
            TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[e.__class__.__name__.lower() for e in distributions],
                evolution_percentages=[distributions[e] for e in distributions],
                num_rows=len(test_dataset.test_data),
                language=evol_lang[0] if len(evol_lang) > 0 else "",
            )
        )

        return test_dataset

    def adapt(
        self,
        language: str,
        evolutions: t.List[Evolution],
        cache_dir: t.Optional[str] = None,
    ) -> None:
        assert isinstance(
            self.docstore, InMemoryDocumentStore
        ), "Must be an instance of in-memory docstore"
        assert self.docstore.extractor is not None, "Extractor is not set"

        self.docstore.extractor.adapt(language, cache_dir=cache_dir)
        for evolution in evolutions:
            self.init_evolution(evolution)
            evolution.init()
            evolution.adapt(language, cache_dir=cache_dir)

    def save(
        self, evolutions: t.List[Evolution], cache_dir: t.Optional[str] = None
    ) -> None:
        """
        Save the docstore prompts to a path.
        """
        assert isinstance(
            self.docstore, InMemoryDocumentStore
        ), "Must be an instance of in-memory docstore"
        assert self.docstore.extractor is not None, "Extractor is not set"

        self.docstore.extractor.save(cache_dir)
        for evolution in evolutions:
            assert evolution.node_filter is not None, "NodeFilter is not set"
            assert evolution.question_filter is not None, "QuestionFilter is not set"
            if isinstance(evolution, ComplexEvolution):
                assert (
                    evolution.evolution_filter is not None
                ), "EvolutionFilter is not set"
            evolution.save(cache_dir=cache_dir)
