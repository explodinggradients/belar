from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from pysbd import Segmenter

from ragas.callbacks import ChainType, new_group
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.executor import is_event_loop_running
from ragas.prompt import PromptMixin
from ragas.run_config import RunConfig
from ragas.utils import RAGAS_SUPPORTED_LANGUAGE_CODES, deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM
logger = logging.getLogger(__name__)


VALID_COLUMNS = [
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "response",
    "reference",
    "rubric",
]


class MetricType(Enum):
    """
    Enumeration of metric types in Ragas.

    Attributes
    ----------
    SINGLE_TURN : str
        Represents a single-turn metric type.
    MULTI_TURN : str
        Represents a multi-turn metric type.
    """

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


@dataclass
class Metric(ABC):
    """
    Abstract base class for metrics in Ragas.

    Attributes
    ----------
    name : str
        The name of the metric.
    required_columns : Dict[str, Set[str]]
        A dictionary mapping metric type names to sets of required column names. This is
        a property and raises `ValueError` if columns are not in `VALID_COLUMNS`.
    """

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=dict, repr=False, init=False
    )

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def required_columns(self) -> t.Dict[str, t.Set[str]]:
        return {k.name: v for k, v in self._required_columns.items()}

    @required_columns.setter
    def required_columns(self, metric_type: MetricType, columns: t.Set[str]):
        for column in columns:
            if column not in VALID_COLUMNS:
                raise ValueError(
                    f"Invalid column '{column}'. Must be one of {VALID_COLUMNS}"
                )
        self._required_columns[metric_type] = columns

    @abstractmethod
    def init(self, run_config: RunConfig): ...

    @deprecated("0.2", removal="0.3", alternative="single_turn_ascore")
    def score(self, row: t.Dict, callbacks: Callbacks = None) -> float:
        """
        Calculates the score for a single row of data.

        Note
        ----
        This method is deprecated and will be removed in 0.3. Please use `single_turn_ascore` or `multi_turn_ascore` instead.
        """
        callbacks = callbacks or []
        rm, group_cm = new_group(
            self.name,
            inputs=row,
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            if is_event_loop_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    raise ImportError(
                        "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                    )
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(self._ascore(row=row, callbacks=group_cm))
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @deprecated("0.2", removal="0.3", alternative="single_turn_ascore")
    async def ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        """
        Asynchronously calculates the score for a single row of data.

        Note
        ----
        This method is deprecated and will be removed in 0.3. Please use `single_turn_ascore` instead.
        """
        callbacks = callbacks or []
        rm, group_cm = new_group(
            self.name,
            inputs=row,
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            score = await asyncio.wait_for(
                self._ascore(row=row, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float: ...


@dataclass
class MetricWithLLM(Metric, PromptMixin):
    """
    A metric class that uses a language model for evaluation.

    Attributes
    ----------
    llm : Optional[BaseRagasLLM]
        The language model used for the metric.
    """

    llm: BaseRagasLLM = field(kw_only=True)

    def init(self, run_config: RunConfig):
        self.llm.set_run_config(run_config)


@dataclass
class MetricWithEmbeddings(Metric):
    embeddings: BaseRagasEmbeddings = field(kw_only=True)

    def init(self, run_config: RunConfig):
        self.embeddings.set_run_config(run_config)


class SingleTurnMetric(Metric):
    """
    A metric class for evaluating single-turn interactions.

    This class provides methods to score single-turn samples, both synchronously and asynchronously.
    """

    def _only_required_columns_single_turn(
        self, sample: SingleTurnSample
    ) -> SingleTurnSample:
        """
        Simplify the sample to only include the required columns.
        """
        required_columns = self.required_columns.get(MetricType.SINGLE_TURN.name, set())
        if not required_columns:
            return sample
        return SingleTurnSample(**sample.model_dump(include=required_columns))

    def single_turn_score(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        """
        Synchronously score a single-turn sample.

        May raise ImportError if nest_asyncio is not installed in a Jupyter-like environment.
        """
        callbacks = callbacks or []
        # only get the required columns
        sample = self._only_required_columns_single_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            if is_event_loop_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    raise ImportError(
                        "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                    )
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(
                self._single_turn_ascore(sample=sample, callbacks=group_cm)
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    async def single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        """
        Asynchronously score a single-turn sample with an optional timeout.

        May raise asyncio.TimeoutError if the scoring process exceeds the specified timeout.
        """
        callbacks = callbacks or []
        # only get the required columns
        sample = self._only_required_columns_single_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            score = await asyncio.wait_for(
                self._single_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks,
    ) -> float:
        """
        Abstract method to be implemented by subclasses for actual scoring logic.
        """
        ...


class MultiTurnMetric(Metric):
    """
    A metric class for evaluating multi-turn conversations.

    This class extends the base Metric class to provide functionality
    for scoring multi-turn conversation samples.
    """

    def _only_required_columns_multi_turn(
        self, sample: MultiTurnSample
    ) -> MultiTurnSample:
        """
        Simplify the sample to only include the required columns.
        """
        required_columns = self.required_columns.get(MetricType.MULTI_TURN.name, set())
        if not required_columns:
            return sample
        return MultiTurnSample(**sample.model_dump(include=required_columns))

    def multi_turn_score(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        """
        Score a multi-turn conversation sample synchronously.

        May raise ImportError if nest_asyncio is not installed in Jupyter-like environments.
        """
        callbacks = callbacks or []
        sample = self._only_required_columns_multi_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            if is_event_loop_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    raise ImportError(
                        "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                    )
            loop = asyncio.get_event_loop()
            score = loop.run_until_complete(
                self._multi_turn_ascore(sample=sample, callbacks=group_cm)
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    async def multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        """
        Score a multi-turn conversation sample asynchronously.

        May raise asyncio.TimeoutError if the scoring process exceeds the specified timeout.
        """
        callbacks = callbacks or []
        sample = self._only_required_columns_multi_turn(sample)

        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            score = await asyncio.wait_for(
                self._multi_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})
        return score

    @abstractmethod
    async def _multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks,
    ) -> float:
        """
        Abstract method to be implemented by subclasses for actual multi-turn scoring logic.
        """
        ...


class Ensember:
    """
    Combine multiple llm outputs for same input (n>1) to a single output
    """

    def from_discrete(
        self, inputs: list[list[t.Dict]], attribute: str
    ) -> t.List[t.Dict]:
        """
        Simple majority voting for binary values, ie [0,0,1] -> 0
        inputs: list of list of dicts each containing verdict for a single input
        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not all(len(item) == len(inputs[0]) for item in inputs):
            logger.warning("All inputs must have the same length")
            return inputs[0]

        if not all(attribute in item for input in inputs for item in input):
            logger.warning(f"All inputs must have {attribute} attribute")
            return inputs[0]

        if len(inputs) == 1:
            return inputs[0]

        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            item[attribute] = list(verdict_counts.keys())[0]
            verdict_agg.append(item)

        return verdict_agg


def get_segmenter(
    language: str = "english", clean: bool = False, char_span: bool = False
):
    """
    Get a sentence segmenter for a given language
    """
    language = language.lower()
    if language not in RAGAS_SUPPORTED_LANGUAGE_CODES:
        raise ValueError(
            f"Language '{language}' not supported. Supported languages: {RAGAS_SUPPORTED_LANGUAGE_CODES.keys()}"
        )
    return Segmenter(
        language=RAGAS_SUPPORTED_LANGUAGE_CODES[language],
        clean=clean,
        char_span=char_span,
    )


def is_reproducable(metric: Metric) -> bool:
    """
    Check if a metric is reproducible by checking if it has a `_reproducibility` attribute.
    """
    return hasattr(metric, "_reproducibility")


ensembler = Ensember()
