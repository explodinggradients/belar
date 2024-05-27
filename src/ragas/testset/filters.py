from __future__ import annotations

import logging
import typing as t
from abc import ABC
from dataclasses import dataclass, field

from ragas.run_config import RunConfig
from ragas.testset.prompts import (
    context_scoring_parser,
    context_scoring_prompt,
    evolution_elimination_parser,
    evolution_elimination_prompt,
    filter_question_prompt,
    question_filter_parser,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.llms.prompt import Prompt
    from ragas.testset.docstore import Node


logger = logging.getLogger(__name__)


@dataclass
class Filter(ABC):
    llm: BaseRagasLLM

    def set_run_config(self, run_config: RunConfig):
        self.llm.set_run_config(run_config)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        raise NotImplementedError("adapt() is not implemented for {} Filter")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        raise NotImplementedError("save() is not implemented for {} Filter")


@dataclass
class NodeFilter(Filter):
    threshold: float = 1.5
    context_scoring_prompt: Prompt = field(
        default_factory=lambda: context_scoring_prompt
    )

    async def filter(self, node: Node) -> t.Dict:
        prompt = self.context_scoring_prompt.format(context=node.page_content)
        results = await self.llm.generate(prompt=prompt)
        output = results.generations[0][0].text.strip()
        output = await context_scoring_parser.aparse(output, prompt, self.llm)
        output = output.dict() if output is not None else {}
        if len(output.values()) == 0:
            logger.warning("output.values() is empty")
            output["score"] = 0
        else:
            output["score"] = sum(output.values()) / len(output.values())
        logger.debug("context scoring: %s", output)
        output.update({"score": output.get("score", 0) >= self.threshold})
        return output

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.context_scoring_prompt = self.context_scoring_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.context_scoring_prompt.save(cache_dir)


@dataclass
class QuestionFilter(Filter):
    llm: BaseRagasLLM
    filter_question_prompt: Prompt = field(
        default_factory=lambda: filter_question_prompt
    )

    async def filter(self, question: str) -> t.Tuple[bool, str]:
        prompt = self.filter_question_prompt.format(question=question)
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        results = await question_filter_parser.aparse(results, prompt, self.llm)
        results = results.dict() if results is not None else {}
        logger.debug("filtered question: %s", results)
        return results.get("verdict") == 1, results.get("feedback", "")

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.filter_question_prompt = self.filter_question_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.filter_question_prompt.save(cache_dir)


@dataclass
class EvolutionFilter(Filter):
    llm: BaseRagasLLM
    evolution_elimination_prompt: Prompt = field(
        default_factory=lambda: evolution_elimination_prompt
    )

    async def filter(self, simple_question: str, compressed_question: str) -> bool:
        prompt = self.evolution_elimination_prompt.format(
            question1=simple_question, question2=compressed_question
        )
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        results = await evolution_elimination_parser.aparse(results, prompt, self.llm)
        results = results.dict() if results is not None else {}
        logger.debug("evolution filter: %s", results)
        return results.get("verdict") == 1

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.evolution_elimination_prompt = self.evolution_elimination_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.evolution_elimination_prompt.save(cache_dir)
