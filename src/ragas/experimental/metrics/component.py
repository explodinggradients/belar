from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from transformers import AutoTokenizer, Pipeline, pipeline

from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="The context and statement are unrelated",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


class BaseNLIComponent(ABC):
    @abstractmethod
    async def apply(
        self, hypothesis: t.List[str], premise: str, callbacks: Callbacks
    ) -> t.List[bool]:
        """
        Apply the NLI component to a list of premises and a hypothesis.
        """
        raise NotImplementedError("apply method must be implemented by subclasses")


@dataclass
class LLMNLIComponent(BaseNLIComponent):
    llm: BaseRagasLLM
    nli_prompt: PydanticPrompt = NLIStatementPrompt()

    async def apply(
        self, hypothesis: t.List[str], premise: str, callbacks: Callbacks
    ) -> t.List[bool]:
        assert self.llm is not None, "LLM must be set"
        prompt_input = NLIStatementInput(context=premise, statements=hypothesis)
        response = await self.nli_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return [bool(result.verdict) for result in response.statements]


@dataclass
class TextClassificationNLIComponent(BaseNLIComponent):
    hf_pipeline: Pipeline
    prompt: str
    label: str
    batch_size: int = 32
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        if "{premise}" not in self.prompt or "{hypothesis}" not in self.prompt:
            raise ValueError("Prompt should not contain 'premise' or 'hypothesis'")

        self.model_kwargs["top_k"] = 1

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        prompt: str,
        label: str,
        model_kwargs: t.Dict[str, t.Any] = {},
        pipeline_kwargs: t.Dict[str, t.Any] = {},
    ) -> TextClassificationNLIComponent:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_pipeline = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=tokenizer,
            **pipeline_kwargs,
        )

        return cls(
            hf_pipeline=hf_pipeline,
            prompt=prompt,
            label=label,
            model_kwargs=model_kwargs,
        )

    async def apply(
        self, hypothesis: t.List[str], premise: str, callbacks: Callbacks = None
    ) -> t.List[bool]:
        scores = []
        prompt_input_list = [
            self.prompt.format(hypothesis=text, premise=premise) for text in hypothesis
        ]
        for i in range(0, len(prompt_input_list), self.batch_size):
            prompt_input_list_batch = prompt_input_list[i : i + self.batch_size]
            response = self.hf_pipeline(prompt_input_list_batch, **self.model_kwargs)
            assert isinstance(response, list), "Response should be a list"
            assert all(
                isinstance(item, dict) for item in response
            ), "Items in response should be dictionaries"
            response = [item[0].get("label") == self.label for item in response]  # type: ignore
            scores.extend(response)

        return scores
