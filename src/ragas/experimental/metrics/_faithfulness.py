from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.experimental.llms.prompt import PydanticPrompt
from ragas.experimental.metrics.component import BaseNLIComponent, LLMNLIComponent
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._faithfulness import HasSegmentMethod


logger = logging.getLogger(__name__)


class FaithfulnessStatements(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")
    sentences: t.Dict[int, str] = Field(
        description="A mapping of sentence index to the sentence"
    )


class SentenceComponents(BaseModel):
    sentence_index: int = Field(description="The index of the sentence")
    simpler_statements: t.List[str] = Field(
        description="A list of simpler statements that can be directly inferred from the context"
    )


class SentencesSimplified(BaseModel):
    sentences: t.List[SentenceComponents] = Field(
        description="A list of sentences and their simpler versions"
    )


# examples
example_input_1 = FaithfulnessStatements(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentences={
        0: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",
        1: "He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    },
)

example_output_1 = SentencesSimplified(
    sentences=[
        SentenceComponents(
            sentence_index=0,
            simpler_statements=[
                "Albert Einstein was a German-born theoretical physicist.",
                "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
            ],
        ),
        SentenceComponents(
            sentence_index=1,
            simpler_statements=[
                "Albert Einstein was best known for developing the theory of relativity.",
                "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
            ],
        ),
    ]
)


class LongFormAnswerPrompt(PydanticPrompt[FaithfulnessStatements, SentencesSimplified]):
    instruction = "Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON."
    input_model = FaithfulnessStatements
    output_model = SentencesSimplified
    examples = [(example_input_1, example_output_1)]


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


@dataclass
class FaithfulnessExperimental(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness_experimental"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    nli_component: t.Optional[BaseNLIComponent] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self):
        self.long_form_answer_prompt = LongFormAnswerPrompt()
        self.nli_component = LLMNLIComponent(llm=self.llm)  # type: ignore
        if self.sentence_segmenter is None:
            # TODO: make this dynamic, taking language from prompt
            language = "english"
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"
        assert self.nli_component is not None, "NLI component is not set"

        answer, question, contexts = (
            row["response"],
            row["user_input"],
            row["retrieved_contexts"],
        )
        contexts = "\n".join(contexts)
        # get the sentences from the answer
        if self.sentence_segmenter is None:
            raise ValueError("Sentence segmenter is not set")
        sentences = self.sentence_segmenter.segment(answer)
        # TODO: why do we do this?
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentence_components = await self.long_form_answer_prompt.generate(
            data=FaithfulnessStatements(
                question=question,
                answer=answer,
                sentences={i: sentence for i, sentence in enumerate(sentences)},
            ),
            llm=self.llm,
            callbacks=callbacks,
        )

        statements = [
            statement
            for component in sentence_components.sentences
            for statement in component.simpler_statements
        ]
        verdicts = await self.nli_component.apply(
            hypothesis=statements, premise=contexts, callbacks=callbacks
        )

        # compute the score
        num_faithful_statements = sum(verdicts)
        if len(statements):
            score = num_faithful_statements / len(statements)
        else:
            score = np.nan
        return score

    async def _single_turn_ascore(
        self: t.Self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.dict()
        return await self._ascore(row, callbacks)
