from dataclasses import dataclass, field

from ragas.experimental.prompt import StringIO

from .base import BaseSimulator, Scenario
from .prompts import (
    CriticUserInput,
    GenerateReference,
    ModifyUserInput,
    PydanticPrompt,
    UserInputAndContext,
    UserInputWithStyleAndLength,
    extend_modify_input_prompt,
)


@dataclass
class QASimulator(BaseSimulator[Scenario]):
    critic_user_input_prompt: PydanticPrompt = field(default_factory=CriticUserInput)
    user_input_modification_prompt: PydanticPrompt = field(
        default_factory=ModifyUserInput
    )
    generate_reference_prompt: PydanticPrompt = field(default_factory=GenerateReference)

    async def critic_question(self, question: str) -> bool:
        critic = await self.critic_user_input_prompt.generate(
            data=StringIO(text=question), llm=self.llm
        )
        return critic.independence > 1 and critic.clear_intent > 1

    async def modify_question(self, question: str, scenario: Scenario) -> str:
        prompt = extend_modify_input_prompt(
            question_modification_prompt=self.user_input_modification_prompt,
            style=scenario.style,
            length=scenario.length,
        )
        modified_question = await prompt.generate(
            data=UserInputWithStyleAndLength(
                user_input=question,
                style=scenario.style,
                length=scenario.length,
            ),
            llm=self.llm,
        )
        return modified_question.text

    async def generate_answer(
        self,
        question: str,
        scenario: Scenario,
        reference_property_name: str = "page_content",
    ) -> str:
        reference = await self.generate_reference_prompt.generate(
            data=UserInputAndContext(
                user_input=question,
                context=self.make_source_text(scenario, reference_property_name),
            ),
            llm=self.llm,
        )
        return reference.text

    @staticmethod
    def make_source_text(
        scenario: Scenario, property_name: str = "page_content"
    ) -> str:
        page_contents = []
        for node in scenario.nodes:
            page_contents.append(node.get_property(property_name))
        return "\n\n".join(page_contents)
