from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI

from ragas.run_config import RunConfig, add_async_retry, add_retry

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from llama_index.core.base.llms.base import BaseLLM

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)

MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 0.01

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
        is_async: bool = True,
    ) -> LLMResult:
        """Generate text using the given event loop."""
        if is_async:
            agenerate_text_with_retry = add_async_retry(
                self.agenerate_text, self.run_config
            )
            return await agenerate_text_with_retry(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            loop = asyncio.get_event_loop()
            generate_text_with_retry = add_retry(self.generate_text, self.run_config)
            generate_text = partial(
                generate_text_with_retry,
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            return await loop.run_in_executor(None, generate_text)


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self, langchain_llm: BaseLanguageModel, run_config: t.Optional[RunConfig] = None
    ):
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # configure if using OpenAI API
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.langchain_llm.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
    ):
        self.llm = llm

        self._signature = ""
        if type(self.llm).__name__.lower() == "bedrock":
            self._signature = "bedrock"
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def check_args(
        self,
        n: int,
        temperature: float,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
    ) -> dict[str, t.Any]:
        if n != 1:
            logger.warning("n values greater than 1 not support for LlamaIndex LLMs")
        if temperature != 1e-8:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature == "bedrock":
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = self.llm.complete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])


def llm_factory(
    model: str = "gpt-3.5-turbo", run_config: t.Optional[RunConfig] = None
) -> BaseRagasLLM:
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout
    openai_model = ChatOpenAI(model=model, timeout=timeout)
    return LangchainLLMWrapper(openai_model, run_config)
