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
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI

from ragas.run_config import RunConfig, add_async_retry, add_retry
import re
import hashlib
import traceback


if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

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
        return 0.3 if n > 1 else 1e-8

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
        is_async: bool = True,
    ) -> LLMResult:
        # traceback.print_stack()
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

@dataclass
class LLMConfig:
    stop: t.Optional[t.List[str]] = None
    params: t.Optional[t.Dict[str, t.Any]] = None
    prompt_callback: t.Optional[t.Callable[[PromptValue], t.Tuple[t.List[PromptValue], t.Dict[str, t.Any]]]] = None
    result_callback: t.Optional[t.Callable[[LLMResult], t.Tuple[t.List[LLMResult]]]] = None

    def __init__(self, stop: t.Optional[t.List[str]] = None, prompt_callback: t.Optional[t.Callable[[PromptValue], t.Tuple[t.List[PromptValue], t.Dict[str, t.Any]]]] = None, **kwargs):
        self.stop = stop
        self.params = kwargs
        self.prompt_callback = prompt_callback

class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self,
        langchain_llm: BaseLanguageModel,
        run_config: t.Optional[RunConfig] = None,
        llm_config: LLMConfig = None,
    ):
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm_config = llm_config

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        stop = stop or self.llm_config.stop

        if self.llm_config.prompt_callback:
            prompts, extra_params = self.llm_config.prompt_callback(prompt)
        else:
            prompts = [prompt]
            extra_params = {}

        if is_multiple_completion_supported(self.langchain_llm):
            result = self.langchain_llm.generate_prompt(
                prompts=prompts,
                n=n,
                temperature=temperature,
                callbacks=callbacks,
                stop=stop,
                **self.llm_config.params,
                **extra_params,
            )
            if self.llm_config.result_callback:
                return self.llm_config.result_callback(result)
            return result
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
                **self.llm_config.params,
                **extra_params,
            )
            if self.llm_config.result_callback:
                result = self.llm_config.result_callback(result)
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # to trace request/response for multi-threaded execution
        gen_id = hashlib.md5(str(prompt).encode('utf-8')).hexdigest()[:4]
        stop = stop or self.llm_config.stop
        prompt_str = prompt.prompt_str
        logger.debug(f"Generating text for [{gen_id}] with prompt: {prompt_str}")
        temperature = self.get_temperature(n=n)
        if self.llm_config.prompt_callback:
            prompts, extra_params = self.llm_config.prompt_callback(prompt)
        else:
            prompts = [prompt] * n
            extra_params = {}
        if is_multiple_completion_supported(self.langchain_llm):
            result = await self.langchain_llm.agenerate_prompt(
                prompts=prompts,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
                **self.llm_config.params,
                **extra_params,
            )
            if self.llm_config.result_callback:
                result = self.llm_config.result_callback(result)
            logger.debug(f"got result (m): {result.generations[0][0].text}")
            return result
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=prompts,
                temperature=temperature,
                callbacks=callbacks,
                **self.llm_config.params,
                **extra_params,
            )
            if self.llm_config.result_callback:
                result = self.llm_config.result_callback(result)
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

            # this part should go to LLMConfig.result_callback
            if len(result.generations[0][0].text) > 0:
                result.generations[0][0].text = re.sub(r"</?bot>", '', result.generations[0][0].text)
            logger.debug(f"got result [{gen_id}]: {result.generations[0][0].text}")
            # todo configure on question?
            if len(result.generations[0][0].text) < 24:
                logger.warning(f"truncated response?: {result.generations}")
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


def llm_factory(
    model: str = "gpt-3.5-turbo", run_config: t.Optional[RunConfig] = None
) -> BaseRagasLLM:
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout
    openai_model = ChatOpenAI(model=model, timeout=timeout)
    return LangchainLLMWrapper(openai_model, run_config)
