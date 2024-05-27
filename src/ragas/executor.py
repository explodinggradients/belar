from __future__ import annotations

import asyncio
import logging
import sys
import threading
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.exceptions import MaxRetriesExceeded
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


def runner_exception_hook(args: threading.ExceptHookArgs):
    print(args)
    raise args.exc_type


# set a custom exception hook
# threading.excepthook = runner_exception_hook


def as_completed(loop, coros, max_workers):
    loop_arg_dict = {} # {"loop": loop} if sys.version_info[:2] < (3, 10) else {}
    if max_workers == -1:
        return asyncio.as_completed(coros, **loop_arg_dict)

    # loop argument is removed since Python 3.10
    semaphore = asyncio.Semaphore(max_workers)

    async def sema_coro(coro):
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                logger.error(f"Error executing task: {e}")
                raise  # Ensure exceptions are not swallowed silently

    sema_coros = [sema_coro(c) for c in coros]
    return asyncio.as_completed(sema_coros, **loop_arg_dict)


class Runner(threading.Thread):
    def __init__(
        self,
        jobs: t.List[t.Tuple[t.Coroutine, str]],
        desc: str,
        keep_progress_bar: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        super().__init__()
        self.jobs = jobs
        self.desc = desc
        self.keep_progress_bar = keep_progress_bar
        self.raise_exceptions = raise_exceptions
        self.run_config = run_config or RunConfig()

        # create task
        self.loop = asyncio.new_event_loop()
        self.futures = as_completed(
            loop=self.loop,
            coros=[coro for coro, _ in self.jobs],
            max_workers=self.run_config.max_workers,
        )

    async def _aresults(self) -> t.List[t.Any]:
        results = []
        for future in tqdm(
            self.futures,
            desc=self.desc,
            total=len(self.jobs),
            # whether you want to keep the progress bar after completion
            leave=self.keep_progress_bar,
        ):
            r = (-1, np.nan)
            try:
                r = await future
            except MaxRetriesExceeded as e:
                logger.warning(f"max retries exceeded for {e.evolution}")
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    logger.error(
                        "Runner in Executor raised an exception", exc_info=True
                    )
            results.append(r)

        return results

    def run(self):
        results = []
        try:
            results = self.loop.run_until_complete(self._aresults())
        finally:
            self.results = results
            self.loop.stop()


@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = field(default_factory=RunConfig, repr=False)

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        return wrapped_callable_async

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index(*args, **kwargs), name))

    def results(self) -> t.List[t.Any]:
        executor_job = Runner(
            jobs=self.jobs,
            desc=self.desc,
            keep_progress_bar=self.keep_progress_bar,
            raise_exceptions=self.raise_exceptions,
            run_config=self.run_config,
        )
        executor_job.start()
        try:
            executor_job.join()
        finally:
            ...

        if executor_job.results is None:
            if self.raise_exceptions:
                raise RuntimeError(
                    "Executor failed to complete. Please check logs above for full info."
                )
            else:
                logger.error("Executor failed to complete. Please check logs above.")
                return []
        sorted_results = sorted(executor_job.results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
