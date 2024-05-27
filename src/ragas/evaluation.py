from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset, concatenate_datasets
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM

from ragas._analytics import EvaluationEvent, track
from ragas.callbacks import new_group
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper, LLMConfig
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    is_reproducable,
)
from ragas.metrics.critique import AspectCritique
from ragas.run_config import RunConfig
from ragas.utils import get_feature_language

# from ragas.metrics.critique import AspectCritique
from ragas.validation import (
    handle_deprecated_ground_truths,
    remap_column_names,
    validate_column_dtypes,
    validate_evaluation_modes,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


def evaluate(
    dataset: Dataset,
    metrics: list[Metric] | None = None,
    llm: t.Optional[BaseRagasLLM | LangchainLLM] = None,
    llm_config: t.Optional[LLMConfig] = None,
    embeddings: t.Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
    callbacks: Callbacks = None,
    in_ci: bool = False,
    is_async: bool = True,
    run_config: t.Optional[RunConfig] = None,
    raise_exceptions: bool = True,
    column_map: t.Optional[t.Dict[str, str]] = None,
) -> Result:
    """
    Run the evaluation on the dataset with different metrics

    Parameters
    ----------
    dataset : Dataset[question: list[str], contexts: list[list[str]], answer: list[str], ground_truth: list[list[str]]]
        The dataset in the format of ragas which the metrics will use to score the RAG
        pipeline with
    metrics : list[Metric] , optional
        List of metrics to use for evaluation. If not provided then ragas will run the
        evaluation on the best set of metrics to give a complete view.
    llm: BaseRagasLLM, optional
        The language model to use for the metrics. If not provided then ragas will use
        the default language model for metrics which require an LLM. This can we overridden by the llm specified in
        the metric level with `metric.llm`.
    embeddings: BaseRagasEmbeddings, optional
        The embeddings to use for the metrics. If not provided then ragas will use
        the default embeddings for metrics which require embeddings. This can we overridden by the embeddings specified in
        the metric level with `metric.embeddings`.
    callbacks: Callbacks, optional
        Lifecycle Langchain Callbacks to run during evaluation. Check the
        [langchain documentation](https://python.langchain.com/docs/modules/callbacks/)
        for more information.
    in_ci: bool
        Whether the evaluation is running in CI or not. If set to True then some
        metrics will be run to increase the reproducability of the evaluations. This
        will increase the runtime and cost of evaluations. Default is False.
    is_async: bool
        Whether to run the evaluation in async mode or not. If set to True then the
        evaluation is run by calling the `metric.ascore` method. In case the llm or
        embeddings does not support async then the evaluation can be run in sync mode
        with `is_async=False`. Default is False.
    run_config: RunConfig, optional
        Configuration for runtime settings like timeout and retries. If not provided,
        default values are used.
    raise_exceptions: True
        Whether to raise exceptions or not. If set to True then the evaluation will
        raise an exception if any of the metrics fail. If set to False then the
        evaluation will return `np.nan` for the row that failed. Default is True.
    column_map : dict[str, str], optional
        The column names of the dataset to use for evaluation. If the column names of
        the dataset are different from the default ones then you can provide the
        mapping as a dictionary here. Example: If the dataset column name is contexts_v1,
        column_map can be given as {"contexts":"contexts_v1"}

    Returns
    -------
    Result
        Result object containing the scores of each metric. You can use this do analysis
        later.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

    Examples
    --------
    the basic usage is as follows:
    ```
    from ragas import evaluate

    >>> dataset
    Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 30
    })

    >>> result = evaluate(dataset)
    >>> print(result)
    {'context_precision': 0.817,
    'faithfulness': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    column_map = column_map or {}
    callbacks = callbacks or []

    if dataset is None:
        raise ValueError("Provide dataset!")

    # default run_config
    run_config = run_config or RunConfig()
    # default metrics
    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    # remap column names from the dataset
    dataset = remap_column_names(dataset, column_map)
    # validation
    dataset = handle_deprecated_ground_truths(dataset)
    validate_evaluation_modes(dataset, metrics)
    validate_column_dtypes(dataset)

    # set the llm and embeddings
    if isinstance(llm, LangchainLLM):
        llm = LangchainLLMWrapper(llm, llm_config=llm_config, run_config=run_config)
    if isinstance(embeddings, LangchainEmbeddings):
        embeddings = LangchainEmbeddingsWrapper(embeddings)

    # init llms and embeddings
    binary_metrics = []
    llm_changed: t.List[int] = []
    embeddings_changed: t.List[int] = []
    reproducable_metrics: t.List[int] = []
    answer_correctness_is_set = -1

    # loop through the metrics and perform initializations
    for i, metric in enumerate(metrics):
        # set llm and embeddings if not set
        if isinstance(metric, AspectCritique):
            binary_metrics.append(metric.name)
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            if llm is None:
                llm = llm_factory()
            metric.llm = llm
            llm_changed.append(i)
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            if embeddings is None:
                embeddings = embedding_factory()
            metric.embeddings = embeddings
            embeddings_changed.append(i)
        if isinstance(metric, AnswerCorrectness):
            if metric.answer_similarity is None:
                answer_correctness_is_set = i
        # set reproducibility for metrics if in CI
        if in_ci and is_reproducable(metric):
            if metric.reproducibility == 1:  # type: ignore
                # only set a value if not already set
                metric.reproducibility = 3  # type: ignore
                reproducable_metrics.append(i)

        # init all the models
        metric.init(run_config)

    executor = Executor(
        desc="Evaluating",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
    )
    # new evaluation chain
    row_run_managers = []
    evaluation_rm, evaluation_group_cm = new_group(
        name="ragas evaluation", inputs={}, callbacks=callbacks, is_async=is_async
    )
    for i, row in enumerate(dataset):
        row = t.cast(t.Dict[str, t.Any], row)
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            is_async=is_async,
        )
        row_run_managers.append((row_rm, row_group_cm))
        [
            executor.submit(
                metric.ascore, row, row_group_cm, is_async, name=f"{metric.name}-{i}"
            )
            for metric in metrics
        ]

    scores = []
    try:
        # get the results
        results = executor.results()
        if results == []:
            raise ExceptionInRunner()

        # convert results to dataset_like
        for i, _ in enumerate(dataset):
            s = {}
            for j, m in enumerate(metrics):
                s[m.name] = results[len(metrics) * i + j]
            scores.append(s)
            # close the row chain
            row_rm, row_group_cm = row_run_managers[i]
            if not row_group_cm.ended:
                row_rm.on_chain_end(s)

    # run evaluation task
    except Exception as e:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)

        raise e
    else:
        result = Result(
            scores=Dataset.from_list(scores),
            dataset=dataset,
            binary_columns=binary_metrics,
        )
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end(result)
    finally:
        # reset llms and embeddings if changed
        for i in llm_changed:
            t.cast(MetricWithLLM, metrics[i]).llm = None
        for i in embeddings_changed:
            t.cast(MetricWithEmbeddings, metrics[i]).embeddings = None
        if answer_correctness_is_set != -1:
            t.cast(
                AnswerCorrectness, metrics[answer_correctness_is_set]
            ).answer_similarity = None

        for i in reproducable_metrics:
            metrics[i].reproducibility = 1  # type: ignore

    # log the evaluation event
    metrics_names = [m.name for m in metrics]
    metric_lang = [get_feature_language(m) for m in metrics]
    metric_lang = np.unique([m for m in metric_lang if m is not None])
    track(
        EvaluationEvent(
            event_type="evaluation",
            metrics=metrics_names,
            evaluation_mode="",
            num_rows=dataset.shape[0],
            language=metric_lang[0] if len(metric_lang) > 0 else "",
            in_ci=in_ci,
        )
    )
    return result


@dataclass
class Result(dict):
    scores: Dataset
    dataset: t.Optional[Dataset] = None
    binary_columns: t.List[str] = field(default_factory=list)

    def __post_init__(self):
        values = []
        for cn in self.scores[0].keys():
            value = np.nanmean(self.scores[cn])
            self[cn] = value
            if cn not in self.binary_columns:
                value = t.cast(float, value)
                values.append(value + 1e-10)

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def __repr__(self) -> str:
        scores = self.copy()
        score_strs = [f"'{k}': {v:0.4f}" for k, v in scores.items()]
        return "{" + ", ".join(score_strs) + "}"
