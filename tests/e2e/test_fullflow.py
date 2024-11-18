from datasets import load_dataset

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness


def test_evaluate_e2e():
    dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])  # type: ignore
    result = evaluate(
        eval_dataset,
        metrics=[
            AnswerRelevancy(llm=llm_factory()),
            ContextPrecision(llm=llm_factory()),
            Faithfulness(llm=llm_factory()),
        ],
    )
    assert result is not None
