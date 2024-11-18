import os
import uuid

from ragas.testset import TestsetGenerator


def test_testset_generation():
    from ragas.llms import llm_factory
    from ragas.testset.graph import KnowledgeGraph

    kg = KnowledgeGraph.load(
        os.path.join(os.path.dirname(__file__), "scratchpad_kg.json")
    )
    tg = TestsetGenerator(llm=llm_factory(), knowledge_graph=kg)
    testset = tg.generate(testset_size=10)
    assert testset is not None


def test_transforms():
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.testset.graph import KnowledgeGraph, Node
    from ragas.testset.transforms import apply_transforms, default_transforms

    transforms = default_transforms(
        llm=llm_factory(), embedding_model=embedding_factory()
    )

    kg = KnowledgeGraph()
    kg.nodes.append(Node(id=uuid.uuid4(), properties={"page_content": "Hello, world!"}))
    assert len(kg.nodes) == 1

    apply_transforms(kg, transforms)
    assert len(kg.nodes) == 1
