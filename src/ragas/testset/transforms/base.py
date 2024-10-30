import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragas.llms import BaseRagasLLM, llm_factory
from ragas.prompt import PromptMixin
from ragas.testset.graph import KnowledgeGraph, Node, Relationship

logger = logging.getLogger(__name__)


def default_filter(node: Node) -> bool:
    return True


@dataclass
class BaseGraphTransformation(ABC):
    """
    Abstract base class for graph transformations on a KnowledgeGraph.
    """

    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    async def transform(self, kg: KnowledgeGraph) -> t.Any:
        """
        Abstract method to transform the KnowledgeGraph. Transformations should be
        idempotent, meaning that applying the transformation multiple times should
        yield the same result as applying it once.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.Any
            The transformed knowledge graph.
        """
        pass

    def filter(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Filters the KnowledgeGraph and returns the filtered graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be filtered.

        Returns
        -------
        KnowledgeGraph
            The filtered knowledge graph.
        """
        return kg

    @abstractmethod
    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """
        Generates a list of coroutines to be executed in sequence by the Executor. This
        coroutine will, upon execution, write the transformation into the KnowledgeGraph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Coroutine]
            A list of coroutines to be executed in parallel.
        """
        pass


@dataclass
class Extractor(BaseGraphTransformation):
    """
    Abstract base class for extractors that transform a KnowledgeGraph by extracting
    specific properties from its nodes.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]
        Transforms the KnowledgeGraph by extracting properties from its nodes.

    extract(node: Node) -> t.Tuple[str, t.Any]
        Abstract method to extract a specific property from a node.
    """

    filter_nodes: t.Callable[[Node], bool] = field(
        default_factory=lambda: default_filter
    )

    async def transform(
        self, kg: KnowledgeGraph
    ) -> t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]:
        """
        Transforms the KnowledgeGraph by extracting properties from its nodes. Uses
        the `filter` method to filter the graph and the `extract` method to extract
        properties from each node.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]
            A list of tuples where each tuple contains a node and the extracted
            property.

        Examples
        --------
        >>> kg = KnowledgeGraph(nodes=[Node(id=1, properties={"name": "Node1"}), Node(id=2, properties={"name": "Node2"})])
        >>> extractor = SomeConcreteExtractor()
        >>> extractor.transform(kg)
        [(Node(id=1, properties={"name": "Node1"}), ("property_name", "extracted_value")),
         (Node(id=2, properties={"name": "Node2"}), ("property_name", "extracted_value"))]
        """
        filtered = self.filter(kg)
        return [(node, await self.extract(node)) for node in filtered.nodes]

    @abstractmethod
    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """
        Abstract method to extract a specific property from a node.

        Parameters
        ----------
        node : Node
            The node from which to extract the property.

        Returns
        -------
        t.Tuple[str, t.Any]
            A tuple containing the property name and the extracted value.
        """
        pass

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """
        Generates a list of coroutines to be executed in parallel by the Executor.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Coroutine]
            A list of coroutines to be executed in parallel.
        """

        async def apply_extract(node: Node):
            property_name, property_value = await self.extract(node)
            if node.get_property(property_name) is None:
                node.add_property(property_name, property_value)
            else:
                logger.warning(
                    "Property '%s' already exists in node '%.6s'. Skipping!",
                    property_name,
                    node.id,
                )

        filtered = self.filter(kg)
        return [apply_extract(node) for node in filtered.nodes]

    def filter(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[node for node in kg.nodes if self.filter_nodes(node)],
            relationships=[
                rel
                for rel in kg.relationships
                if rel.source in kg.nodes and rel.target in kg.nodes
            ],
        )


@dataclass
class LLMBasedExtractor(Extractor, PromptMixin):
    llm: BaseRagasLLM = field(kw_only=True)
    merge_if_possible: bool = True


class Splitter(BaseGraphTransformation):
    """
    Abstract base class for splitters that transform a KnowledgeGraph by splitting
    its nodes into smaller chunks.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.Tuple[t.List[Node], t.List[Relationship]]
        Transforms the KnowledgeGraph by splitting its nodes into smaller chunks.

    split(node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]
        Abstract method to split a node into smaller chunks.
    """

    async def transform(
        self, kg: KnowledgeGraph
    ) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        """
        Transforms the KnowledgeGraph by splitting its nodes into smaller chunks.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.Tuple[t.List[Node], t.List[Relationship]]
            A tuple containing a list of new nodes and a list of new relationships.
        """
        filtered = self.filter(kg)

        all_nodes = []
        all_relationships = []
        for node in filtered.nodes:
            nodes, relationships = await self.split(node)
            all_nodes.extend(nodes)
            all_relationships.extend(relationships)

        return all_nodes, all_relationships

    @abstractmethod
    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        """
        Abstract method to split a node into smaller chunks.

        Parameters
        ----------
        node : Node
            The node to be split.

        Returns
        -------
        t.Tuple[t.List[Node], t.List[Relationship]]
            A tuple containing a list of new nodes and a list of new relationships.
        """
        pass

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """
        Generates a list of coroutines to be executed in parallel by the Executor.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Coroutine]
            A list of coroutines to be executed in parallel.
        """

        async def apply_split(node: Node):
            nodes, relationships = await self.split(node)
            kg.nodes.extend(nodes)
            kg.relationships.extend(relationships)

        filtered = self.filter(kg)
        return [apply_split(node) for node in filtered.nodes]


class RelationshipBuilder(BaseGraphTransformation):
    """
    Abstract base class for building relationships in a KnowledgeGraph.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.List[Relationship]
        Transforms the KnowledgeGraph by building relationships.
    """

    @abstractmethod
    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        """
        Transforms the KnowledgeGraph by building relationships.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[Relationship]
            A list of new relationships.
        """
        pass

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """
        Generates a list of coroutines to be executed in parallel by the Executor.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Coroutine]
            A list of coroutines to be executed in parallel.
        """

        async def apply_build_relationships(
            filtered_kg: KnowledgeGraph, original_kg: KnowledgeGraph
        ):
            relationships = await self.transform(filtered_kg)
            original_kg.relationships.extend(relationships)

        filtered_kg = self.filter(kg)
        return [apply_build_relationships(filtered_kg=filtered_kg, original_kg=kg)]
