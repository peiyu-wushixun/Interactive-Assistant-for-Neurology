"""LightRAG 知识图谱查询相关 mixin。

包含：
- get_graph_labels    取出图中所有标签
- get_knowledge_graph 根据起始节点取出子图
"""
from __future__ import annotations

import inspect

from .types import KnowledgeGraph


class KGMixin:
    """提供对底层知识图谱的查询能力。"""

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            min_degree (int, optional): Minimum degree of nodes to include. Defaults to 0.
            inclusive (bool, optional): Whether to use inclusive search mode. Defaults to False.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # get params supported by get_knowledge_graph of specified storage
        storage_params = inspect.signature(
            self.chunk_entity_relation_graph.get_knowledge_graph
        ).parameters

        kwargs = {"node_label": node_label, "max_depth": max_depth}

        if "min_degree" in storage_params and min_degree > 0:
            kwargs["min_degree"] = min_degree

        if "inclusive" in storage_params:
            kwargs["inclusive"] = inclusive

        return await self.chunk_entity_relation_graph.get_knowledge_graph(**kwargs)