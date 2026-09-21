"""LightRAG 查询相关 mixin。

包含：
- query / aquery                                标准查询入口（按 param.mode 分发）
- query_with_separate_keyword_extraction / ...  单独的关键词抽取查询
- _query_done                                   查询后回写 LLM 响应缓存

被 lightrag.LightRAG 通过多继承复用。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import AsyncIterator, Iterator

from .base import QueryParam
from .operate import (
    kg_query,
    mix_kg_vector_query,
    naive_query,
    query_with_keywords,
)
from .utils import always_get_an_event_loop


class QueryMixin:
    """提供基于图谱/向量的多模式查询能力。"""

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior.

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param, system_prompt))

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """Perform an async query, dispatching on param.mode."""
        global_config = asdict(self)

        if param.mode in ["local", "global", "hybrid", "ll_medical"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        Query with separate keyword extraction step.

        Args:
            query: User query
            prompt: Additional prompt for the query
            param: Query parameters

        Returns:
            Query response
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> str | AsyncIterator[str]:
        """Async version of query_with_separate_keyword_extraction."""
        response = await query_with_keywords(
            query=query,
            prompt=prompt,
            param=param,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            chunks_vdb=self.chunks_vdb,
            text_chunks_db=self.text_chunks,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache,
        )

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()