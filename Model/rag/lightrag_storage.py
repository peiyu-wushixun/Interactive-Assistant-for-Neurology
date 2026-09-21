"""LightRAG 存储管理相关 mixin。

包含：
- _run_async_safely            在事件循环里安全地跑异步函数
- initialize_storages          异步初始化所有 storage
- finalize_storages            异步关闭所有 storage
- _get_storage_class           根据名字懒加载 storage 实现
- check_storage_env_vars       检查 storage 所需环境变量

被 lightrag.LightRAG 通过多继承复用。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from lightrag.kg import (
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
)
from .base import StoragesStatus
from .utils import (
    always_get_an_event_loop,
    lazy_external_import,
    logger,
)


class StorageMixin:
    """提供存储初始化/收尾以及按名加载存储实现的能力。

    注意：本 mixin 不定义任何 dataclass 字段，所有字段都在 ``LightRAG`` 上；
    这里仅声明方法，运行时 ``self`` 即为 ``LightRAG`` 实例。
    """

    def _run_async_safely(self, async_func, action_name: str = ""):
        """Safely execute an async function, avoiding event loop conflicts."""
        try:
            loop = always_get_an_event_loop()
            if loop.is_running():
                task = loop.create_task(async_func())
                task.add_done_callback(
                    lambda t: logger.info(f"{action_name} completed!")
                )
            else:
                loop.run_until_complete(async_func())
        except RuntimeError:
            logger.warning(
                f"No running event loop, creating a new loop for {action_name}."
            )
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_func())
            loop.close()

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist.

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )