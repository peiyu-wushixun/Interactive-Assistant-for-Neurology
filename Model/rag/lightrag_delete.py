"""LightRAG 删除相关 mixin。

包含：
- delete_by_entity / adelete_by_entity          按实体删除
- _delete_by_entity_done
- delete_by_relation / adelete_by_relation      按关系删除
- _delete_relation_done
- get_processing_status / get_docs_by_status    文档处理状态查询
- adelete_by_doc_id                             按文档删除（包含 entity/relation 牵连处理）
"""
from __future__ import annotations

import asyncio
from typing import cast

from .base import DocStatus, StorageNameSpace
from .prompt import GRAPH_FIELD_SEP
from .utils import (
    always_get_an_event_loop,
    compute_mdhash_id,
    logger,
)


class DeleteMixin:
    """提供按实体 / 关系 / 文档粒度的删除能力。"""

    # ---------- 按实体删除 ----------

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    # ---------- 按关系删除 ----------

    def delete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Synchronously delete a relation between two entities."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
        )

    async def adelete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Asynchronously delete a relation between two entities."""
        try:
            edge_exists = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                logger.warning(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
                return

            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await self.relationships_vdb.delete([relation_id])

            await self.chunk_entity_relation_graph.remove_edges(
                [(source_entity, target_entity)]
            )

            logger.info(
                f"Successfully deleted relation from '{source_entity}' to '{target_entity}'"
            )
            await self._delete_relation_done()
        except Exception as e:
            logger.error(
                f"Error while deleting relation from '{source_entity}' to '{target_entity}': {e}"
            )

    async def _delete_relation_done(self) -> None:
        """Callback after relation deletion is complete"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    # ---------- 文档处理状态 ----------

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts."""
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict:
        """Get documents by status."""
        return await self.doc_status.get_docs_by_status(status)

    # ---------- 按文档删除 ----------

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data."""
        try:
            # 1. Get the document status and related data
            if not await self.doc_status.get_by_id(doc_id):
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all chunks related to this document
            all_chunks = await self.text_chunks.get_all()
            related_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in all_chunks.items()
                if isinstance(chunk_data, dict)
                and chunk_data.get("full_doc_id") == doc_id
            }

            if not related_chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return

            chunk_ids = set(related_chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships
            for chunk_id in chunk_ids:
                entities_storage = await self.entities_vdb.client_storage
                entities = [
                    dp
                    for dp in entities_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                relationships_storage = await self.relationships_vdb.client_storage
                relations = [
                    dp
                    for dp in relationships_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            entities_to_delete = set()
            entities_to_update = {}
            relationships_to_delete = set()
            relationships_to_update = {}

            all_labels = await self.chunk_entity_relation_graph.get_all_labels()
            for node_label in all_labels:
                node_data = await self.chunk_entity_relation_graph.get_node(node_label)
                if node_data and "source_id" in node_data:
                    sources = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node_label)
                        logger.debug(
                            f"Entity {node_label} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node_label] = new_source_id
                        logger.debug(
                            f"Entity {node_label} will be updated with new source_id: {new_source_id}"
                        )

            for node_label in all_labels:
                node_edges = await self.chunk_entity_relation_graph.get_node_edges(
                    node_label
                )
                if node_edges:
                    for src, tgt in node_edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(
                            src, tgt
                        )
                        if edge_data and "source_id" in edge_data:
                            sources = set(
                                edge_data["source_id"].split(GRAPH_FIELD_SEP)
                            )
                            sources.difference_update(chunk_ids)
                            if not sources:
                                relationships_to_delete.add((src, tgt))
                                logger.debug(
                                    f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                                )
                            else:
                                new_source_id = GRAPH_FIELD_SEP.join(sources)
                                relationships_to_update[(src, tgt)] = new_source_id
                                logger.debug(
                                    f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                                )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                await self.chunk_entity_relation_graph.remove_nodes(
                    list(entities_to_delete)
                )
                logger.debug(
                    f"Deleted {len(entities_to_delete)} entities from graph"
                )

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = await self.chunk_entity_relation_graph.get_node(entity)
                if node_data:
                    node_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_node(
                        entity, node_data
                    )
                    logger.debug(
                        f"Updated entity {entity} with new source_id: {new_source_id}"
                    )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(
                        f"Deleted relationship {src}-{tgt} from vector DB"
                    )
                await self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                if edge_data:
                    edge_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_edge(
                        src, tgt, edge_data
                    )
                    logger.debug(
                        f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                    )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            async def process_data(data_type, vdb, chunk_id):
                storage = await vdb.client_storage
                data_with_chunk = [
                    dp
                    for dp in storage["data"]
                    if chunk_id in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                ]

                data_for_vdb = {}
                if data_with_chunk:
                    logger.warning(
                        f"found {len(data_with_chunk)} {data_type} still referencing chunk {chunk_id}"
                    )

                    for item in data_with_chunk:
                        old_sources = item["source_id"].split(GRAPH_FIELD_SEP)
                        new_sources = [src for src in old_sources if src != chunk_id]

                        if not new_sources:
                            logger.info(
                                f"{data_type} {item.get('entity_name', 'N/A')} is deleted because source_id is not exists"
                            )
                            await vdb.delete_entity(item)
                        else:
                            item["source_id"] = GRAPH_FIELD_SEP.join(new_sources)
                            item_id = item["__id__"]
                            data_for_vdb[item_id] = item.copy()
                            if data_type == "entities":
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    item.get("entity_name", "")
                                    + (item.get("description") or "")
                                )
                            else:  # relationships
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    (item.get("keywords") or "")
                                    + (item.get("src_id") or "")
                                    + (item.get("tgt_id") or "")
                                    + (item.get("description") or "")
                                )

                    if data_for_vdb:
                        await vdb.upsert(data_for_vdb)
                        logger.info(
                            f"Successfully updated {data_type} in vector DB"
                        )

            # Add verification step
            async def verify_deletion():
                if await self.full_docs.get_by_id(doc_id):
                    logger.warning(f"Document {doc_id} still exists in full_docs")

                all_remaining_chunks = await self.text_chunks.get_all()
                remaining_related_chunks = {
                    chunk_id: chunk_data
                    for chunk_id, chunk_data in all_remaining_chunks.items()
                    if isinstance(chunk_data, dict)
                    and chunk_data.get("full_doc_id") == doc_id
                }

                if remaining_related_chunks:
                    logger.warning(
                        f"Found {len(remaining_related_chunks)} remaining chunks"
                    )

                for chunk_id in chunk_ids:
                    await process_data("entities", self.entities_vdb, chunk_id)
                    await process_data(
                        "relationships", self.relationships_vdb, chunk_id
                    )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")