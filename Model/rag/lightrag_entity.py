"""LightRAG 实体 / 关系 CRUD 相关 mixin。

包含：
- get_entity_info / get_relation_info          单实体 / 关系信息查询
- acreate_entity / create_entity               创建实体
- acreate_relation / create_relation           创建关系
- aedit_entity / edit_entity                   编辑实体
- _edit_entity_done
- aedit_relation / edit_relation               编辑关系
- _edit_relation_done
- amerge_entities / merge_entities             合并多个实体到一个实体
- _merge_entity_attributes / _merge_relation_attributes   合并策略
- _merge_entities_done
"""
from __future__ import annotations

import asyncio
from typing import Any, cast

from .base import StorageNameSpace
from .prompt import GRAPH_FIELD_SEP
from .utils import (
    always_get_an_event_loop,
    compute_mdhash_id,
    logger,
)


class EntityMixin:
    """提供实体 / 关系的创建、编辑、合并、查询能力。"""

    # ---------- 实体 / 关系信息查询 ----------

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, Any]:
        """Get detailed information of an entity"""
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: dict[str, Any] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = await self.entities_vdb.get_by_id(entity_id)
            result["vector_data"] = vector_data

        return result

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, Any]:
        """Get detailed information of a relationship"""
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result: dict[str, Any] = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = await self.relationships_vdb.get_by_id(rel_id)
            result["vector_data"] = vector_data

        return result

    # ---------- 创建实体 ----------

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new entity."""
        try:
            existing_node = await self.chunk_entity_relation_graph.has_node(entity_name)
            if existing_node:
                raise ValueError(f"Entity '{entity_name}' already exists")

            node_data = {
                "entity_id": entity_name,
                "entity_type": entity_data.get("entity_type", "UNKNOWN"),
                "description": entity_data.get("description", ""),
                "source_id": entity_data.get("source_id", "manual"),
            }

            await self.chunk_entity_relation_graph.upsert_node(entity_name, node_data)

            description = node_data.get("description", "")
            source_id = node_data.get("source_id", "")
            entity_type = node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            await self.entities_vdb.upsert(entity_data_for_vdb)
            await self._edit_entity_done()

            logger.info(f"Entity '{entity_name}' successfully created")
            return await self.get_entity_info(entity_name, include_vector_data=True)
        except Exception as e:
            logger.error(f"Error while creating entity '{entity_name}': {e}")
            raise

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously create a new entity."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

    # ---------- 创建关系 ----------

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities."""
        try:
            source_exists = await self.chunk_entity_relation_graph.has_node(
                source_entity
            )
            target_exists = await self.chunk_entity_relation_graph.has_node(
                target_entity
            )

            if not source_exists:
                raise ValueError(f"Source entity '{source_entity}' does not exist")
            if not target_exists:
                raise ValueError(f"Target entity '{target_entity}' does not exist")

            existing_edge = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if existing_edge:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' already exists"
                )

            edge_data = {
                "description": relation_data.get("description", ""),
                "keywords": relation_data.get("keywords", ""),
                "source_id": relation_data.get("source_id", "manual"),
                "weight": float(relation_data.get("weight", 1.0)),
            }

            await self.chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, edge_data
            )

            description = edge_data.get("description", "")
            keywords = edge_data.get("keywords", "")
            source_id = edge_data.get("source_id", "")
            weight = edge_data.get("weight", 1.0)
            content = f"{keywords}\t{source_entity}\n{target_entity}\n{description}"

            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            relation_data_for_vdb = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }

            await self.relationships_vdb.upsert(relation_data_for_vdb)
            await self._edit_relation_done()

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully created"
            )
            return await self.get_relation_info(
                source_entity, target_entity, include_vector_data=True
            )
        except Exception as e:
            logger.error(
                f"Error while creating relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously create a new relation between entities."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.acreate_relation(source_entity, target_entity, relation_data)
        )

    # ---------- 编辑实体 ----------

    async def aedit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Asynchronously edit entity information."""
        try:
            node_exists = await self.chunk_entity_relation_graph.has_node(entity_name)
            if not node_exists:
                raise ValueError(f"Entity '{entity_name}' does not exist")
            node_data = await self.chunk_entity_relation_graph.get_node(entity_name)

            new_entity_name = updated_data.get("entity_name", entity_name)
            is_renaming = new_entity_name != entity_name

            if is_renaming:
                if not allow_rename:
                    raise ValueError(
                        "Entity renaming is not allowed. Set allow_rename=True to enable this feature"
                    )
                existing_node = await self.chunk_entity_relation_graph.has_node(
                    new_entity_name
                )
                if existing_node:
                    raise ValueError(
                        f"Entity name '{new_entity_name}' already exists, cannot rename"
                    )

            new_node_data = {**node_data, **updated_data}
            new_node_data["entity_id"] = new_entity_name
            if "entity_name" in new_node_data:
                del new_node_data["entity_name"]

            if is_renaming:
                logger.info(f"Renaming entity '{entity_name}' to '{new_entity_name}'")

                await self.chunk_entity_relation_graph.upsert_node(
                    new_entity_name, new_node_data
                )

                relations_to_update = []
                relations_to_delete = []
                edges = await self.chunk_entity_relation_graph.get_node_edges(
                    entity_name
                )
                if edges:
                    for source, target in edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(
                            source, target
                        )
                        if edge_data:
                            relations_to_delete.append(
                                compute_mdhash_id(source + target, prefix="rel-")
                            )
                            relations_to_delete.append(
                                compute_mdhash_id(target + source, prefix="rel-")
                            )
                            if source == entity_name:
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    new_entity_name, target, edge_data
                                )
                                relations_to_update.append(
                                    (new_entity_name, target, edge_data)
                                )
                            else:
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    source, new_entity_name, edge_data
                                )
                                relations_to_update.append(
                                    (source, new_entity_name, edge_data)
                                )

                await self.chunk_entity_relation_graph.delete_node(entity_name)

                old_entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([old_entity_id])
                logger.info(
                    f"Deleted old entity '{entity_name}' and its vector embedding from database"
                )

                await self.relationships_vdb.delete(relations_to_delete)
                logger.info(
                    f"Deleted {len(relations_to_delete)} relation records for entity '{entity_name}' from vector database"
                )

                for src, tgt, edge_data in relations_to_update:
                    description = edge_data.get("description", "")
                    keywords = edge_data.get("keywords", "")
                    source_id = edge_data.get("source_id", "")
                    weight = float(edge_data.get("weight", 1.0))
                    content = f"{src}\t{tgt}\n{keywords}\n{description}"
                    relation_id = compute_mdhash_id(src + tgt, prefix="rel-")
                    relation_data = {
                        relation_id: {
                            "content": content,
                            "src_id": src,
                            "tgt_id": tgt,
                            "source_id": source_id,
                            "description": description,
                            "keywords": keywords,
                            "weight": weight,
                        }
                    }
                    await self.relationships_vdb.upsert(relation_data)

                entity_name = new_entity_name
            else:
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, new_node_data
                )

            description = new_node_data.get("description", "")
            source_id = new_node_data.get("source_id", "")
            entity_type = new_node_data.get("entity_type", "")
            content = entity_name + "\n" + description
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_data = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }
            await self.entities_vdb.upsert(entity_data)

            await self._edit_entity_done()

            logger.info(f"Entity '{entity_name}' successfully updated")
            return await self.get_entity_info(entity_name, include_vector_data=True)
        except Exception as e:
            logger.error(f"Error while editing entity '{entity_name}': {e}")
            raise

    def edit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Synchronously edit entity information."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename)
        )

    async def _edit_entity_done(self) -> None:
        """Callback after entity editing/creation, ensures updates are persisted"""
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

    # ---------- 编辑关系 ----------

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information."""
        try:
            edge_exists = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
            edge_data = await self.chunk_entity_relation_graph.get_edge(
                source_entity, target_entity
            )

            old_relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await self.relationships_vdb.delete([old_relation_id])
            logger.info(
                f"Deleted old relation record from vector database for relation {source_entity} -> {target_entity}"
            )

            new_edge_data = {**edge_data, **updated_data}
            await self.chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, new_edge_data
            )

            description = new_edge_data.get("description", "")
            keywords = new_edge_data.get("keywords", "")
            source_id = new_edge_data.get("source_id", "")
            weight = float(new_edge_data.get("weight", 1.0))
            content = f"{source_entity}\t{target_entity}\n{keywords}\n{description}"

            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            relation_data = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }
            await self.relationships_vdb.upsert(relation_data)

            await self._edit_relation_done()

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully updated"
            )
            return await self.get_relation_info(
                source_entity, target_entity, include_vector_data=True
            )
        except Exception as e:
            logger.error(
                f"Error while editing relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously edit relation information."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
        )

    async def _edit_relation_done(self) -> None:
        """Callback after relation editing/creation, ensures updates are persisted"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    # ---------- 合并实体 ----------

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity."""
        try:
            default_strategy = {
                "description": "concatenate",
                "entity_type": "keep_first",
                "source_id": "join_unique",
            }
            merge_strategy = (
                default_strategy
                if merge_strategy is None
                else {**default_strategy, **merge_strategy}
            )
            target_entity_data = (
                {} if target_entity_data is None else target_entity_data
            )

            # 1. Check if all source entities exist
            source_entities_data = {}
            for entity_name in source_entities:
                node_exists = await self.chunk_entity_relation_graph.has_node(
                    entity_name
                )
                if not node_exists:
                    raise ValueError(f"Source entity '{entity_name}' does not exist")
                node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
                source_entities_data[entity_name] = node_data

            # 2. Check if target entity exists
            target_exists = await self.chunk_entity_relation_graph.has_node(
                target_entity
            )
            existing_target_entity_data = {}
            if target_exists:
                existing_target_entity_data = (
                    await self.chunk_entity_relation_graph.get_node(target_entity)
                )
                logger.info(
                    f"Target entity '{target_entity}' already exists, will merge data"
                )

            # 3. Merge entity data
            merged_entity_data = self._merge_entity_attributes(
                list(source_entities_data.values())
                + ([existing_target_entity_data] if target_exists else []),
                merge_strategy,
            )
            for key, value in target_entity_data.items():
                merged_entity_data[key] = value

            # 4. Get all relationships of the source entities
            all_relations = []
            for entity_name in source_entities:
                edges = await self.chunk_entity_relation_graph.get_node_edges(
                    entity_name
                )
                if edges:
                    for src, tgt in edges:
                        if src == entity_name:
                            edge_data = await self.chunk_entity_relation_graph.get_edge(
                                src, tgt
                            )
                            all_relations.append((src, tgt, edge_data))

            # 5. Create or update the target entity
            merged_entity_data["entity_id"] = target_entity
            await self.chunk_entity_relation_graph.upsert_node(
                target_entity, merged_entity_data
            )
            if not target_exists:
                logger.info(f"Created new target entity '{target_entity}'")
            else:
                logger.info(f"Updated existing target entity '{target_entity}'")

            # 6. Recreate all relationships, pointing to the target entity
            relation_updates = {}
            relations_to_delete = []
            for src, tgt, edge_data in all_relations:
                relations_to_delete.append(compute_mdhash_id(src + tgt, prefix="rel-"))
                relations_to_delete.append(compute_mdhash_id(tgt + src, prefix="rel-"))
                new_src = target_entity if src in source_entities else src
                new_tgt = target_entity if tgt in source_entities else tgt

                if new_src == new_tgt:
                    logger.info(
                        f"Skipping relationship between source entities: {src} -> {tgt} to avoid self-loop"
                    )
                    continue

                relation_key = f"{new_src}|{new_tgt}"
                if relation_key in relation_updates:
                    existing_data = relation_updates[relation_key]["data"]
                    merged_relation = self._merge_relation_attributes(
                        [existing_data, edge_data],
                        {
                            "description": "concatenate",
                            "keywords": "join_unique",
                            "source_id": "join_unique",
                            "weight": "max",
                        },
                    )
                    relation_updates[relation_key]["data"] = merged_relation
                    logger.info(
                        f"Merged duplicate relationship: {new_src} -> {new_tgt}"
                    )
                else:
                    relation_updates[relation_key] = {
                        "src": new_src,
                        "tgt": new_tgt,
                        "data": edge_data.copy(),
                    }

            for rel_data in relation_updates.values():
                await self.chunk_entity_relation_graph.upsert_edge(
                    rel_data["src"], rel_data["tgt"], rel_data["data"]
                )
                logger.info(
                    f"Created or updated relationship: {rel_data['src']} -> {rel_data['tgt']}"
                )

                await self.relationships_vdb.delete(relations_to_delete)
                logger.info(
                    f"Deleted {len(relations_to_delete)} relation records for entity '{entity_name}' from vector database"
                )

            # 7. Update entity vector representation
            description = merged_entity_data.get("description", "")
            source_id = merged_entity_data.get("source_id", "")
            entity_type = merged_entity_data.get("entity_type", "")
            content = target_entity + "\n" + description

            entity_id = compute_mdhash_id(target_entity, prefix="ent-")
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }
            await self.entities_vdb.upsert(entity_data_for_vdb)

            # 8. Update relationship vector representations
            for rel_data in relation_updates.values():
                src = rel_data["src"]
                tgt = rel_data["tgt"]
                edge_data = rel_data["data"]
                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                source_id = edge_data.get("source_id", "")
                weight = float(edge_data.get("weight", 1.0))
                content = f"{keywords}\t{src}\n{tgt}\n{description}"
                relation_id = compute_mdhash_id(src + tgt, prefix="rel-")
                relation_data_for_vdb = {
                    relation_id: {
                        "content": content,
                        "src_id": src,
                        "tgt_id": tgt,
                        "source_id": source_id,
                        "description": description,
                        "keywords": keywords,
                        "weight": weight,
                    }
                }
                await self.relationships_vdb.upsert(relation_data_for_vdb)

            # 9. Delete source entities
            for entity_name in source_entities:
                if entity_name == target_entity:
                    logger.info(
                        f"Skipping deletion of '{entity_name}' as it's also the target entity"
                    )
                    continue
                await self.chunk_entity_relation_graph.delete_node(entity_name)
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([entity_id])
                logger.info(
                    f"Deleted source entity '{entity_name}' and its vector embedding from database"
                )

            # 10. Save changes
            await self._merge_entities_done()

            logger.info(
                f"Successfully merged {len(source_entities)} entities into '{target_entity}'"
            )
            return await self.get_entity_info(target_entity, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            raise

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Synchronously merge multiple entities into one entity."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    def _merge_entity_attributes(
        self, entity_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple entities."""
        merged_data: dict[str, Any] = {}

        all_keys = set()
        for data in entity_data_list:
            all_keys.update(data.keys())

        for key in all_keys:
            values = [data.get(key) for data in entity_data_list if data.get(key)]
            if not values:
                continue

            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                unique_items = set()
                for value in values:
                    items = value.split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            else:
                merged_data[key] = values[0]

        return merged_data

    def _merge_relation_attributes(
        self, relation_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple relationships."""
        merged_data: dict[str, Any] = {}

        all_keys = set()
        for data in relation_data_list:
            all_keys.update(data.keys())

        for key in all_keys:
            values = [
                data.get(key)
                for data in relation_data_list
                if data.get(key) is not None
            ]
            if not values:
                continue

            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(str(v) for v in values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                unique_items = set()
                for value in values:
                    items = str(value).split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            elif strategy == "max":
                try:
                    merged_data[key] = max(float(v) for v in values)
                except (ValueError, TypeError):
                    merged_data[key] = values[0]
            else:
                merged_data[key] = values[0]

        return merged_data

    async def _merge_entities_done(self) -> None:
        """Callback after entity merging, ensures updates are persisted"""
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