"""LightRAG 数据导出与缓存清理相关 mixin。

包含：
- aclear_cache / clear_cache    按 mode 清理 LLM 响应缓存
- aexport_data / export_data    把实体、关系导出为 csv / excel / md / txt
"""
from __future__ import annotations

import asyncio
import csv
from typing import Literal

import pandas as pd

from .utils import (
    always_get_an_event_loop,
    logger,
)


class ExportMixin:
    """提供缓存清理与数据导出能力。"""

    # ---------- 缓存清理 ----------

    async def aclear_cache(self, modes: list[str] | None = None) -> None:
        """Clear cache data from the LLM response cache storage.

        Args:
            modes (list[str] | None): Modes of cache to clear.
                Options: ["default", "naive", "local", "global", "hybrid", "mix"].
                "default" represents extraction cache.
                If None, clears all cache.
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        valid_modes = ["default", "naive", "local", "global", "hybrid", "mix"]

        if modes and not all(mode in valid_modes for mode in modes):
            raise ValueError(f"Invalid mode. Valid modes are: {valid_modes}")

        try:
            if modes:
                await self.llm_response_cache.delete(modes)
                logger.info(f"Cleared cache for modes: {modes}")
            else:
                await self.llm_response_cache.delete(valid_modes)
                logger.info("Cleared all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self, modes: list[str] | None = None) -> None:
        """Synchronous version of aclear_cache."""
        return always_get_an_event_loop().run_until_complete(self.aclear_cache(modes))

    # ---------- 数据导出 ----------

    async def aexport_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """Asynchronously exports all entities, relations, and relationships."""
        # Collect data
        entities_data = []
        relations_data = []
        relationships_data = []

        # --- Entities ---
        all_entities = await self.chunk_entity_relation_graph.get_all_labels()
        for entity_name in all_entities:
            entity_info = await self.get_entity_info(
                entity_name, include_vector_data=include_vector_data
            )
            entity_row = {
                "entity_name": entity_name,
                "source_id": entity_info["source_id"],
                "graph_data": str(entity_info["graph_data"]),
            }
            if include_vector_data and "vector_data" in entity_info:
                entity_row["vector_data"] = str(entity_info["vector_data"])
            entities_data.append(entity_row)

        # --- Relations ---
        for src_entity in all_entities:
            for tgt_entity in all_entities:
                if src_entity == tgt_entity:
                    continue
                edge_exists = await self.chunk_entity_relation_graph.has_edge(
                    src_entity, tgt_entity
                )
                if edge_exists:
                    relation_info = await self.get_relation_info(
                        src_entity,
                        tgt_entity,
                        include_vector_data=include_vector_data,
                    )
                    relation_row = {
                        "src_entity": src_entity,
                        "tgt_entity": tgt_entity,
                        "source_id": relation_info["source_id"],
                        "graph_data": str(relation_info["graph_data"]),
                    }
                    if include_vector_data and "vector_data" in relation_info:
                        relation_row["vector_data"] = str(relation_info["vector_data"])
                    relations_data.append(relation_row)

        # --- Relationships (from VectorDB) ---
        all_relationships = await self.relationships_vdb.client_storage
        for rel in all_relationships["data"]:
            relationships_data.append(
                {
                    "relationship_id": rel["__id__"],
                    "data": str(rel),
                }
            )

        # Export based on format
        if file_format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                if entities_data:
                    csvfile.write("# ENTITIES\n")
                    writer = csv.DictWriter(
                        csvfile, fieldnames=entities_data[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(entities_data)
                    csvfile.write("\n\n")

                if relations_data:
                    csvfile.write("# RELATIONS\n")
                    writer = csv.DictWriter(
                        csvfile, fieldnames=relations_data[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(relations_data)
                    csvfile.write("\n\n")

                if relationships_data:
                    csvfile.write("# RELATIONSHIPS\n")
                    writer = csv.DictWriter(
                        csvfile, fieldnames=relationships_data[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(relationships_data)

        elif file_format == "excel":
            entities_df = (
                pd.DataFrame(entities_data) if entities_data else pd.DataFrame()
            )
            relations_df = (
                pd.DataFrame(relations_data) if relations_data else pd.DataFrame()
            )
            relationships_df = (
                pd.DataFrame(relationships_data)
                if relationships_data
                else pd.DataFrame()
            )

            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                if not entities_df.empty:
                    entities_df.to_excel(
                        writer, sheet_name="Entities", index=False
                    )
                if not relations_df.empty:
                    relations_df.to_excel(
                        writer, sheet_name="Relations", index=False
                    )
                if not relationships_df.empty:
                    relationships_df.to_excel(
                        writer, sheet_name="Relationships", index=False
                    )

        elif file_format == "md":
            with open(output_path, "w", encoding="utf-8") as mdfile:
                mdfile.write("# LightRAG Data Export\n\n")

                mdfile.write("## Entities\n\n")
                if entities_data:
                    mdfile.write(
                        "| " + " | ".join(entities_data[0].keys()) + " |\n"
                    )
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(entities_data[0].keys()))
                        + " |\n"
                    )
                    for entity in entities_data:
                        mdfile.write(
                            "| "
                            + " | ".join(str(v) for v in entity.values())
                            + " |\n"
                        )
                    mdfile.write("\n\n")
                else:
                    mdfile.write("*No entity data available*\n\n")

                mdfile.write("## Relations\n\n")
                if relations_data:
                    mdfile.write(
                        "| " + " | ".join(relations_data[0].keys()) + " |\n"
                    )
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(relations_data[0].keys()))
                        + " |\n"
                    )
                    for relation in relations_data:
                        mdfile.write(
                            "| "
                            + " | ".join(str(v) for v in relation.values())
                            + " |\n"
                        )
                    mdfile.write("\n\n")
                else:
                    mdfile.write("*No relation data available*\n\n")

                mdfile.write("## Relationships\n\n")
                if relationships_data:
                    mdfile.write(
                        "| "
                        + " | ".join(relationships_data[0].keys())
                        + " |\n"
                    )
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(relationships_data[0].keys()))
                        + " |\n"
                    )
                    for relationship in relationships_data:
                        mdfile.write(
                            "| "
                            + " | ".join(str(v) for v in relationship.values())
                            + " |\n"
                        )
                else:
                    mdfile.write("*No relationship data available*\n\n")

        elif file_format == "txt":
            with open(output_path, "w", encoding="utf-8") as txtfile:
                txtfile.write("LIGHTRAG DATA EXPORT\n")
                txtfile.write("=" * 80 + "\n\n")

                txtfile.write("ENTITIES\n")
                txtfile.write("-" * 80 + "\n")
                if entities_data:
                    col_widths = {
                        k: max(
                            len(k),
                            max(len(str(e[k])) for e in entities_data),
                        )
                        for k in entities_data[0]
                    }
                    header = "  ".join(
                        k.ljust(col_widths[k]) for k in entities_data[0]
                    )
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")
                    for entity in entities_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in entity.items()
                        )
                        txtfile.write(row + "\n")
                    txtfile.write("\n\n")
                else:
                    txtfile.write("No entity data available\n\n")

                txtfile.write("RELATIONS\n")
                txtfile.write("-" * 80 + "\n")
                if relations_data:
                    col_widths = {
                        k: max(
                            len(k),
                            max(len(str(r[k])) for r in relations_data),
                        )
                        for k in relations_data[0]
                    }
                    header = "  ".join(
                        k.ljust(col_widths[k]) for k in relations_data[0]
                    )
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")
                    for relation in relations_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in relation.items()
                        )
                        txtfile.write(row + "\n")
                    txtfile.write("\n\n")
                else:
                    txtfile.write("No relation data available\n\n")

                txtfile.write("RELATIONSHIPS\n")
                txtfile.write("-" * 80 + "\n")
                if relationships_data:
                    col_widths = {
                        k: max(
                            len(k),
                            max(len(str(r[k])) for r in relationships_data),
                        )
                        for k in relationships_data[0]
                    }
                    header = "  ".join(
                        k.ljust(col_widths[k]) for k in relationships_data[0]
                    )
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")
                    for relationship in relationships_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in relationship.items()
                        )
                        txtfile.write(row + "\n")
                else:
                    txtfile.write("No relationship data available\n\n")

        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Choose from: csv, excel, md, txt"
            )

        if file_format is not None:
            print(f"Data exported to: {output_path} with format: {file_format}")
        else:
            print("Data displayed as table format")

    def export_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """Synchronously exports all entities, relations, and relationships."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.aexport_data(output_path, file_format, include_vector_data)
        )