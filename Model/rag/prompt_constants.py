"""Prompt 模块常量与默认值。

包含图谱分隔符、各类默认值（语言、分隔符、实体类型清单）。
这些常量被 prompt.py 聚合到 PROMPTS 字典中。
"""
from __future__ import annotations
from typing import Any


GRAPH_FIELD_SEP = "<SEP>"


# 顶层默认值，由 prompt.py 合并进 PROMPTS 字典
DEFAULT_PROMPTS: dict[str, Any] = {
    "DEFAULT_LANGUAGE": "中文",
    "DEFAULT_TUPLE_DELIMITER": "<|>",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
    "DEFAULT_ENTITY_TYPES": [
        "疾病",
        "药品",
        "症状",
        "食谱",
        "人口群体",
        "器官/组织",
        "科室",
        "细菌/病毒/寄生虫",
    ],
}