"""实体抽取相关 prompt。

包含：
- entity_extraction              实体抽取主 prompt
- entity_extraction_examples     实体抽取示例
- summarize_entity_descriptions  实体描述合并
- entity_continue_extraction     补充抽取
- entity_if_loop_extraction      是否需要继续抽取的判断

这些 prompt 会被 prompt.py 聚合成 PROMPTS["entity_extraction"] 等键。
"""
from __future__ import annotations
from typing import Any


EXTRACTION_PROMPTS: dict[str, Any] = {}


EXTRACTION_PROMPTS["entity_extraction"] = """
---目的---
给定一段文本和一份文本中可能包含的实体类型列表，从文本中识别出这些类型的所有实体以及识别实体之间的所有关系，不要有所遗漏。
使用{language}作为输出语言。

---步骤---
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。
- entity_type：实体类型： 以下类型之一： [{entity_types}]；
- entity_description： 实体描述： 实体在该段文本中的全面描述。
将每个实体格式化为以下形式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1确定的实体中，找出明确的所有(source_entity, target_entity) 对，对于每一对相关实体，提取以下信息：
- source_entity: ：源实体的名称，步骤1中确定的实体名称；
- target_entity：目标实体的名称，步骤1中确定的实体名称；
- relationship_description: 解释源实体和目标实体相互关联的原因；
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分值，0-10之间；
- relationship_keywords: 一个或多个关键词，概括关系的性质，侧重于具体细节。
将每个关系格式化为以下形式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 找出概括全文主要概念、主题或话题的关键词。这些关键词应能捕捉到文件中的总体思想。
关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}为输出语言，将步骤1和2中识别出的所有实体和关系以单个列表的形式输出。使用 **{record_delimiter}** 作为列表分隔符。

5. 当结束时，输出{completion_delimiter}

######################
---例子---
######################
{examples}

#############################
---真实数据---
######################
实体类型: [{entity_types}]
文本内容:
{input_text}
######################
输出:"""


EXTRACTION_PROMPTS["entity_extraction_examples"] = [
    """例子1:
实体类型: ["疾病", "药品", "症状", "食谱", "人口群体", "器官/组织", "科室", "细菌/病毒/寄生虫"]
文本内容:
```
肺-胸膜阿米巴病:肺-胸膜阿米巴病是溶组织阿米巴原虫感染所致的肺及胸膜化脓性炎症,肝原性病变多发生在右下肺,血源性则多为两肺多发病变.科室:呼吸内科.常用药品:替硝唑片,甲硝唑片
```
输出：  
("entity"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病是溶组织阿米巴原虫感染所致的肺及胸膜化脓性炎症"){record_delimiter}
("entity"{tuple_delimiter}"阿米巴原虫"{tuple_delimiter}"细菌/病毒/寄生虫"{tuple_delimiter}"感染肺-胸膜阿米巴病的寄生虫"){record_delimiter}
("entity"{tuple_delimiter}"肺"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺-胸膜阿米巴病的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"胸膜"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺-胸膜阿米巴病的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病的主要病变类型"){record_delimiter}
("entity"{tuple_delimiter}"血源性病变"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病的主要病变类型"){record_delimiter}
("entity"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"科室"{tuple_delimiter}"肺-胸膜阿米巴病所属科室"){record_delimiter}
("entity"{tuple_delimiter}"替硝唑片"{tuple_delimiter}"药品"{tuple_delimiter}"肺-胸膜阿米巴病患者常用药品"){record_delimiter}
("entity"{tuple_delimiter}"甲硝唑片"{tuple_delimiter}"药品"{tuple_delimiter}"肺-胸膜阿米巴病患者常用药品"){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"阿米巴原虫"{tuple_delimiter}"溶组织阿米巴原虫感染可能会导致肺-胸膜阿米巴病"{tuple_delimiter}"感染,致病原因"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"肺"{tuple_delimiter}"溶组织阿米巴原虫感染所致的肺化脓性炎症可能是肺-胸膜阿米巴病"{tuple_delimiter}"靶向器官,作用组织"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"胸膜"{tuple_delimiter}"溶组织阿米巴原虫感染所致的胸膜化脓性炎症可能是肺-胸膜阿米巴病"{tuple_delimiter}"靶向器官,作用组织"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"肝原性病变是肺-胸膜阿米巴病的主要病变类型，主要作用器官是右下肺"{tuple_delimiter}"病变类型"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"血源性病变"{tuple_delimiter}"血源性病变是肺-胸膜阿米巴病的主要病变类型，主要作用器官是两肺"{tuple_delimiter}"病变类型"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"呼吸内科是肺-胸膜阿米巴病所属的科室，疑似患有该疾病的人群应于呼吸内科进行诊断治疗"{tuple_delimiter}"所属科室"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"肺-胸膜阿米巴病肝原性病变的主要作用器官是右下肺"{tuple_delimiter}"靶向器官"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"肺"{tuple_delimiter}"血源性病变"{tuple_delimiter}"肺-胸膜阿米巴病血源性病变的主要作用器官是两肺"{tuple_delimiter}"靶向器官"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"替硝唑片"{tuple_delimiter}"替硝唑片是肺-胸膜阿米巴病的常用药品，但是否需要需要参考医生建议"{tuple_delimiter}"常用药品"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"甲硝唑片"{tuple_delimiter}"甲硝唑片是肺-胸膜阿米巴病的常用药品，但是否需要需要参考医生建议"{tuple_delimiter}"常用药品"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"肺-胸膜阿米巴病简介,肺-胸膜阿米巴病药物,肺-胸膜阿米巴病科室"){completion_delimiter}
###############""",
    """例子2:
实体类型: ["疾病", "药品", "症状", "食谱", "人口群体", "器官/组织", "科室", "细菌/病毒/寄生虫"]
```
肺泡蛋白沉着症:肺泡蛋白沉着症是一种原因未明的少见疾病.症状:肺泡炎症,胸痛,乏力.科室:呼吸内科
```
输出：
("entity"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"疾病"{tuple_delimiter}"肺泡蛋白沉着症是一种原因未明的少见疾病"){record_delimiter}
("entity"{tuple_delimiter}"肺泡"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺泡蛋白沉着症的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"胸痛"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"乏力"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"科室"{tuple_delimiter}"肺泡蛋白沉着症所属科室"){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"肺泡"{tuple_delimiter}"肺泡是肺泡蛋白沉着症的主要靶向器官"{tuple_delimiter}"靶向器官"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"肺泡炎症是一种疾病，同时也是肺泡蛋白沉着症的呈现症状，即出现肺泡炎症的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"胸痛"{tuple_delimiter}"胸痛是肺泡蛋白沉着症的呈现症状，即出现胸痛的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"乏力"{tuple_delimiter}"乏力是肺泡蛋白沉着症的呈现症状，即出现乏力的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"呼吸内科是肺泡蛋白沉着症所属的科室，疑似患有该疾病的人群应于呼吸内科进行诊断治疗"{tuple_delimiter}"所属科室"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"肺泡"{tuple_delimiter}"肺泡是肺泡蛋白沉着症的主要靶向器官，同时也是肺泡炎症的主要靶向器官"{tuple_delimiter}"靶向器官"{tuple_delimiter}2){record_delimiter}
("content_keywords"{tuple_delimiter}"肺泡蛋白沉着症简介,肺泡蛋白沉着症症状,肺泡蛋白沉着症科室"){completion_delimiter}
#############################""",
]


EXTRACTION_PROMPTS[
    "summarize_entity_descriptions"
] = """
你作为智能助手，需要根据以下实体描述信息进行最终描述信息的整合，具体要求如下：
1、合并所有相关描述内容，但需要保留全部的描述关键信息
2、描述间可能存在矛盾点，需要通过你的专业知识进行辨识和解决
3、采用第三人称客观叙述，确保包含体名称作为主语
5、使用{language}语言输出

#######
---数据---
实体：{entity_name}
描述列表：{description_list}
#######
输出:
"""


EXTRACTION_PROMPTS["entity_continue_extraction"] = """
注意：前次提取存在明显遗漏实体和关系，请严格按以下流程进行补充。
---步骤---
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。
- entity_type：实体类型： 以下类型之一： [{entity_types}]；
- entity_description： 实体描述： 实体在该段文本中的全面描述。
将每个实体格式化为以下形式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1确定的实体中，找出明确的所有(source_entity, target_entity) 对，对于每一对相关实体，提取以下信息：
- source_entity: ：源实体的名称，步骤1中确定的实体名称；
- target_entity：目标实体的名称，步骤1中确定的实体名称；
- relationship_description: 解释源实体和目标实体相互关联的原因；
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分值，0-10之间；
- relationship_keywords: 一个或多个关键词，概括关系的性质，侧重于具体细节。
将每个关系格式化为以下形式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 找出概括全文主要概念、主题或话题的关键词。这些关键词应能捕捉到文件中的总体思想。
关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}为输出语言，将步骤1和2中识别出的所有实体和关系以单个列表的形式输出。使用 **{record_delimiter}** 作为列表分隔符。

5. 当结束时，输出{completion_delimiter}

---输出---
使用相同的格式将它们添加到下面：\n
""".strip()


EXTRACTION_PROMPTS["entity_if_loop_extraction"] = """
---目标---
似乎仍有部分实体被遗漏。

---输出---
仅回答 YES 或 NO，判断是否还有需要添加的实体。
""".strip()