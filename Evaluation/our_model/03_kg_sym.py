# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 3/11: 疾病与症状提取
Slug: kg_sym
Note: outputs and cell magic were dropped; source code only.
"""


async def kg_sym(history, index=True):
    if index:
        example = """
        {
          "症状": [
            "反复咳嗽3天，夜间加重",
            "伴有喘息和喉咙痛",
            "既往有过敏性鼻炎史",
            "父亲有高血压病史",
            "每天吸烟约10支"
          ]
        }
        """
        entity_prompt = """
        你是一名专业的医疗文书结构化助手。你的任务是从非结构化的医患对话文本中，精准提取关键临床信息。

        # Task
        请阅读提供的【医患对话文本】，提取以下维度的信息，并将所有提取到的内容合并为一个字符串列表，统一赋值给 JSON 中的 "症状" 键：
        1. 患者描述的所有症状（包括主诉及伴随症状）
        2. 现病史（症状的演变、加重/缓解因素等关键描述）
        3. 既往史（过往疾病、手术史、过敏史等）
        4. 家族史（家族遗传病或相关疾病史）
        5. 生活习惯（吸烟、饮酒、饮食、作息等与健康相关的习惯）

        # Constraints
        - 准确性：严格基于对话原文提取，禁止推测或添加对话中未提及的信息。
        - 格式要求：必须且只能输出一个合法的 JSON 对象。
        - 键名限制：JSON 中只允许包含 "症状" 这一个键，严禁添加任何额外的键、解释性文字或 Markdown 标记。
        - 值格式："症状" 的值必须是一个字符串数组（List[str]），不需要分点描述，每个元素为一个独立的信息点。
        - 空值处理：如果对话中未提及某类信息，直接忽略该维度，不要输出“未提及”或 null。
        - 着重注意：本对话为真实医患沟通的转写，可能包含患者的犹豫（呃、嗯、那个）、自我修正（"不对，是右边"）、无关闲聊和口语化表述。请忽略这些噪声，专注于提取症状，并保留患者原始表达中的关键修饰词。
        """
        entity_input = f"""下列为患者和医生的对话历史:{history}\n
                    输出示例：{example}"""
        entity_input = entity_input + entity_prompt
    else:
        example = '{"疾病":["内耳问题","颈椎病","贫血"]}'
        entity_prompt = "请从医生的诊断结果中提取关键信息。请识别医生诊断可能患有的疾病，确保你的答案是准确无误的,且一定要按照输出示例以 JSON 格式进行输出，需要且只需要'疾病'这一个键，不需要添加格外内容。"
        entity_input = f"""下列为诊断结果:{history}\n
                        输出示例：{example}"""
        entity_input = entity_input + entity_prompt
    # response = chat(
    #     model='xiaowangge/minicpm4:8b',
    #     messages=[
    #         {'role': 'user', 'content': entity_input}
    #     ]
    # )
    # entity_responses = response['message']['content']
    # if "think" in entity_responses:
    #     entity_responses = re.sub(r'<think>.*?</think>', '', entity_responses, flags=re.DOTALL)
    #     entity_responses = entity_responses.replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
    #     if entity_responses[0]=="\n":
    #         entity_responses = entity_responses[1:]
    # entity_responses = entity_responses.replace('“', '"').replace('”', '"').replace('，', ',').replace(']]', ']').replace('[[', '[').replace("'", '"').replace("‘", '"').replace("’", '"').replace("json", '').replace("```", '').replace("\\", '').replace('心脏"漏跳"', '心脏漏跳')
    # print(entity_responses)
    # parts = entity_responses.split('（')
    # entity_responses = parts[0] + ''.join('（' + p.replace('"', "'") for p in parts[1:])
    # entity_responses = json.loads(entity_responses)
    
    
    while True:
        try:
            # 1. 获取 response
            response = chat(
                model='xiaowangge/minicpm4:8b',
                messages=[
                    {'role': 'user', 'content': entity_input}
                ]
            )
            entity_responses = response['message']['content']

            # 2. 清洗数据
            if "think" in entity_responses:
                entity_responses = re.sub(r'<think>.*?</think>', '', entity_responses, flags=re.DOTALL)
                entity_responses = entity_responses.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
                if entity_responses[0] == "\n":
                    entity_responses = entity_responses[1:]

            entity_responses = (entity_responses
                .replace('“', '"').replace('”', '"')
                .replace('，', ',').replace(']]', ']').replace('[[', '[')
                .replace("'", '"').replace('‘', '"').replace('’', '"')
                .replace("json", '').replace("```", '').replace("\\", '')
                .replace('心脏"漏跳"', '心脏漏跳')
                .replace('\\', '')
                .replace('(', ':')
                .replace(')', '')
                .replace('（', ':')
                .replace('）', '')
                .replace('可能性的疾病', '疾病')
            )
            print(entity_responses)

            # 3. 尝试解析 JSON
            entity_responses = json.loads(entity_responses, strict=False)
            if index:
                result = entity_responses["症状"]
            else:
                result = entity_responses["疾病"]
            # 如果解析成功，跳出循环
            break

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 如果解析失败或获取内容出错，打印错误并继续循环重新请求
            print(f"JSON 解析失败，正在重试... 错误信息: {e}")
            continue
    return result
