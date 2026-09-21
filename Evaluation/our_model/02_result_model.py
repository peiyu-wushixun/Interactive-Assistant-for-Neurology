# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 2/11: 初步诊断模型
Slug: result_model
Note: outputs and cell magic were dropped; source code only.
"""


async def result_model(his):
    content = f"""
        你是一名三甲医院的医生，请根据你和用户之间的对话，向用户提供一些可能性的疾病：
        医患对话：
        {his}
    """
    response = chat(
        model='baichuan-inc/baichuan-m2-32b:q4_k_m',
        messages=[
               {'role': 'user', 'content': content}
        ]
    )
    result = response['message']['content']
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    result = result.replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
    if result[0]=="\n":
        result = result[1:]
    return result
