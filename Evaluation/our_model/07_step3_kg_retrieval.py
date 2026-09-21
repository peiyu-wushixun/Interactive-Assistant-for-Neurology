# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 7/11: 第三步 知识图谱补充疾病
Slug: step3_kg_retrieval
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/5 ===
ii = 0
eval_data = []
data_kouyu = pd.read_csv("merge_sym_deepseek_60.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['input_dio'] = data_kouyu['input_dio'][i]
    mm['merge_sym'] = data_kouyu['merge_sym'][i]
    eval_data.append(mm)

# === code block 2/5 ===
ca_desise_k = []

# === code block 3/5 ===
ii = 0

# === code block 4/5 ===
for i in eval_data:
    input_dio = eval_data[ii]['input_dio']
    # sysmpot = eval_data[ii]['merge_sym'][1:][:-1].replace("'","").split(", ")
    sysmpot = ast.literal_eval(eval_data[ii]['merge_sym'])
    hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_dio #医患对话历史（不包含医生的结论）
    conversation_history=[{"role": "user", "content": input_dio}]
    response = rag.query(0, hist, [], sysmpot, data_line, data_med, param=QueryParam(mode="ll_medical",conversation_history=conversation_history)) #一级检索
    # print(response)
    response = response.split('11111')
    ll_keywords = response[0].split(", ") #症状2列表
    ills = response[1].split(", ") #疾病2列表
    ca_desise_k.append(ills)
    ii = ii + 1
    print(ii)

# === code block 5/5 ===
data_kouyu['ca_desise_rag'] = ca_desise_k
data_kouyu.to_csv('result_c_desises_60.csv', index=False, encoding='utf-8-sig')
