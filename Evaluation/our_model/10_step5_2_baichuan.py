# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 10/11: 第五.2步 疾病诊断结果（baichuan）
Slug: step5_2_baichuan
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/3 ===
data_kouyu = pd.read_csv("result_e_60.csv")
ii = 0
eval_data = []
for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['desise'] = data_kouyu['desise'][i]
    mm['input_dio'] = data_kouyu['input_dio'][i]
    mm['merge_disease'] = data_kouyu['merge_disease'][i]
    mm['merge_sym'] = data_kouyu['merge_sym'][i]
    mm['ca_desise_rag'] = data_kouyu['ca_desise_rag'][i]
    mm['rag_know'] = data_kouyu['rag_know'][i]
    eval_data.append(mm)

# === code block 2/3 ===
responses = []
for i in eval_data:
    des = eval_data[ii]['desise']
    input_dio = eval_data[ii]['input_dio']
    diseases = eval_data[ii]['merge_disease'][1:][:-1].replace("'","").split(", ")
    sysmpot = eval_data[ii]['merge_sym'][1:][:-1].replace("'","").split(", ")
    extract_context = []
    extract_context.append(eval_data[ii]['rag_know'])
    hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_dio #医患对话历史（不包含医生的结论）
    conversation_history=[{"role": "user", "content": input_dio}]
    response = rag.query(2, hist, diseases, sysmpot, extract_context, [], param=QueryParam(mode="ll_medical",conversation_history=conversation_history)) #三级检索
    responses.append(response)
    ii = ii + 1
    print(ii)

# === code block 3/3 ===
data_kouyu['baichuan_ds'] = responses
data_kouyu.to_csv('result_f_60.csv', index=False, encoding='utf-8-sig')
