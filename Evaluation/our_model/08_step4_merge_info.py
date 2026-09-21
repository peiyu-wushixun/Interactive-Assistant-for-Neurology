# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 8/11: 第四步 全部疾病补充信息
Slug: step4_merge_info
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
data_kouyu = pd.read_csv("result_c_desises_60.csv")
ii = 0
eval_data = []
for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['desise'] = data_kouyu['desise'][i]
    mm['input_dio'] = data_kouyu['input_dio'][i]
    mm['ca_desise'] = data_kouyu['ca_desises'][i]
    mm['merge_sym'] = data_kouyu['merge_sym'][i]
    mm['ca_desise_rag'] = data_kouyu['ca_desise_rag'][i]
    eval_data.append(mm)

# === code block 2/4 ===
async def desise_total(desise_a, ca_desise_k):
    all_diseases = desise_a + ca_desise_k
    disease_counter = Counter(all_diseases)
    repeated_diseases = [disease for disease, count in disease_counter.items() if count >= 2]
    if repeated_diseases:
        sorted_repeated = sorted(disease_counter.items(), key=lambda x: (-x[1], x[0]))
        top_three_repeated = [disease for disease, count in sorted_repeated if count >= 2]
        index = 3-len(repeated_diseases)
        all_diseases_re = list(set(all_diseases)-set(top_three_repeated))
        if len(repeated_diseases) >= 3:
            top_three_repeated = top_three_repeated[:3]
            result = top_three_repeated
        elif len(all_diseases_re)>=index:
            selected_diseases = random.sample(all_diseases, index)
            result = top_three_repeated + selected_diseases
        else:
            result = top_three_repeated + all_diseases_re
    else:
        result = list(set(all_diseases))
    return result

# === code block 3/4 ===
rag_know = []
merge_disease = []
for i in eval_data:
    des = eval_data[ii]['desise']
    input_dio = eval_data[ii]['input_dio']
    desise_a = eval_data[ii]['ca_desise'][1:][:-1].replace("'","").split(", ")
    sysmpot = eval_data[ii]['merge_sym'][1:][:-1].replace("'","").split(", ")
    ca_desise_k = eval_data[ii]['ca_desise_rag'][1:][:-1].replace("'","").split(", ")
    diseases = await desise_total(desise_a, ca_desise_k)
    merge_disease.append(diseases)
    hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_dio #医患对话历史（不包含医生的结论）
    conversation_history=[{"role": "user", "content": input_dio}]
    response = rag.query(1, hist, diseases, sysmpot, [], [], param=QueryParam(mode="ll_medical",conversation_history=conversation_history)) #二级检索
    rag_know.append(response)
    ii = ii + 1
    print(ii)

# === code block 4/4 ===
data_kouyu['merge_disease'] = merge_disease
data_kouyu['rag_know'] = rag_know
data_kouyu.to_csv('result_d_knowledge_60.csv', index=False, encoding='utf-8-sig')
