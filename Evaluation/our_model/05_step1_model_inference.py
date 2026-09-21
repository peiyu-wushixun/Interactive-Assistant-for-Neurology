# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 5/11: 第一步 初诊断
Slug: step1_model_inference
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/5 ===
eval_data = []
data_kouyu = pd.read_csv("外部验证数据集.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['desise'] = data_kouyu['desise'][i]
    mm['input_dio'] = data_kouyu['input_dio'][i]
    eval_data.append(mm)

# === code block 2/5 ===
pre_results = []
ii = 0
for i in eval_data:
    des = i['desise']
    input_dio = i['input_dio'].replace("\n\n","\n").replace("\n\n","\n")
    pre_result = await get_result(input_dio) 
    pre_results.append(pre_result)
    print(ii)
    ii = ii+1

# === code block 3/5 ===
data_kouyu['baichuan_M2_result'] = pre_results

# === code block 4/5 ===
len(pre_results)

# === code block 5/5 ===
data_kouyu.to_csv('result_a_end_60.csv', index=False, encoding='utf-8-sig')
