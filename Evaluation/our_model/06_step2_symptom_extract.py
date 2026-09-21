# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 6/11: 第二步 症状疾病提取
Slug: step2_symptom_extract
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/9 ===
eval_data = []
data_kouyu = pd.read_csv("result_a_end_60.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['desise'] = data_kouyu['desise'][i]
    mm['input_dio'] = data_kouyu['input_dio'][i]
    eval_data.append(mm)

# === code block 2/9 ===
sysmpots = []

# === code block 3/9 ===
ii = 0

# === code block 4/9 ===
for i in eval_data:
    des = eval_data[ii]['desise']
    input_dio = eval_data[ii]['input_dio']
    sysmpot = await kg_sym(input_dio, index=True) #提取症状
    sysmpots.append(sysmpot)
    ii = ii + 1
    print(ii)

# === code block 5/9 ===
data_kouyu['ori_sysmpots'] = sysmpots
data_kouyu.to_csv('result_b_sysmpots_60.csv', index=False, encoding='utf-8-sig')

# === code block 6/9 ===
eval_data = []
data_kouyu = pd.read_csv("result_b_sysmpots_60.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['desise'] = data_kouyu['desise'][i]
    mm['input_dio'] = data_kouyu['input_dio'][i]
    mm['baichuan_M2_result'] = data_kouyu['baichuan_M2_result'][i]
    eval_data.append(mm)

# === code block 7/9 ===
ca_desises = []
ii = 0

# === code block 8/9 ===
for i in eval_data:
    des = eval_data[ii]['desise']
    input_dio = eval_data[ii]['input_dio']
    pre_result = eval_data[ii]['baichuan_M2_result']
    desise_a = await kg_sym(pre_result, index=False) #提取疾病
    ca_desises.append(desise_a)
    ii = ii + 1
    print(ii)

# === code block 9/9 ===
data_kouyu['ca_desises'] = ca_desises
data_kouyu.to_csv('result_b_desises_60.csv', index=False, encoding='utf-8-sig')
