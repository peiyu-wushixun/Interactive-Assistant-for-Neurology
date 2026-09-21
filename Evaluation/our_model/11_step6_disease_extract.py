# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 11/11: 第六步 疾病提取
Slug: step6_disease_extract
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/8 ===
eval_data = []
data_kouyu = pd.read_csv("result_f_60.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['deepseek_ds'] = data_kouyu['deepseek_ds'][i]
    eval_data.append(mm)
desises_a = []
ii = 0

# === code block 2/8 ===
for i in eval_data:
    deepseek_result = eval_data[ii]['deepseek_ds']
    desise_a = await kg_sym(deepseek_result, index=False) #提取疾病
    desises_a.append(desise_a)
    ii = ii + 1
    print(ii)

# === code block 3/8 ===
mm = []
for i in desises_a:
    m = len(i)
    mm.append(m)

# === code block 4/8 ===
data_kouyu['desises_a'] = desises_a
data_kouyu['desises_num_a'] = mm
data_kouyu.to_csv('result_h_a_60.csv', index=False, encoding='utf-8-sig')

# === code block 5/8 ===
eval_data = []
data_kouyu = pd.read_csv("result_h_a_60.csv")

for i in range(data_kouyu.shape[0]):
    mm = {}
    mm['baichuan_ds'] = data_kouyu['baichuan_ds'][i]
    eval_data.append(mm)
desises_b= []
ii = 0

# === code block 6/8 ===
for i in eval_data:
    baichuan_result = eval_data[ii]['baichuan_ds']
    desise_b = await kg_sym(baichuan_result, index=False) #提取疾病
    desises_b.append(desise_b)
    ii = ii + 1
    print(ii)

# === code block 7/8 ===
mm = []
for i in desises_b:
    m = len(i)
    mm.append(m)

# === code block 8/8 ===
data_kouyu['desises_b'] = desises_b
data_kouyu['desises_num_b'] = mm
data_kouyu.to_csv('result_h_b_60.csv', index=False, encoding='utf-8-sig')
