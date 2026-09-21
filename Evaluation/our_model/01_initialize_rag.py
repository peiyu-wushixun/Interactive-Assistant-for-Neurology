# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 1/11: 知识图谱加载
Slug: initialize_rag
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
import os
import io
import torch
from transformers import AutoModel, AutoTokenizer,T5Tokenizer, T5ForConditionalGeneration,AutoModelForCausalLM
from fastapi import FastAPI, Depends, Response
from fastapi.responses import StreamingResponse
from res_models import ChatVO
import httpx
import json
from tqdm import trange
import ast
from gensim.models import KeyedVectors,word2vec
import itertools
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import xiangsi as xs
import requests
import pandas as pd
import logging
import asyncio
import nest_asyncio
nest_asyncio.apply()
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from typing import Any, AsyncIterator
from lightrag.operate import chunking_by_token_size
from ollama import chat
import pandas as pd
import re
import csv
from collections import Counter
import random

import warnings
warnings.filterwarnings("ignore")

# === code block 2/4 ===
# desise = list(set(knowledge['疾病名称']))
# with open('med.txt', 'w', encoding='utf-8') as f:
#     for item in desise:
#         f.write(str(item) + '\n')

# with open('med_sys.txt', 'w', encoding='utf-8') as f:
#     for i in range(knowledge.shape[0]):
#         item = knowledge.iloc[i,0] + " 亚型：" + knowledge.iloc[i,1] + " 症状：" + knowledge.iloc[i,4]
#         f.write(str(item) + '\n')

# === code block 3/4 ===
random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 默认使用0号显卡，避免Windows用户忘记修改该处
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = "../lightragpag/nerLightRAG/workapces"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


#不知道需不需要
knowledge=pd.read_csv('../data/知识图谱.csv') #疾病表格
with open('../data/med_sys.txt','r',encoding = 'UTF-8') as f:
    data_line = f.readlines()
data_line = [i[:-1] for i in data_line]
with open('../data/med.txt','r',encoding = 'UTF-8') as f:
    data_med = f.readlines()
data_med = [i[:-1] for i in data_med] #去掉/n

# === code block 4/4 ===
async def initialize_rag():
    rag = LightRAG(
            working_dir=WORKING_DIR,
            chunking_func=chunking_by_token_size,
            llm_model_func=ollama_model_complete,
            llm_model_name="baichuan-inc/baichuan-m2-32b:q4_k_m",
            # llm_model_name="deepseek-r1:32b8192",
            llm_model_max_async=4,
            llm_model_max_token_size=131072,
            llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 8192}},
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model="bge-m3:latest8192", host="http://localhost:11434"
                ),
            ),
        )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
