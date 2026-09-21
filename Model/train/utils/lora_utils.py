"""LoRA 相关的辅助工具。

目前只放一个 find_all_linear_names:在给定的模型里扫描所有线性层(4-bit 或全精度),
把 LoRA 需要打到的 target module 名字汇总成列表。
"""

import torch.nn as nn

import bitsandbytes as bnb
from loguru import logger


def find_all_linear_names(model, train_mode):
    """找出模型中所有可注入 LoRA adapter 的线性层模块名。

    Args:
        model: 预训练 HF 模型。
        train_mode: 'lora' 或 'qlora',决定扫描的层类型。

    Returns:
        list[str]: 线性层的最顶层或最底层名称列表(已去除 lm_head)。
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 取最顶层或最底层的名称
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head 一般不参与 LoRA
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    lora_module_names = list(lora_module_names)
    # 如需手动指定,可以写死:
    # lora_module_names = ['q_proj', 'v_proj', 'W_pack']
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names