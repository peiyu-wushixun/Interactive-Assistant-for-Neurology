"""LoRA adapter 合并到 base model 的核心逻辑。

把原来 merge_lora.py 中硬编码在函数体内的几个路径(model / adapter / / output)
外置为参数,既兼容原本写死的 MiniCPM4-8B 行为,又方便后续改配置。
"""

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def merge_lora_to_base_model(
    model_name_or_path: str,
    adapter_name_or_path: str,
    save_path: str,
    device_map=None,
    dtype=torch.float16,
):
    """把 LoRA adapter 合并回 base model,并保存到 save_path。

    Args:
        model_name_or_path: 原始基础模型路径(HF 格式)。
        adapter_name_or_path: 训练产出的 LoRA adapter 路径。
        save_path: 合并后模型的保存目录。
        device_map: torch.device_map;默认 CPU 合并(`{'': 'cpu'}`)以避免显存不够。
        dtype: 加载 base model 的精度,默认 fp16。
    """
    if device_map is None:
        device_map = {'': 'cpu'}

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama 不支持 fast
        use_fast=False if config.model_type == 'llama' else True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # 套上 LoRA adapter 并合并权重
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map=device_map)
    model = model.merge_and_unload()

    # 保存合并后的模型与 tokenizer
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)