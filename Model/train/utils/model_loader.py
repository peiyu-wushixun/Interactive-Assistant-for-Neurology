"""模型加载模块。

两个入口:
- load_model:常规 HF + PEFT(LoRA / QLoRA / 全量)加载路径。
- load_unsloth_model:当环境装了 unsloth 时,用 FastLanguageModel 加速。

二者都返回统一格式的字典 {'model', 'ref_model', 'peft_config'},方便上层组装。
"""

import importlib

import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import get_kbit_device_map

from .lora_utils import find_all_linear_names

# 可选依赖:有 unsloth 就用,没有就走常规路径
if importlib.util.find_spec('unsloth') is not None:
    from unsloth import FastLanguageModel


def load_unsloth_model(args, training_args):
    """走 unsloth 加速的模型加载路径(自动套 LoRA)。"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        trust_remote_code=True,
        load_in_4bit=True if args.train_mode == 'qlora' else False,
    )
    if args.train_mode in ['lora', 'qlora']:
        logger.info('Initializing PEFT Model...')
        target_modules = find_all_linear_names(model, args.train_mode)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=training_args.seed,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f'target_modules: {target_modules}')
    return {
        'model': model,
        'ref_model': None,
        'peft_config': None,
    }


def load_model(args, training_args):
    """常规 HF 模型加载 + PEFT 注入。

    Returns:
        dict: {'model': ..., 'ref_model': ..., 'peft_config': ...}
    """
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')

    # init model kwargs
    # todo: 加 flash attention
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # 其它可选入口(已被注释的版本):
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, **model_kwargs)
    # model = AutoModel.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # MoE 模型需要把 router logits 打开,以便计算负载均衡 loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True

    # QLoRA:把非 int8 模块转换为全精度(fp32)以提高稳定性
    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft']:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    # LoRA:启用输入嵌入的梯度
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft']:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.train_mode == 'full':
        peft_config = None
    else:
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # init peft model
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(
            f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB'
        )
        model.print_trainable_parameters()

    # init ref_model(DPO 才用)
    if args.task_type == 'dpo':
        ref_model = (
            AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
            if args.train_mode == 'full' else None
        )
    else:
        ref_model = None

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config,
    }