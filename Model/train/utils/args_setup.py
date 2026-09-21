"""参数解析与全局配置。

把原始 train.py 中的命令行参数解析、HfArgumentParser 解析、训练参数保存、
随机种子设置等统一收敛到 setup_everything() 这一个函数里,保持职责单一。
"""

import argparse
import json
import os

from loguru import logger
from os.path import join
from transformers import HfArgumentParser, TrainingArguments, set_seed

from component.argument import CustomizedArguments


def setup_everything():
    """解析命令行 + JSON 配置文件,准备输出目录,落盘训练参数。

    Returns:
        tuple: (args, training_args)
            - args: CustomizedArguments,自定义训练/微调类型相关参数
            - training_args: TrainingArguments,HF 自带训练参数
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str,
    #                     default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument(
        "--train_args_file",
        type=str,
        default='train_args/sft/qlora/qwen-7b-sft-qlora.json',
        help="",
    )
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file

    # 解析得到自定义参数,以及自带参数
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))

    # 加载训练配置文件(原始 dict,用于原样落盘)
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)

    # 设置随机种子
    set_seed(training_args.seed)

    # 基本校验
    assert args.task_type in ['pretrain', 'sft', 'dpo'], \
        "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora'], \
        "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, \
        "only one of fp16 and bf16 can be True"

    return args, training_args