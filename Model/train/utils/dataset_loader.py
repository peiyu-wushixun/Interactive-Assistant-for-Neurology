"""数据集加载:pretrain / sft / dpo 三类任务的入口全部收敛在这里。

- pretrain:扫描 jsonl 文件 -> tokenize -> packing(拼接到 max_seq_length)
- sft:按 model 名字选择 ChatGLM2 / ChatGLM3 / UnifiedSFTDataset
- dpo:走 UnifiedDPODataset
"""

import os
from itertools import chain
from os.path import join

import datasets
from datasets import concatenate_datasets, load_dataset
from loguru import logger
from tqdm import tqdm

from component.collator import PretrainCollator, SFTDataCollator  # noqa: F401 (被 trainer_builder 用)
from component.dataset import (
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset,
    UnifiedSFTDataset,
)
from component.template import template_dict


def load_pretrain_dataset(training_args, args, tokenizer):
    """加载预训练 jsonl 数据,做 tokenize + 拼接打包,支持缓存复用。

    Returns:
        datasets.Dataset: 已 tokenize & group 好的训练集。
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # 扫描所有 jsonl 文件
    logger.info('Scanning all the training file...')
    files = []
    for root, _dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # 预处理所有文本:id 化 + packing
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file).replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset(
                    "json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False
                )
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={
                        k: os.path.join(tmp_cache_path, 'tokenized.arrow')
                        for k in raw_dataset
                    },
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={
                        k: os.path.join(tmp_cache_path, 'grouped.arrow')
                        for k in tokenized_dataset
                    },
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert (
                    pretrain_dataset.features.type == processed_dataset["train"].features.type
                )
                pretrain_dataset = concatenate_datasets(
                    [pretrain_dataset, processed_dataset["train"]]
                )
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_sft_dataset(args, tokenizer):
    """根据 model 名选择对应的 SFT 数据集类,完成 tokenize + 模板套用。"""
    if args.template_name not in template_dict.keys():
        raise Exception(
            f"template_name doesn't exist, all template_name: {template_dict.keys()}"
        )
    template = template_dict[args.template_name]
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(
            args.train_file, tokenizer, args.max_seq_length, template
        )
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(
            args.train_file, tokenizer, args.max_seq_length, template
        )
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(
            args.train_file, tokenizer, args.max_seq_length, template
        )
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    """加载 DPO 训练数据集(走 UnifiedDPODataset)。"""
    if args.template_name not in template_dict.keys():
        raise Exception(
            f"template_name doesn't exist, all template_name: {template_dict.keys()}"
        )
    template = template_dict[args.template_name]
    train_dataset = UnifiedDPODataset(
        args.train_file, tokenizer, args.max_seq_length, args.max_prompt_length, template
    )
    return train_dataset