"""组装最终 Trainer:DPO -> DPOTrainer,其它 -> Trainer。

把 tokenizer / model / dataset / collator 全部连成一条线,避免 main() 太长。
"""

from loguru import logger
from transformers import Trainer
from trl import DPOTrainer

from component.collator import PretrainCollator, SFTDataCollator

from .dataset_loader import load_dpo_dataset, load_pretrain_dataset, load_sft_dataset
from .model_loader import load_model, load_unsloth_model
from .tokenizer_loader import load_tokenizer


def init_components(args, training_args):
    """初始化训练所需的全部组件,返回一个可用的 Trainer。

    Args:
        args: CustomizedArguments。
        training_args: TrainingArguments。

    Returns:
        Trainer: 已配置好 model/dataset/tokenizer/collator 的 Trainer 实例。
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载 tokenizer
    tokenizer = load_tokenizer(args)

    # 加载 model(根据是否启用 unsloth 选路径)
    if args.use_unsloth:
        components = load_unsloth_model(args, training_args)
    else:
        components = load_model(args, training_args)
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']

    # 初始化 dataset + collator
    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with dpo task')
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None

    # 根据任务类型选 trainer
    if args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    return trainer