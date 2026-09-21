"""Tokenizer 加载:统一处理不同系列模型(internlm2 / orion / gemma / QWen 等)
在 special token 上的差异,并保证 pad_token / eos_token 一定可用。
"""

from loguru import logger
from transformers import AddedToken, AutoConfig, AutoTokenizer


def load_tokenizer(args):
    """加载并修正 tokenizer,使其满足训练所需的最小要求。

    Args:
        args: CustomizedArguments,需要包含 model_name_or_path。

    Returns:
        tokenizer: 已修正 pad_token / eos_token 的 AutoTokenizer。
    """
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama / internlm2 不支持 fast
        # use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
        use_fast=False,
    )

    # 部分模型的 base 与 chat 版本 tokenizer 存在差异,这里按需补 special token
    if 'internlm2' in args.model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'': 92543})
        tokenizer._added_tokens_encoder.update({'': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['', '']})
    elif 'orion' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']}
        )

    # QWenTokenizer 的特殊处理
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer