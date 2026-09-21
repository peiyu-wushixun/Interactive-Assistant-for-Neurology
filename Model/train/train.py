"""训练主入口(精简版)。

原来 451 行的 train.py 被拆到了 utils/ 下的各个模块,这里只保留:
  1) 解析参数
  2) 组装 trainer
  3) 启动训练 & 保存

直接 `python train.py --train_args_file xxx.json` 即可。

注意:运行环境需要能 import `component.*`(collator / argument / template / dataset),
这是项目原有的依赖,本拆分不包含其实现。
"""

import os
from os.path import join

from loguru import logger

from utils.args_setup import setup_everything
from utils.trainer_builder import init_components

# 抑制 tokenizer 的 parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    # 1) 解析命令行 + JSON 配置
    args, training_args = setup_everything()

    # 2) 加载各组件并组装 trainer
    trainer = init_components(args, training_args)

    # 3) 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()

    # 4) 保存模型 & 指标
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)  # 同时会保存 tokenizer

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()