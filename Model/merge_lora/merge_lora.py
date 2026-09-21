"""LoRA 合并脚本主入口。

原来的 merge_lora.py 把路径硬编码在函数体里,而且注释了一堆其它模型的版本,
不太好维护。改成现在这样:
  - 路径集中在 CONFIG 区,一眼就能看到当前在跑哪个模型
  - 合并逻辑放到 lora_merger.py,可复用
  - 想跑别的模型,直接复制一份 CONFIG 改路径即可
"""

import torch

from lora_merger import merge_lora_to_base_model

# -------- CONFIG --------
# 想要切换到其它模型,只需改下面这三行。
# 注释里保留了原脚本里出现过的几组配置供参考:
#   Llama3-8B-Chinese-Chat     -> ../output/llama3-8b-sft-qlora920_5epoch
#   MiniCPM3-4B                -> ../output/MiniCPM3_4B_5epoch109
CONFIG = {
    "model_name_or_path":   "../../1_testmodel/ori/MiniCPM4-8B",
    "adapter_name_or_path": "../output/firefly-minicpm-8b-sft-qlora",
    "save_path":            "../output/merge-minicpm-8b-3",
}


def main():
    merge_lora_to_base_model(
        model_name_or_path=CONFIG["model_name_or_path"],
        adapter_name_or_path=CONFIG["adapter_name_or_path"],
        save_path=CONFIG["save_path"],
        device_map={'': 'cpu'},        # CPU 合并,稳;显存够也可改 'auto'
        dtype=torch.float16,
    )


if __name__ == '__main__':
    main()