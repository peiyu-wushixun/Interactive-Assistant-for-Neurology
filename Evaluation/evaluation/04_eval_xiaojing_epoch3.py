# -*- coding: utf-8 -*-
"""
Auto-extracted from evaluation.ipynb
Section 4/5: 评估模型-小金刚微调版本-3
Slug: eval_xiaojing_epoch3
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#小金刚-微调之后
chimpanzee_path = "../../../4_LoraFireflymaster/output/merge-minicpm-8b-3"
chimpanzee_tokenizer = AutoTokenizer.from_pretrained(chimpanzee_path,trust_remote_code=True)
chimpanzee_model = AutoModelForCausalLM.from_pretrained(chimpanzee_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
def qianwen(content):
    messages = [
        {"role": "user", "content": content},
    ]
    inputs = chimpanzee_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    model_outputs = chimpanzee_model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        pad_token_id=chimpanzee_tokenizer.eos_token_id  #设置 `pad_token_id` 为 `eos_token_id`
    ).to(device)
    # 解码生成的文本
    output_token_ids = [
        model_outputs[i][len(inputs[i]):] for i in range(len(inputs))
    ]
    responses = chimpanzee_tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    responses = responses.replace("\n\n","").replace("\n","")
    if "医生：" in responses:
        responses = responses.split("医生：")[1]
    return responses

# === code block 2/4 ===
much_bleu_rouge('./ori_datasets.jsonl', 'ori_epoch3_xiaojinggang')
# much_bleu_rouge("./change_datasets.jsonl", 'change_epoch3_xiaojinggang')
# much_bleu_rouge("./val_datasets_5.jsonl", 'val_epoch3_xiaojinggang')

# === code block 3/4 ===
much_bleu_rouge("./change_datasets.jsonl", 'change_epoch3_xiaojinggang')

# === code block 4/4 ===
much_bleu_rouge("./val_datasets_5.jsonl", 'val_epoch3_xiaojinggang')
