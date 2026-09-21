# -*- coding: utf-8 -*-
"""
Auto-extracted from evaluation.ipynb
Section 2/5: 评估模型-小金刚
Slug: eval_xiaojing_orig
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
#小金刚-
chimpanzee_path = "../../../1_testmodel/ori/MiniCPM4-8B"
chimpanzee_tokenizer = AutoTokenizer.from_pretrained(chimpanzee_path,trust_remote_code=True)
chimpanzee_model = AutoModelForCausalLM.from_pretrained(chimpanzee_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
device = "cuda"
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
    output_token_ids = [
        model_outputs[i][len(inputs[i]):] for i in range(len(inputs))
    ]
    responses = chimpanzee_tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    responses = responses.replace("\n\n","").replace("\n","")
    return responses

# === code block 2/4 ===
much_bleu_rouge("./ori_datasets.jsonl", 'ori_ori_xiaojinggang')
# much_bleu_rouge("./change_datasets.jsonl", 'change_ori_xiaojinggang')
# much_bleu_rouge("./val_datasets_5.jsonl", 'val_ori_xiaojinggang')

# === code block 3/4 ===
much_bleu_rouge("./change_datasets.jsonl", 'change_ori_xiaojinggang')

# === code block 4/4 ===
much_bleu_rouge("./val_datasets_5.jsonl", 'val_ori_xiaojinggang')
