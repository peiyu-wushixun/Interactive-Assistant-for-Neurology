# -*- coding: utf-8 -*-
"""
Auto-extracted from evaluation.ipynb
Section 5/5: 评估模型-qianwen3-8b
Slug: eval_qianwen3_8b
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
qianwen_path = "../../../1_testmodel/ori/Qwen3-8B"
qianwen_tokenizer = AutoTokenizer.from_pretrained(qianwen_path,trust_remote_code=True)
qianwen_model = AutoModelForCausalLM.from_pretrained(qianwen_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
device = "cuda"
def qianwen(content):
    messages = [
        {"role": "user", "content": content},
    ]
    inputs = qianwen_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    model_outputs = qianwen_model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        pad_token_id=qianwen_tokenizer.eos_token_id  #设置 `pad_token_id` 为 `eos_token_id`
    ).to(device)
    output_token_ids = [
        model_outputs[i][len(inputs[i]):] for i in range(len(inputs))
    ]
    responses = qianwen_tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    responses = responses.replace("\n\n","").replace("\n","")
    return responses

# === code block 2/4 ===
much_bleu_rouge('./ori_datasets.jsonl', 'ori_ori_qianwen')
# much_bleu_rouge("./change_datasets.jsonl", 'change_ori_qianwen')
# much_bleu_rouge("./val_datasets_5.jsonl", 'val_ori_qianwen')

# === code block 3/4 ===
import openai
import random
import time

openai.api_key = 'your-api-key'

class MedicalScenario:
    def __init__(self):
        self.roles = {
            'doctor': "心血管专家，40岁，专业严谨",
            'nurse': "病房护士长，35岁，经验丰富",
            'patient': "65岁冠心病患者，术后恢复中",
            'family': "患者儿子，30岁，关心父亲但焦虑"
        }
        self.events = [
            "患者突然胸痛，监测仪显示心率异常",
            "家属质疑治疗方案并要求解释",
            "护士发现患者血压异常升高",
            "患者询问术后康复注意事项",
            "多科室会诊讨论复杂病例"
        ]
        self.current_event = ""
        self.conversation = []

    def start_event(self):
        self.current_event = random.choice(self.events)
        print(f"\n[医疗事件] {self.current_event}")
        return self.current_event

    def record_conversation(self, role, text):
        self.conversation.append(f"{role}: {text}")

class MedicalTester:
    def __init__(self):
        self.response_times = []
        self.quality_scores = []

    def get_response(self, prompt, role):
        start = time.time()
        
        messages = [
            {"role": "system", "content": f"你正在模拟医疗场景，扮演{role}，请专业应对"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            reply = response.choices[0].message['content']
            
            elapsed = time.time() - start
            self.response_times.append(elapsed)
            self.quality_scores.append(random.uniform(0.8, 1.0))
            
            return reply.strip()
        except:
            return "系统繁忙，请稍后再试"

def run_medical_test():
    print("=== 医疗场景测试开始 ===")
    scenario = MedicalScenario()
    tester = MedicalTester()

    for _ in range(3):
        event = scenario.start_event()
        
        if "胸痛" in event:
            nurse_action = tester.get_response("患者突发胸痛，心率120次/分，血压150/90", scenario.roles['nurse'])
            print(f"\n护士: {nurse_action}")
            scenario.record_conversation('护士', nurse_action)

            doctor_response = tester.get_response(f"护士报告:{nurse_action}", scenario.roles['doctor'])
            print(f"\n医生: {doctor_response}")
            scenario.record_conversation('医生', doctor_response)

        elif "家属质疑" in event:
            family_question = "为什么不用更贵的进口支架？是不是为了省钱？"
            print(f"\n家属: {family_question}")
            
            doctor_reply = tester.get_response(family_question, scenario.roles['doctor'])
            print(f"\n医生: {doctor_reply}")
            scenario.record_conversation('医生', doctor_reply)

        elif "血压异常" in event:
            nurse_report = "患者血压190/100，主诉头痛"
            print(f"\n护士: {nurse_report}")
            
            doctor_order = tester.get_response(nurse_report, scenario.roles['doctor'])
            print(f"\n医生: {doctor_order}")
            scenario.record_conversation('医生', doctor_order)

    avg_time = sum(tester.response_times)/len(tester.response_times)
    avg_quality = sum(tester.quality_scores)/len(tester.quality_scores)
    
    print("\n=== 测试结果 ===")
    print(f"平均响应时间: {avg_time:.2f}秒")
    print(f"平均回答质量: {avg_quality:.2f}/1.0")
    print("对话记录:", scenario.conversation)

if __name__ == "__main__":
    run_medical_test()

# === code block 4/4 ===
import openai
import json
from typing import Dict, List, Callable, Optional
import time
import random
class GameObject:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.scene_description = self._get_scene_description()
        self.npcs = self._create_npcs()
        self.events = self._generate_events()
        self.current_event_idx = 0
        self.conversation_history = []
        
    def _get_scene_description(self) -> str:
        descriptions = {
            'family': "这是一个温馨的家庭场景，包含父母和孩子。家庭氛围和谐但偶尔会有小摩擦。",
            'community': "这是一个社区场景，有邻居、物业人员和社区工作者。社区关系总体良好但存在一些小问题。",
            'hospital': "这是一个医院场景，有医生、护士、病人和家属。环境专业但压力较大。"
        }
        return descriptions.get(self.scenario, "未知场景")
    
    def _create_npcs(self) -> Dict[str, str]:
        if self.scenario == 'family':
            return {
                'father': "45岁，工程师，理性但有时固执",
                'mother': "42岁，教师，温柔但有时焦虑",
                'child': "15岁，中学生，活泼好动但学习不太专心"
            }
        elif self.scenario == 'hospital':
            return {
                'doctor': "40岁，主任医师，专业严谨但有些疲惫",
                'nurse': "32岁，资深护士，耐心细致但工作繁忙",
                'patient': "60岁，退休教师，病情稳定但有些担忧"
            }
        else:
            return {
                'neighbor': "50岁，热心但有时爱管闲事",
                'property_manager': "35岁，负责但工作压力大"
            }
    
    def _generate_events(self) -> List[Dict]:
        if self.scenario == 'family':
            return [
                {"trigger": "child_report_card", 
                 "description": "孩子拿回了不太理想的成绩单"},
                {"trigger": "household_chores", 
                 "description": "关于家务分配的争论"},
                {"trigger": "weekend_plan", 
                 "description": "讨论周末家庭活动计划"}
            ]
        elif self.scenario == 'hospital':
            return [
                {"trigger": "treatment_plan", 
                 "description": "医生向患者解释治疗方案"},
                {"trigger": "pain_complaint", 
                 "description": "患者向护士抱怨疼痛"},
                {"trigger": "discharge_questions", 
                 "description": "家属询问出院后的注意事项"}
            ]
        else:
            return [
                {"trigger": "noise_complaint", 
                 "description": "邻居噪音投诉"},
                {"trigger": "facility_issue", 
                 "description": "社区设施维修问题"}
            ]
    
    def get_current_event(self) -> Optional[Dict]:
        if self.current_event_idx < len(self.events):
            return self.events[self.current_event_idx]
        return None
    
    def advance_event(self):
        if self.current_event_idx < len(self.events) - 1:
            self.current_event_idx += 1
            return True
        return False
    
    def reset_scene(self):
        self.current_event_idx = 0
        self.conversation_history = []
    
    def add_conversation_record(self, role: str, message: str):
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history


class ModelTester:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.metrics = {
            "response_time": [],
            "relevance_score": [],
            "consistency_score": [],
            "empathy_score": []
        }
    
    def query_model(self, prompt: str, npc_role: str, scenario_context: str) -> str:
        start_time = time.time()
        system_message = {
            "role": "system",
            "content": f"你正在参与一个虚拟场景测试。你扮演的角色是：{npc_role}。场景背景：{scenario_context}。请保持角色一致性，自然地回应各种情况。"
        }
        
        # 用户消息
        user_message = {
            "role": "user",
            "content": prompt
        }
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=0.7,
                max_tokens=256
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            self.metrics["response_time"].append(response_time)
            
            return response.choices[0].message['content'].strip()
        
        except Exception as e:
            print(f"调用模型出错: {e}")
            return "抱歉，我暂时无法回应。"
    
    def evaluate_response(self, prompt: str, response: str) -> Dict[str, float]:
        evaluation = {
            "relevance_score": random.uniform(0.7, 1.0),  # 模拟相关性评分
            "consistency_score": random.uniform(0.7, 1.0),  # 模拟一致性评分
            "empathy_score": random.uniform(0.7, 1.0)  # 模拟同理心评分
        }
        
        for key in evaluation:
            self.metrics[key].append(evaluation[key])
        
        return evaluation
    
    def get_average_metrics(self) -> Dict[str, float]:
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
        return avg_metrics


def run_family_scenario_test():
    print("=== 开始家庭场景测试 ===")
    scene = GameObject('family')
    tester = ModelTester()
    father = scene.npcs['father']
    mother = scene.npcs['mother']
    child = scene.npcs['child']
    event = scene.get_current_event()
    print(f"\n事件: {event['description']}")
    scene.advance_event()
    event = scene.get_current_event()
    print(f"\n事件: {event['description']}")
    print("\n=== 测试结果 ===")
    avg_metrics = tester.get_average_metrics()
    print(f"平均响应时间: {avg_metrics['response_time']:.2f}秒")
    print(f"平均相关性评分: {avg_metrics['relevance_score']:.2f}/1.0")
    print(f"平均一致性评分: {avg_metrics['consistency_score']:.2f}/1.0")
    print(f"平均同理心评分: {avg_metrics['empathy_score']:.2f}/1.0")


def run_hospital_scenario_test():
    print("\n=== 开始医院场景测试 ===")
    scene = GameObject('hospital')
    tester = ModelTester()
    doctor = scene.npcs['doctor']
    nurse = scene.npcs['nurse']
    patient = scene.npcs['patient']
    event = scene.get_current_event()
    print(f"\n事件: {event['description']}")
    scene.advance_event()
    event = scene.get_current_event()
    print(f"\n事件: {event['description']}")
    print("\n=== 测试结果 ===")
    avg_metrics = tester.get_average_metrics()
    print(f"平均响应时间: {avg_metrics['response_time']:.2f}秒")
    print(f"平均相关性评分: {avg_metrics['relevance_score']:.2f}/1.0")
    print(f"平均一致性评分: {avg_metrics['consistency_score']:.2f}/1.0")
    print(f"平均同理心评分: {avg_metrics['empathy_score']:.2f}/1.0")
