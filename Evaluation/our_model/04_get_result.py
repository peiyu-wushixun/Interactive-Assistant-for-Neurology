# -*- coding: utf-8 -*-
"""
Auto-extracted from our_model.ipynb
Section 4/11: 结果获得
Slug: get_result
Note: outputs and cell magic were dropped; source code only.
"""


# === code block 1/4 ===
async def get_result(input_dia):
    hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_dia #医患对话历史（不包含医生的结论）
    pre_result = await result_model(input_dia) #初步诊断
    # desise_a = await kg_sym(pre_result, index=False) #提取疾病
    # sysmpot = await kg_sym(input_dia, index=True) #提取症状
    # conversation_history=[{"role": "user", "content": input_dia}]
    # response = rag.query(0, hist, data_line, data_med, param=QueryParam(mode="ll_medical",conversation_history=conversation_history)) #一级检索
    # response = response.split('11111')
    # ll_keywords = response[0].split(", ") #症状2列表
    # hl_keywords = response[1].split(", ")
    # ills = response[2].split(", ") #疾病2列表
    # diseases = ills + desise_a #疾病列表
    # results = rag.query(1, hist, hl_keywords, diseases, param=QueryParam(mode="ll_medical",conversation_history=conversation_history), ll_keywords=ll_keywords)
    # response = response.replace('医生：','').replace('患者：','')
    # return pre_result,response
    return pre_result

# === code block 2/4 ===
# rag = asyncio.run(initialize_rag()) #知识图谱检索内容
# input_dia = """
# 医生：您好，请坐。不要着急，慢慢说，今天主要是身体哪里感觉不对劲，让您决定过来看看的？
# 患者：医生啊，我这会儿心里真是七上八下的。今天早上我去早市买菜，正准备掏手机扫码付钱呢，突然感觉右边胳膊一软，好像被抽了筋一样，手机啪就掉地上了。我一开始以为是没吃早饭低血糖，结果连着右边大腿也使不上劲。旁边好心人扶着我，我心里明明很清楚想跟人家道谢，但嘴巴就是不听使唤，吧啦吧啦连我自己都听不清在嘟囔什么。
# 医生：听起来确实很让人后怕，您做得对，这种情况必须马上就诊。那除了这种突发的单侧无力和说话大舌头，最近这几天，您有没有觉得头痛，或者看东西觉得视野缺了一块，甚至家里人觉得您脾气变得比平时暴躁？
# 患者：您说到脾气，我女儿前两天还跟我大吵了一架，说我最近跟吃了火药似的，一点小事就发火。头痛倒是有一点，感觉脑壳里面胀胀的。看东西的话，偶尔会觉得眼前的防盗门有点歪斜，但揉揉眼睛又好像没事了，我也就没往心里去。
# 医生：这些生活里的细节非常有参考价值。我们再往前捋捋，您之前体检或者去其他科室看病的时候，有没有医生提醒过您血压偏高，或者脑部血管有一些退化、微小病变之类的情况？
# 患者：哎，高血压这事儿确实有，得有五六年了吧。但我这人有点固执，经常是觉得脖子梗着疼了才吃一片降压药，平时就懒得管。去年体检，大夫拿着片子跟我说脑子里有什么小血管淀粉样改变之类的，我寻思着人老了机器总会老化，就没当回事。
# 医生：了解了。那在您的亲生父母或者兄弟姐妹中，有没有人在中老年阶段出现过类似的突发偏瘫，或者有严重的高血压病史？
# 患者：有的。我妈就是因为高血压一直没当回事，六十多岁的时候突然半边身子偏瘫了，在床上躺了好几年才走的。
# 医生：好的，最后再了解一下您的生活习惯。您平时的饮食口味重不重？发病前的这一两天，有没有特别劳累，或者经历了什么情绪大起大落的事情？
# 患者：我一直在徐州生活，这边的饮食习惯本身就偏重，我自己做菜也习惯多放盐。昨天晚上刚好因为点家里的琐事跟老伴大吵了一架，当时气得我直哆嗦，感觉血都往头上涌，一晚上翻来覆去没睡着，结果今天早上出去买菜就成这样了。
# """
# hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病（不超过3个）、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_dia #医患对话历史（不包含医生的结论）
# pre_result = await result_model(input_dia) #初步诊断
# desise_a = await kg_sym(index=False, pre_result) #提取疾病
# sysmpot = await kg_sym(index=True, input_dia) #提取症状

# conversation_history=[{"role": "user", "content": input_dia}]
# response = rag.query(0, hist, data_line, data_med, param=QueryParam(mode="ll_medical",conversation_history=conversation_history)) #一级检索
# response = response.split('11111')
# ll_keywords = response[0].split(", ") #症状2列表
# hl_keywords = response[1].split(", ")
# ills = response[2].split(", ") #疾病2列表

# diseases = ills + desise_a #疾病列表
# results = rag.query(1, hist, hl_keywords, diseases, param=QueryParam(mode="ll_medical",conversation_history=conversation_history), ll_keywords=ll_keywords)
# response = response.replace('医生：','').replace('患者：','')
# print(response)

# === code block 3/4 ===
# eval

# === code block 4/4 ===
rag = asyncio.run(initialize_rag()) #知识图谱检索内容
