"""
GRPO 奖励函数定义
用于 veRL GRPO 训练，对模型生成的回答进行打分

奖励设计思路:
  1. 规则奖励 (Rule-based): 不需要 Reward Model，用规则直接打分
  2. 适合中科大问答场景: 鼓励详细、结构化、准确的回答

奖励组成:
  - 长度奖励: 鼓励适当长度的回答 (100-500 字)
  - 格式奖励: 鼓励结构化回答 (分点、编号)
  - 关键词奖励: 鼓励包含领域关键词
  - 流畅度惩罚: 惩罚重复、乱码
"""

import re
from typing import List, Dict, Any


def compute_reward(
    prompts: List[str],
    responses: List[str],
    **kwargs
) -> List[float]:
    """
    计算一批 response 的奖励分数

    Args:
        prompts: 输入的 prompt 列表
        responses: 模型生成的 response 列表

    Returns:
        rewards: 每个 response 的奖励分数列表
    """
    rewards = []
    for prompt, response in zip(prompts, responses):
        reward = _score_single(prompt, response)
        rewards.append(reward)
    return rewards


def _score_single(prompt: str, response: str) -> float:
    """
    对单个回答计算综合奖励分数

    采用乘法结构：总分 = 内容分 × 长度系数
    这样长度异常时无论内容多好，总分都会被压低，防止 reward hacking
    """
    # Step 1: 计算内容质量分（加法，各维度独立）
    content_score = 0.0
    content_score += _format_reward(response)           # max 2.0
    content_score += _keyword_reward(prompt, response)  # max 1.5
    content_score += _completeness_reward(response)     # max 1.0
    content_score += _information_density_reward(response)  # max 1.0
    content_score += _fluency_penalty(response)         # max 0, min -3.0
    # content_score 范围: [-3.0, 5.5]

    # 归一化到 [0, 1] 区间，方便乘法
    normalized_content = max(0.0, (content_score + 3.0) / 8.5)  # -3→0, 5.5→1.0

    # Step 2: 计算长度系数（乘法，直接缩放总分）
    length_multiplier = _length_multiplier(response)  # [0.0, 1.0]

    # Step 3: 总分 = 基础分 + 内容分 × 长度系数 × 缩放
    # 最佳情况: 0 + 1.0 × 1.0 × 6.0 = 6.0
    # 内容好但太长: 0 + 1.0 × 0.3 × 6.0 = 1.8
    # 内容差且太长: 0 + 0.2 × 0.3 × 6.0 = 0.36
    total = normalized_content * length_multiplier * 6.0

    return round(total, 3)


def _length_multiplier(response: str) -> float:
    """
    长度乘法系数: 作为总分的缩放因子，直接控制最终得分
    最佳区间 (100-300字): 系数 1.0，不打折
    可接受 (50-100, 300-400字): 系数 0.6-0.8
    超长/超短: 系数 0.1-0.3，无论内容多好都会被压低
    """
    length = len(response)

    if length < 30:
        return 0.1
    elif length < 50:
        return 0.3
    elif length < 100:
        return 0.6
    elif length <= 300:
        return 1.0
    elif length <= 400:
        return 0.7
    elif length <= 500:
        return 0.4
    elif length <= 600:
        return 0.2
    else:
        return 0.1


def _format_reward(response: str) -> float:
    """
    格式奖励: 鼓励结构化回答
    - 分点列举 (1. 2. 3. 或 - )
    - 分段落
    - 使用标题/小标题
    """
    score = 0.0

    # 检查是否有编号列表 (1. 2. 3. 或 一、二、三)
    numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[.、]|[一二三四五六七八九十]+[、.])', response)
    if len(numbered_pattern) >= 2:
        score += 1.0

    # 检查是否有无序列表 (- 或 • )
    bullet_pattern = re.findall(r'(?:^|\n)\s*[-•]\s', response)
    if len(bullet_pattern) >= 2:
        score += 0.5

    # 检查是否有多个段落 (换行分隔)
    paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
    if len(paragraphs) >= 3:
        score += 0.5

    return min(score, 2.0)


def _extract_entities_from_text(text):
    """从文本中动态提取关键实体（专有名词、技术术语、机构名等）"""
    entities = set()

    # 1. 英文专有名词/技术术语（2字符以上的连续英文+数字）
    english_terms = re.findall(r'[A-Za-z][A-Za-z0-9_.+-]{1,30}', text)
    stop_words = {
        "the", "and", "for", "with", "from", "that", "this", "are",
        "was", "not", "can", "will", "has", "have", "but", "also",
    }
    for term in english_terms:
        if term.lower() not in stop_words:
            entities.add(term)

    # 2. 中文专有名词：书名号/引号内的内容
    quoted = re.findall(r'[《「『"](.*?)[》」』"]', text)
    entities.update(q for q in quoted if 2 <= len(q) <= 20)

    # 3. 中文机构/地点名
    org_patterns = re.findall(
        r'[\u4e00-\u9fff]{2,10}(?:大学|学院|医院|公司|实验室|研究院|研究所|中心|平台|基金会)',
        text,
    )
    entities.update(org_patterns)

    # 4. 人名模式：xx教授、xx老师、xx学长等
    name_patterns = re.findall(
        r'([\u4e00-\u9fff]{2,4})(?:教授|老师|同学|学长|学姐|院士|博士|导师)',
        text,
    )
    entities.update(name_patterns)

    # 5. 数字+单位的事实
    num_facts = re.findall(
        r'\d+(?:年|月|天|人|分|元|%|GB|MB|门|学分|个|条|篇|次|届|期)',
        text,
    )
    entities.update(num_facts)

    return entities


# 核心高频关键词（跨主题通用词，作为额外加分项）
_CORE_KEYWORDS = {
    "中科大", "中国科学技术大学", "USTC", "科大",
    "软件学院", "软院", "科软",
    "苏高院", "苏州高等研究院", "苏州校区",
    "合肥", "合肥校区",
    "导师", "论文", "毕业", "实习", "就业",
    "选课", "学分", "宿舍", "校区",
}


def _keyword_reward(prompt: str, response: str) -> float:
    """
    关键词奖励: 动态从问题中提取关键实体 + 核心词表
    参考 evaluate_model.py 的 _extract_entities_from_text 逻辑
    """
    score = 0.0

    # 1. 动态提取：从 prompt 中提取关键实体，检查 response 是否覆盖
    prompt_entities = _extract_entities_from_text(prompt)
    if prompt_entities:
        hit_count = sum(1 for entity in prompt_entities if entity in response)
        hit_rate = hit_count / len(prompt_entities)
        score += hit_rate * 1.0  # 最高 1.0

    # 2. 核心词表：response 中出现的核心关键词（降低权重，防止堆砌）
    core_hits = sum(1 for kw in _CORE_KEYWORDS if kw in response)
    score += min(core_hits * 0.15, 0.5)  # 最高 0.5（从 1.0 降低）

    return min(score, 1.5)


def _fluency_penalty(response: str) -> float:
    """
    流畅度惩罚: 惩罚低质量回答
    - 大量重复
    - 乱码
    - 不完整句子
    """
    penalty = 0.0

    # 检查重复: 连续重复的短语 (n>=6 避免误伤正常表述如"一一对应")
    for n in [6, 10, 20]:
        for i in range(len(response) - 2 * n):
            segment = response[i:i+n]
            if segment == response[i+n:i+2*n] and len(segment.strip()) > 0:
                penalty -= 0.5
                break

    # 检查乱码: 非中文非英文非标点的字符比例
    total_chars = len(response)
    if total_chars > 0:
        valid_pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s.,;:!?，。；：！？、\-\n()（）\[\]【】""\'\'\"\"·/]')
        valid_chars = len(valid_pattern.findall(response))
        invalid_ratio = 1 - valid_chars / total_chars
        if invalid_ratio > 0.3:
            penalty -= 2.0
        elif invalid_ratio > 0.1:
            penalty -= 1.0

    return max(penalty, -3.0)


def _completeness_reward(response: str) -> float:
    """
    完整性奖励: 回答是否完整
    - 以句号/感叹号等结尾
    - 不是截断的
    """
    response = response.strip()
    if not response:
        return -1.0

    # 检查是否以完整标点结尾
    ending_chars = '。！？.!?）)】」'
    if response[-1] in ending_chars:
        return 1.0
    elif len(response) > 100:
        return 0.5
    else:
        return 0.0


def _information_density_reward(response: str) -> float:
    """
    信息密度奖励: 鼓励简洁有效的回答，惩罚注水和废话
    - 高密度: 短回答中包含较多实体/数字/专有名词
    - 低密度: 长回答中大量重复表述或空洞内容
    """
    length = len(response)
    if length < 30:
        return 0.0

    # 统计有效信息元素
    info_elements = set()
    # 数字事实
    info_elements.update(re.findall(r'\d+(?:年|月|天|人|分|元|%|门|学分|个|条)', response))
    # 英文术语
    info_elements.update(
        term for term in re.findall(r'[A-Za-z][A-Za-z0-9_.+-]{2,}', response)
        if term.lower() not in {"the", "and", "for", "with", "from", "that", "this", "are", "was", "not", "can", "will"}
    )
    # 中文专有名词（书名号/引号内）
    info_elements.update(q for q in re.findall(r'[《「](.*?)[》」]', response) if 2 <= len(q) <= 15)
    # 机构名
    info_elements.update(re.findall(r'[\u4e00-\u9fff]{2,8}(?:大学|学院|公司|实验室|研究院|中心)', response))

    density = len(info_elements) / (length / 100)  # 每 100 字的信息元素数

    if density >= 3.0:
        return 1.0
    elif density >= 2.0:
        return 0.5
    elif density >= 1.0:
        return 0.0
    else:
        return -0.5


# veRL 要求的入口函数
def reward_function(data: Dict[str, Any]) -> List[float]:
    """
    veRL 框架调用的奖励函数入口

    Args:
        data: 包含 'prompts' 和 'responses' 的字典

    Returns:
        rewards: 奖励分数列表
    """
    prompts = data.get('prompts', data.get('prompt', []))
    responses = data.get('responses', data.get('response', []))

    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]

    return compute_reward(prompts, responses)


if __name__ == '__main__':
    # 测试奖励函数
    test_cases = [
        {
            'prompt': '中科大软件学院研一选课有什么建议？',
            'response': '中科大软件学院研一选课建议如下：1）上学期优先修必修课和感兴趣的限选课，最多选18学分；2）下学期补足剩余学分，研一总共需要修满34学分；3）选课前多参考学长学姐的评价，了解课程难度和老师风格；4）注意部分课程有先修要求，提前规划好顺序。建议研一上学期不要选太多课，留出时间适应研究生生活和了解导师的研究方向。'
        },
        {
            'prompt': '苏高院宿舍条件怎么样？',
            'response': '还行吧，就那样。'
        },
        {
            'prompt': '科软就业怎么样？',
            'response': '就业就业就业就业就业就业就业就业就业就业'
        },
    ]

    print('=' * 60)
    print('  奖励函数测试')
    print('=' * 60)

    for i, case in enumerate(test_cases):
        reward = _score_single(case['prompt'], case['response'])
        print(f'\n  测试 {i+1}:')
        print(f'    问题: {case["prompt"]}')
        print(f'    回答: {case["response"][:80]}...' if len(case['response']) > 80 else f'    回答: {case["response"]}')
        print(f'    奖励: {reward:.2f}')
        print(f'      长度系数: {_length_multiplier(case["response"]):.2f}')
        print(f'      格式: {_format_reward(case["response"]):.1f}')
        print(f'      关键词: {_keyword_reward(case["prompt"], case["response"]):.1f}')
        print(f'      流畅度: {_fluency_penalty(case["response"]):.1f}')
        print(f'      完整性: {_completeness_reward(case["response"]):.1f}')
        print(f'      信息密度: {_information_density_reward(case["response"]):.1f}')
