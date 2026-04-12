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
    references: List[str] = None,
    **kwargs
) -> List[float]:
    """
    计算一批 response 的奖励分数

    Args:
        prompts: 输入的 prompt 列表
        responses: 模型生成的 response 列表
        references: 参考答案列表（可选，有则计算相似度奖励和相对长度比）

    Returns:
        rewards: 每个 response 的奖励分数列表
    """
    rewards = []
    if references is None:
        references = [""] * len(prompts)
    for prompt, response, reference in zip(prompts, responses, references):
        reward = _score_single(prompt, response, reference)
        rewards.append(reward)
    return rewards


def _score_single(prompt: str, response: str, reference: str = "") -> float:
    """
    对单个回答计算综合奖励分数

    采用乘法结构：总分 = 内容分 × 长度系数
    长度系数优先使用相对长度比（相对于参考答案），无参考答案时用绝对长度
    """
    # Step 1: 计算内容质量分（加法，各维度独立）
    content_score = 0.0
    content_score += _format_reward(response)                       # max 2.0
    content_score += _keyword_reward(prompt, response, reference)   # max 1.5
    content_score += _completeness_reward(response)                 # max 1.0
    content_score += _information_density_reward(response)          # max 1.0
    content_score += _fluency_penalty(response)                     # max 0, min -3.0

    # 有参考答案时，加入相似度奖励和事实命中率奖励
    if reference:
        content_score += _similarity_reward(reference, response)    # max 3.0
        content_score += _fact_hit_reward(reference, response)      # max 1.5
    # content_score 范围: [-3.0, 10.0]（有参考答案时）
    # 归一化到 [0, 1] 区间
    max_content = 10.0 if reference else 5.5
    normalized_content = max(0.0, (content_score + 3.0) / (max_content + 3.0))

    # Step 2: 计算长度系数（乘法，直接缩放总分）
    if reference:
        length_multiplier = _relative_length_multiplier(reference, response)
    else:
        length_multiplier = _length_multiplier(response)

    # Step 3: 总分 = 内容分 × 长度系数 × 缩放
    total = normalized_content * length_multiplier * 6.0

    return round(total, 3)


def _relative_length_multiplier(reference: str, response: str) -> float:
    """
    相对长度系数: 基于生成长度与参考答案长度的比值
    和评估函数 compute_length_ratio 的标准一致: 0.3x~3.0x 为合理范围
    """
    ref_len = len(reference)
    gen_len = len(response)
    if ref_len == 0:
        return _length_multiplier(response)

    ratio = gen_len / ref_len

    if ratio < 0.3:
        return 0.2   # 太短
    elif ratio < 0.5:
        return 0.5
    elif ratio <= 2.0:
        return 1.0   # 最佳区间
    elif ratio <= 2.5:
        return 0.7
    elif ratio <= 3.0:
        return 0.4
    elif ratio <= 4.0:
        return 0.2
    else:
        return 0.1   # 严重超长


def _similarity_reward(reference: str, response: str) -> float:
    """
    相似度奖励: 计算生成回答与参考答案的词级别重叠度
    和评估函数 compute_rouge_l 的逻辑对齐，直接优化 ROUGE-L 指标
    """
    if not reference.strip() or not response.strip():
        return 0.0

    # 中文 bigram 分词（和 evaluate_model.py 的 _tokenize_chinese 一致）
    ref_tokens = _tokenize_for_reward(reference)
    resp_tokens = _tokenize_for_reward(response)

    if not ref_tokens or not resp_tokens:
        return 0.0

    overlap = ref_tokens & resp_tokens
    precision = len(overlap) / len(resp_tokens)
    recall = len(overlap) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # 将 F1 映射到奖励分数: 0~0.15→0, 0.15~0.3→1.0, 0.3~0.5→2.0, 0.5+→3.0
    if f1 >= 0.5:
        return 3.0
    elif f1 >= 0.3:
        return 2.0
    elif f1 >= 0.15:
        return 1.0
    else:
        return 0.0


def _tokenize_for_reward(text: str) -> set:
    """简易中文分词: 按字符 bigram 切分，兼顾英文单词完整性"""
    tokens = []
    english_buffer = []
    for char in text:
        if char.isascii() and char.isalnum():
            english_buffer.append(char)
        else:
            if english_buffer:
                tokens.append("".join(english_buffer))
                english_buffer = []
            if char.strip():
                tokens.append(char)
    if english_buffer:
        tokens.append("".join(english_buffer))
    # 生成 bigram
    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add(tokens[i] + tokens[i + 1])
    return set(tokens) | bigrams


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


def _keyword_reward(prompt: str, response: str, reference: str = "") -> float:
    """
    关键词奖励: 和评估函数 compute_keyword_hit_rate 逻辑对齐
    有参考答案时：从参考答案提取实体 + 核心词表中出现在参考答案里的
    无参考答案时：从 prompt 提取实体 + 核心词表
    """
    score = 0.0

    if reference:
        # 和评估逻辑一致：核心词表中出现在参考答案里的 + 从参考答案动态提取的实体
        core_hits_in_ref = {kw for kw in _CORE_KEYWORDS if kw in reference}
        dynamic_entities = _extract_entities_from_text(reference)
        all_keywords = core_hits_in_ref | dynamic_entities

        if all_keywords:
            hit_count = sum(1 for kw in all_keywords if kw in response)
            hit_rate = hit_count / len(all_keywords)
            score = hit_rate * 1.5  # 最高 1.5
    else:
        # 无参考答案时，从 prompt 提取
        prompt_entities = _extract_entities_from_text(prompt)
        if prompt_entities:
            hit_count = sum(1 for entity in prompt_entities if entity in response)
            hit_rate = hit_count / len(prompt_entities)
            score += hit_rate * 1.0

        core_hits = sum(1 for kw in _CORE_KEYWORDS if kw in response)
        score += min(core_hits * 0.15, 0.5)

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


def _fact_hit_reward(reference: str, response: str) -> float:
    """
    事实命中率奖励: 和评估函数 compute_fact_hit_rate 逻辑对齐
    从参考答案中提取关键事实（URL/邮箱/数字+单位/机构/专有名词），
    检查生成回答是否包含这些事实
    """
    if not reference.strip():
        return 0.0

    facts = set()

    # URL
    facts.update(re.findall(r'https?://\S+', reference))
    # 邮箱
    facts.update(re.findall(r'[\w.]+@[\w.]+', reference))
    # 数量事实
    facts.update(re.findall(r'\d{4}年|\d+%|\d+学分|\d+门|\d+人', reference))
    # 机构/地点
    facts.update(re.findall(r'(?:任职于|就职于|来自|位于|地址[是为]?)\s*(\S{2,15})', reference))
    # 引号内的专有名词
    facts.update(q for q in re.findall(r'[「『"](.*?)[」』"]', reference) if len(q) >= 2)

    if not facts:
        return 0.0

    hit_count = sum(1 for fact in facts if fact in response)
    hit_rate = hit_count / len(facts)

    # 映射到奖励: 命中率 × 1.5
    return round(hit_rate * 1.5, 3)


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
            'response': '中科大软件学院研一选课建议如下：1）上学期优先修必修课和感兴趣的限选课，最多选18学分；2）下学期补足剩余学分，研一总共需要修满34学分；3）选课前多参考学长学姐的评价，了解课程难度和老师风格；4）注意部分课程有先修要求，提前规划好顺序。建议研一上学期不要选太多课，留出时间适应研究生生活和了解导师的研究方向。',
            'reference': '研一选课建议：1）优先修必修课，最多选18学分；2）研一需修满34学分；3）参考学长学姐评价选课；4）注意先修要求。',
        },
        {
            'prompt': '苏高院宿舍条件怎么样？',
            'response': '还行吧，就那样。',
            'reference': '苏州高等研究院宿舍为4人间，配有独立卫浴、空调、热水器，每层有公共洗衣房。住宿费约1200元/年。',
        },
        {
            'prompt': '科软就业怎么样？',
            'response': '就业就业就业就业就业就业就业就业就业就业',
            'reference': '中科大软件学院就业率超过98%，主要去向为互联网大厂如阿里、腾讯、字节跳动等，平均年薪约30万。',
        },
    ]

    print('=' * 60)
    print('  奖励函数测试')
    print('=' * 60)

    for i, case in enumerate(test_cases):
        ref = case.get('reference', '')
        reward = _score_single(case['prompt'], case['response'], ref)
        print(f'\n  测试 {i+1}:')
        print(f'    问题: {case["prompt"]}')
        print(f'    回答: {case["response"][:80]}...' if len(case['response']) > 80 else f'    回答: {case["response"]}')
        if ref:
            print(f'    参考: {ref[:60]}...' if len(ref) > 60 else f'    参考: {ref}')
        print(f'    奖励: {reward:.2f}')
        if ref:
            print(f'      相对长度系数: {_relative_length_multiplier(ref, case["response"]):.2f}')
            print(f'      相似度: {_similarity_reward(ref, case["response"]):.1f}')
            print(f'      事实命中: {_fact_hit_reward(ref, case["response"]):.2f}')
        else:
            print(f'      绝对长度系数: {_length_multiplier(case["response"]):.2f}')
        print(f'      格式: {_format_reward(case["response"]):.1f}')
        print(f'      关键词: {_keyword_reward(case["prompt"], case["response"], ref):.1f}')
        print(f'      流畅度: {_fluency_penalty(case["response"]):.1f}')
        print(f'      完整性: {_completeness_reward(case["response"]):.1f}')
        print(f'      信息密度: {_information_density_reward(case["response"]):.1f}')
