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
    """对单个回答计算综合奖励分数"""
    score = 0.0

    # 1. 长度奖励 (max 2.0)
    score += _length_reward(response)

    # 2. 格式奖励 (max 2.0)
    score += _format_reward(response)

    # 3. 关键词奖励 (max 2.0)
    score += _keyword_reward(prompt, response)

    # 4. 流畅度惩罚 (max -3.0)
    score += _fluency_penalty(response)

    # 5. 完整性奖励 (max 1.0)
    score += _completeness_reward(response)

    return score


def _length_reward(response: str) -> float:
    """
    长度奖励: 鼓励 100-500 字的回答
    太短 (<50): 信息不足
    适中 (100-500): 最佳
    太长 (>800): 可能啰嗦
    """
    length = len(response)

    if length < 30:
        return -1.0
    elif length < 50:
        return 0.0
    elif length < 100:
        return 0.5
    elif length <= 500:
        return 2.0
    elif length <= 800:
        return 1.5
    else:
        return 1.0


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


def _keyword_reward(prompt: str, response: str) -> float:
    """
    关键词奖励: 回答中包含与问题相关的领域关键词
    """
    # 中科大软件学院相关关键词
    ustc_keywords = [
        '中科大', '中国科学技术大学', 'USTC', '科大',
        '软件学院', '软院', '科软',
        '苏高院', '苏州高等研究院', '苏州校区',
        '合肥', '合肥校区',
    ]

    # 学术/校园生活关键词
    academic_keywords = [
        '导师', '论文', '毕业', '实习', '就业',
        '选课', '学分', '研一', '研二', '研三',
        '宿舍', '食堂', '校区', '实验室',
        '考研', '跨考', '保研', '选调',
    ]

    score = 0.0
    response_lower = response.lower()

    # 中科大关键词匹配
    ustc_matches = sum(1 for kw in ustc_keywords if kw.lower() in response_lower)
    score += min(ustc_matches * 0.3, 1.0)

    # 学术关键词匹配
    academic_matches = sum(1 for kw in academic_keywords if kw in response)
    score += min(academic_matches * 0.2, 1.0)

    return min(score, 2.0)


def _fluency_penalty(response: str) -> float:
    """
    流畅度惩罚: 惩罚低质量回答
    - 大量重复
    - 乱码
    - 不完整句子
    """
    penalty = 0.0

    # 检查重复: 连续重复的短语
    for n in [3, 5, 10]:
        words = response
        for i in range(len(words) - 2 * n):
            if words[i:i+n] == words[i+n:i+2*n] and len(words[i:i+n].strip()) > 0:
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
            'prompt': '中科大少年班是什么？',
            'response': '中科大少年班创办于1978年，是中国高等教育改革的一面旗帜。少年班面向16周岁以下的优秀高中生和少年，通过严格的选拔考试录取。少年班的特色包括：1）因材施教，为每位学生制定个性化培养方案；2）本科阶段即可接触前沿科研；3）实行导师制，由院士和知名教授担任导师。少年班已培养出大量杰出人才。'
        },
        {
            'prompt': '中科大少年班是什么？',
            'response': '就是让小孩上大学的地方。'
        },
        {
            'prompt': '中科大的优势学科？',
            'response': '不知道不知道不知道不知道不知道'
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
        print(f'      长度: {_length_reward(case["response"]):.1f}')
        print(f'      格式: {_format_reward(case["response"]):.1f}')
        print(f'      关键词: {_keyword_reward(case["prompt"], case["response"]):.1f}')
        print(f'      流畅度: {_fluency_penalty(case["response"]):.1f}')
        print(f'      完整性: {_completeness_reward(case["response"]):.1f}')
