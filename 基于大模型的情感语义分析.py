import csv
import json
import re
import time
import logging
import random
import argparse
from typing import List, Tuple, Dict
import ollama
from ollama import ResponseError

# —— 配置 ——
MODEL_NAME = 'deepseek-r1:8B'
DATA_PATH = 'fake_and_real_news.csv'  # Kaggle “News Detection” 数据集
RETRIES = 2
BACKOFF = 2.0

# —— 日志配置 ——
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_json_block(s: str) -> str:
    """
    1) 去掉 Ollama 输出里的 <think>…</think> 区块
    2) 去掉所有 ```json 和 ``` 围栏标记，但保留内部内容
    3) 返回剩余的完整文本（strip 前后空白）
    """
    logger.debug("原始 LLM 输出: %s", s)
    s = re.sub(r'<think>[\s\S]*?</think>', '', s)
    s = re.sub(r'```json', '', s)
    s = re.sub(r'```', '', s)
    cleaned = s.strip()
    logger.debug("清理后内容: %s", cleaned)
    return cleaned


def call_ollama(prompt: str,
                model_name: str = MODEL_NAME,
                retries: int = RETRIES,
                backoff: float = BACKOFF) -> str:
    last_err: Exception = None
    for attempt in range(retries + 1):
        try:
            logger.debug("调用 Ollama，prompt: %s", prompt)
            resp = ollama.chat(
                model=model_name,
                stream=False,
                messages=[{'role': 'user', 'content': prompt}]
            )
            logger.debug("Ollama 原始响应: %s", resp.message.content)
            return resp.message.content
        except (ResponseError, ConnectionError, OSError) as e:
            last_err = e
            wait = backoff ** attempt + random.uniform(0, backoff)
            logger.warning(
                f"Ollama 调用失败 (attempt {attempt + 1}/{retries + 1}): {e}，等待 {wait:.1f}s 后重试"
            )
            time.sleep(wait)
    logger.error("Ollama 调用多次失败: %s", last_err)
    raise last_err


def load_dataset(path: str) -> List[Tuple[str, int]]:
    data: List[Tuple[str, int]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            raw_label = row.get('label', '').strip().lower()
            if raw_label not in ('fake', 'real'):
                logger.warning("未知标签 '%s'，跳过", raw_label)
                continue
            label = 0 if raw_label == 'fake' else 1
            data.append((text, label))
    logger.info("加载数据集，共 %d 条记录", len(data))
    return data


def _safe_json_load(raw: str) -> dict:
    cleaned = clean_json_block(raw)
    if not cleaned:
        logger.error("清理后为空，原始: %s", raw)
        raise ValueError("LLM 输出中未找到 JSON 内容")
    try:
        result = json.loads(cleaned)
        logger.debug("JSON 解析结果: %s", result)
        return result
    except json.JSONDecodeError as e:
        logger.error("JSONDecodeError: %s", e)
        logger.error("原始 raw: %s", raw)
        logger.error("cleaned: %s", cleaned)
        raise ValueError(f"JSON 解码错误: {e}")


def extract_facts(text: str) -> Dict:
    prompt = f"""
1. 核心任务定义：
   - 明确指令：结构化提取新闻核心要点与关键实体。
   - 核心目标：输出一句话摘要及人物、地点、时间、事件等实体列表。

2. 背景与上下文：
   - 领域/主题：新闻文本处理、信息抽取
   - 目标受众：后续判别脚本
   - 关键背景信息：使用“Kaggle News Detection”数据集中的 text 字段
   - 项目背景：真假新闻判别的第一步

3. 角色扮演与视角：
   - AI 扮演：资深信息抽取专家
   - 口吻/风格：专业、简洁

4. 输入信息：
   - 源材料：{text}

5. 输出要求：
   - **严格** JSON
   - 字段：
     - summary：一句话摘要
     - entities：数组，每项包含 type（人物/地点/时间/事件）和 text

6. 约束与限制：
   - 仅基于给定文本，不添加额外信息
"""
    raw = call_ollama(prompt)
    return _safe_json_load(raw)


def verify_credibility(facts: Dict) -> Dict:
    facts_str = json.dumps(facts, ensure_ascii=False)
    prompt = f"""
1. 核心任务定义：
   - 明确指令：基于前面提取结果评估新闻可信度并给出理由。
   - 核心目标：输出 credibility（高/中/低）和两条理由。

2. 背景与上下文：
   - 数据来源：Kaggle News Detection 数据集
   - 上一步：已提取 summary 与 entities

3. 输入信息：
   - 提取结果 JSON：{facts_str}

4. 输出要求：
   - **严格** JSON
   - 字段：
     - credibility：高 / 中 / 低
     - reasons：数组，包含两条简短推理

5. 约束：
   - 不要给出最终标签
"""
    raw = call_ollama(prompt)
    return _safe_json_load(raw)


def decide_label(facts: Dict, verify: Dict) -> int:
    payload = {'facts': facts, 'verification': verify}
    payload_str = json.dumps(payload, ensure_ascii=False)
    prompt = f"""
1. 核心任务定义：
   - 明确指令：结合事实要点与可信度评估给出新闻真假标签。
   - 核心目标：输出 label（Fake/Real）和一句判断说明。

2. 背景与上下文：
   - 数据集字段：text + label（真/假对照）
   - 已有字段：facts 与 verification

3. 输入信息：
   {payload_str}

4. 输出要求：
   - **严格** JSON
   - 字段：
     - label：Fake 或 Real
     - explanation：一句话说明判断依据

5. 约束：
   - label 必须精确为“Fake”或“Real”
"""
    raw = call_ollama(prompt)
    result = _safe_json_load(raw)
    label = result.get('label')
    if label not in ('Fake', 'Real'):
        logger.error("意外的 label 值: %s", label)
        raise ValueError(f"未知的 label: {label}")
    return 0 if label == 'Fake' else 1


def extract_terms(text: str) -> List[Dict]:
    prompt = f"""
1. 核心任务定义：
   - 明确指令：列出新闻中的情感触发词并标注极性。
   - 核心目标：输出词汇列表及其 + / - / 0 极性。

2. 背景与上下文：
   - 数据集字段：text

3. 输入信息：
   - 新闻正文：{text}

4. 输出要求：
   - **严格** JSON 数组
   - 每项字段：
     - term：情感词
     - polarity：+（正面）、-（负面）或 0（中立）

5. 约束：
   - 仅列出明显的情感词
"""
    raw = call_ollama(prompt)
    return _safe_json_load(raw)


def overall_sentiment(terms: List[Dict]) -> Dict:
    terms_str = json.dumps(terms, ensure_ascii=False)
    prompt = f"""
1. 核心任务定义：
   - 明确指令：基于情感词列表判断整体情感倾向。
   - 核心目标：输出 sentiment（正面/中立/负面）和 rationale。

2. 背景与上下文：
   - 已列出情感词（JSON 数组）

3. 输入信息：
   - 情感词列表：{terms_str}

4. 输出要求：
   - **严格** JSON
   - 字段：
     - sentiment：正面 / 中立 / 负面
     - rationale：一句话说明依据

5. 约束：
   - sentiment 值必须精确
"""
    raw = call_ollama(prompt)
    return _safe_json_load(raw)


def classify_news(text: str) -> int:
    facts = extract_facts(text)
    verify = verify_credibility(facts)
    return decide_label(facts, verify)


def classify_with_sentiment(text: str) -> int:
    logger.info("开始情感增强判别，文本: %s", text)
    terms = extract_terms(text)
    logger.info("提取到的情感词: %s", terms)
    overall = overall_sentiment(terms)
    logger.info("整体情感分析结果: %s", overall)
    payload = {'news': text, 'sentiment_analysis': overall}
    payload_str = json.dumps(payload, ensure_ascii=False)
    prompt = f"""
1. 核心任务定义：
   - 明确指令：结合新闻内容及情感结果评估真假并给出情感影响。
   - 核心目标：输出 emotion_impact、label（Fake/Real）和 explanation。

2. 背景与上下文：
   - 数据源：Kaggle News Detection（text + label）
   - 已有结果：overall_sentiment（JSON）

3. 输入信息：
   {payload_str}

4. 输出要求：
   - **严格** JSON
   - 字段：
     - emotion_impact：一句话说明情感如何影响可信度
     - label：Fake 或 Real
     - explanation：一句话说明最终判定依据

5. 约束：
   - label 必须是“Fake”或“Real”
"""
    raw = call_ollama(prompt)
    logger.debug("情感增强原始响应(raw): %s", raw)
    result = _safe_json_load(raw)
    logger.info("情感增强解析后结果: %s", result)
    label = result.get('label')
    if label not in ('Fake', 'Real'):
        logger.error("情感判别 label 非预期: %s", label)
        raise ValueError(f"未知的 label: {label}")
    return 0 if label == 'Fake' else 1


def evaluate(preds: List[int], trues: List[int]) -> Tuple[float, float, float, int, int]:
    n = len(trues)
    overall = sum(p == t for p, t in zip(preds, trues)) / n if n else 0.0

    fake_idxs = [i for i, t in enumerate(trues) if t == 0]
    real_idxs = [i for i, t in enumerate(trues) if t == 1]

    fake_acc = sum(preds[i] == 0 for i in fake_idxs) / len(fake_idxs) if fake_idxs else 0.0
    real_acc = sum(preds[i] == 1 for i in real_idxs) / len(real_idxs) if real_idxs else 0.0

    # 新增：返回每个类的样本数
    return overall, fake_acc, real_acc, len(fake_idxs), len(real_idxs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="新闻真假判别脚本")
    parser.add_argument('--use_dataset', action='store_true')
    args = parser.parse_args()

    data = load_dataset(DATA_PATH) if args.use_dataset else [
        ("Top Trump Surrogate BRUTALLY Stabs Him In The Back: ‘He’s Pathetic’ (VIDEO) It s looking as though Republican presidential candidate Donald Trump is losing support even from within his own ranks. You know things are getting bad when even your top surrogates start turning against you, which is exactly what just happened on Fox News when Newt Gingrich called Trump  pathetic. Gingrich knows that Trump needs to keep his focus on Hillary Clinton if he even remotely wants to have a chance at defeating her. However, Trump has hurt feelings because many Republicans don t support his sexual assault against women have turned against him, including House Speaker Paul Ryan (R-WI). So, that has made Trump lash out as his own party.Gingrich said on Fox News: Look, first of all, let me just say about Trump, who I admire and I ve tried to help as much as I can. There s a big Trump and a little Trump. The little Trump is frankly pathetic. I mean, he s mad over not getting a phone call? Trump s referring to the fact that Paul Ryan didn t call to congratulate him after the debate. Probably because he didn t win despite what Trump s ego tells him.Gingrich also added: Donald Trump has one opponent. Her name is Hillary Clinton. Her name is not Paul Ryan. It s not anybody else. Trump doesn t seem to realize that the person he should be mad at is himself because he truly is his own worst enemy. This will ultimately lead to his defeat and he will have no one to blame but himself.",
         0)
    ]
    texts, labels = zip(*data)

    preds1 = [classify_news(t) for t in texts]
    preds2 = [classify_with_sentiment(t) for t in texts]
    overall1, fake_acc1, real_acc1, fake_cnt1, real_cnt1 = evaluate(preds1, labels)
    overall2, fake_acc2, real_acc2, fake_cnt2, real_cnt2 = evaluate(preds2, labels)

    logger.info("纯真假判别 完成，预测分布 Fake=%d, Real=%d", preds1.count(0), preds1.count(1))
    logger.info("情感增强判别 完成，预测分布 Fake=%d, Real=%d", preds2.count(0), preds2.count(1))

    logger.info(
        "纯真假判别 - Acc Overall: %.4f, Fake Acc: %.4f (n=%d), Real Acc: %.4f (n=%d)",
        overall1, fake_acc1, fake_cnt1, real_acc1, real_cnt1
    )
    logger.info(
        "情感增强判别 - Acc Overall: %.4f, Fake Acc: %.4f (n=%d), Real Acc: %.4f (n=%d)",
        overall2, fake_acc2, fake_cnt2, real_acc2, real_cnt2
    )
    logger.info(
        "准确率提升 - ΔOverall: %.4f, ΔFakeAcc: %.4f, ΔRealAcc: %.4f",
        overall2 - overall1,
        fake_acc2 - fake_acc1,
        real_acc2 - real_acc1
    )
