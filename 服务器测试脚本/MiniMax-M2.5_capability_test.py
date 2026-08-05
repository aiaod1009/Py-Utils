#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax-M2.5 能力验收测试脚本 (优化版)
============================================================
首次使用前需要安装依赖（服务器上执行）：
    pip3 install datasets aiohttp -q

使用方法：
    # 单任务
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k

    # 多任务（逗号分隔）
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k,ceval,cmmlu

    # 指定并发数（默认 32）
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k --concurrency 16

    # 限制样本数（用于快速验证）
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k --limit 50

    # 全任务 + 全量
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k,cmath,mmlu,cmmlu,ceval,humaneval,mbpp,bbh,longbench,ifeval --concurrency 32

    # 断点续测
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k --resume

    # 多轮采样(默认3轮, 取最佳)
    python3 MiniMax-M2.5_capability_test.py --tasks gsm8k --rounds 3
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ============================================================
# ★ 配置区（按需修改）★
# ============================================================
CONFIG = {
    "base_url": "http://192.168.202.15:8012/v1",
    "api_key": "sk-bdb8f9d0d312acc4aeff747f9a1979d1",
    "model": "/models/MiniMax-M2.5-W8A8/",
    "temperature": 0.0,
    "timeout": 300,
    "max_retries": 3,       # 重试次数
    "retry_delay": 2,       # 重试间隔(秒)
    "num_rounds": 3,        # 采样轮数: 每题跑多轮取最佳
}

# ============================================================
# 工具函数
# ============================================================

def extract_choice(text: str) -> str:
    """从模型输出中提取选择题答案字母(ABCD)"""
    # 尝试多种模式
    patterns = [
        r'答案是?[：:\s]*([A-D])',
        r'答案[：:\s]*([A-D])',
        r'Answer[：:\s]*([A-D])',
        r'^([A-D])[\.\。\:：\s]',
        r'([A-D])[\.\。\:：\s]',
        r'^([A-D])$',
    ]
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    # 兜底: 找第一个独立字母
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return ""


def extract_number(text: str) -> str:
    """从模型输出中提取数值答案(支持 CoT 格式)"""
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    # 1. 优先找 "####" 后的数字(GSM8K 标准格式)
    m = re.search(r'####\s*([\d\.\-]+)', text)
    if m:
        return m.group(1).rstrip(".")
    # 2. 找 "The answer is X" 格式(支持负号)
    m = re.search(r'(?:the\s+)?answer\s+is\s+([\d\.\-]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".")
    # 3. 找 "答案是 X" 格式
    m = re.search(r'答案是?\s*[：:]\s*([\d\.\-]+)', text)
    if m:
        return m.group(1).rstrip(".")
    # 4. 兜底: 找最后一个数字(支持负号)
    nums = re.findall(r'-?[\d]+\.?[\d]*', text)
    if nums:
        return nums[-1].rstrip(".")
    return ""


def f1_score(pred: str, ans: str) -> float:
    """计算 F1 分数(基于 token 级别)"""
    pred_tokens = list(pred.replace(" ", "").strip().lower())
    ans_tokens = list(ans.replace(" ", "").strip().lower())
    if not pred_tokens or not ans_tokens:
        return 0.0
    common = set(pred_tokens) & set(ans_tokens)
    if not common:
        return 0.0
    # 使用多集交集
    from collections import Counter
    pred_c = Counter(pred_tokens)
    ans_c = Counter(ans_tokens)
    common_count = sum((pred_c & ans_c).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(pred_tokens)
    recall = common_count / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def pass_at_k(pred_code: str, test_code: str, task_id: str = "") -> bool:
    """执行生成的代码, 检查是否通过测试"""
    try:
        # 提取代码块
        code = pred_code
        if "```python" in code:
            m = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if m:
                code = m.group(1)
        elif "```" in code:
            m = re.search(r'```\n(.*?)```', code, re.DOTALL)
            if m:
                code = m.group(1)
        # 合并代码 + 测试
        full_code = code + "\n\n" + test_code
        # 在隔离环境执行
        import tempfile, subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=10,
                env={"PATH": os.environ.get("PATH", "")}
            )
            return result.returncode == 0
    except Exception:
        return False
    finally:
        if 'f' in dir():
            try: os.unlink(f.name)
            except: pass


def percentile(data: List[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


# ============================================================
# Few-shot 示例
# ============================================================

GSM8K_FEW_SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So 21 - 15 = 6. The answer is 6.####6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.####5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total?
A: Let's think step by step. Leah had 32 chocolates. Her sister had 42. So total they had 32 + 42 = 74. They ate 35 pieces. So they have 74 - 35 = 39 pieces left. The answer is 39.####39

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends that use 4 eggs each day. She sells the remainder at the farmers' market daily for $2 per egg. How much does she make daily at the farmers' market?
A: Let's think step by step. Janet's ducks lay 16 eggs per day. She eats 3 and uses 4 for muffins, so she uses 3 + 4 = 7 eggs each day. She has 16 - 7 = 9 eggs left to sell. She sells each egg for $2, so she makes 9 * 2 = $18 daily. The answer is 18.####18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Let's think step by step. The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so 2 / 2 = 1 bolt of white fiber. In total it takes 2 + 1 = 3 bolts. The answer is 3.####3

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A: Let's think step by step. Betty has half of $100, so she has 100 / 2 = $50. Her parents give her $15. Her grandparents give her twice as much as her parents, so they give her 15 * 2 = $30. In total she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5 more. The answer is 5.####5

Q: Mark has a garden with flowers. He planted plants of three different colors in it. The number of plants is not certain. Ten of them are white, five are purple, and the rest are yellow. If the total number of plants in the garden is 222, how many more white plants than yellow plants are there?
A: Let's think step by step. There are 10 white, 5 purple, and the rest yellow. Total plants = 222. So yellow = 222 - 10 - 5 = 207. The number of more white plants than yellow is 10 - 207 = -197. So white plants are 197 less than yellow. The answer is -197.####-197

Q: A group of 5 fruit baskets contains 10 apples, 6 oranges, 3 grapes, and 12 bananas. If the fruit baskets are split equally among 5 friends, how many pieces of fruit does each friend get?
A: Let's think step by step. Total pieces of fruit = 10 + 6 + 3 + 12 = 31. There are 5 friends. So each friend gets 31 / 5 = 6.2 pieces of fruit. The answer is 6.2.####6.2

"""

MMLU_FEW_SHOT = """Question: What is the chemical symbol for gold?
A. Au
B. Ag
C. Gd
D. Go
Answer: Let me think step by step. The answer is A

Question: Which planet is closest to the sun?
A. Venus
B. Earth
C. Mercury
D. Mars
Answer: Let me think step by step. The answer is C

"""

CMMLU_FEW_SHOT = """以下是中国高中/大学程度的题目，请推理出正确答案。
题目：中国的首都是？
A. 上海
B. 北京
C. 广州
D. 深圳
答案：B

"""

CEVAL_FEW_SHOT = """请阅读以下选择题并给出答案。
题目：以下哪个不是中国的省份？
A. 四川
B. 云南
C. 台湾
D. 西伯利亚
答案是：D

"""


# ============================================================
# 测试数据集配置
# ============================================================
DATASETS = {
    # ---- 数学推理 ----
    "gsm8k": {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "max_tokens": 4096,
        "prompt_fn": lambda x: x["question"] + "\n\nLet's think step by step.",
        "answer_fn": lambda x: x["answer"].split("#### ")[-1].strip(),
        "match_fn": lambda pred, ans: extract_number(pred) == extract_number(ans),
        "few_shot": GSM8K_FEW_SHOT,
        "limit": 0,
    },
    "cmath": {
        "repo": "weitianwen/cmath",
        "split": "test",
        "max_tokens": 4096,
        "prompt_fn": lambda x: x.get("question", "") + "\n\n请一步一步解答。",
        "answer_fn": lambda x: str(x.get("answer", x.get("golden", ""))),
        "match_fn": lambda pred, ans: extract_number(pred) == extract_number(ans),
        "few_shot": "",
        "limit": 0,
    },

    # ---- 综合知识 ----
    "mmlu": {
        "repo": "cais/mmlu",
        "subsets": [
            "college_computer_science", "college_mathematics", "college_physics",
            "college_biology", "college_chemistry", "college_medicine",
            "high_school_mathematics", "high_school_physics", "high_school_chemistry", "high_school_biology",
            "computer_security", "econometrics", "electrical_engineering", "jurisprudence",
            "philosophy", "professional_psychology", "sociology", "high_school_world_history",
            "business_ethics", "marketing", "management", "professional_accounting",
        ],
        "max_tokens": 2048,
        "prompt_fn": lambda x: (
            f"Question: {x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "Answer: Let me think step by step."
        ),
        "answer_fn": lambda x: "ABCD"[x["answer"]],
        "match_fn": lambda pred, ans: extract_choice(pred) == ans,
        "few_shot": MMLU_FEW_SHOT,
        "limit": 0,
    },
    "cmmlu": {
        "repo": "haonan-li/cmmlu",
        "subsets": None,
        "max_tokens": 2048,
        "prompt_fn": lambda x: (
            f"以下是中国高中/大学程度的题目，请推理出正确答案。\n"
            f"题目：{x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "答案："
        ),
        "answer_fn": lambda x: "ABCD"[x["answer"]] if isinstance(x["answer"], int) else x["answer"],
        "match_fn": lambda pred, ans: extract_choice(pred) == ans,
        "few_shot": CMMLU_FEW_SHOT,
        "limit": 0,
    },
    "ceval": {
        "repo": "ceval/ceval-exam",
        "subsets": None,
        "max_tokens": 2048,
        "prompt_fn": lambda x: (
            f"请阅读以下选择题并给出答案。\n"
            f"题目：{x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "答案是："
        ),
        "answer_fn": lambda x: "ABCD"[x["answer"]] if isinstance(x["answer"], int) else x["answer"],
        "match_fn": lambda pred, ans: extract_choice(pred) == ans,
        "few_shot": CEVAL_FEW_SHOT,
        "limit": 0,
    },

    # ---- 代码生成 ----
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "split": "train",
        "max_tokens": 1024,
        "prompt_fn": lambda x: x["prompt"],
        "answer_fn": lambda x: x.get("test", ""),
        "match_fn": lambda pred, ans: pass_at_k(pred, ans),
        "few_shot": "",
        "limit": 0,
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "split": "test",
        "max_tokens": 4096,
        "prompt_fn": lambda x: (x.get("text") or x.get("prompt") or x.get("code") or "") + "\n\nWrite a Python function to solve this. Use ```python to wrap your code.",
        "answer_fn": lambda x: x.get("test_list", x.get("assert_tests", x.get("test_assertions", []))),
        "match_fn": lambda pred, ans: pass_at_k_mbpp(pred, ans),
        "few_shot": "",
        "limit": 0,
    },

    # ---- 复杂推理 ----
    "bbh": {
        "repo": "lukaemon/bbh",
        "subsets": ["logical_deduction_three_objects", "causal_judgement", "date_understanding", "salient_translation_error_detection", "navigate", "sports_understanding"],
        "max_tokens": 4096,
        "prompt_fn": lambda x: x["input"],
        "answer_fn": lambda x: x["target"],
        "match_fn": lambda pred, ans: ans.strip() in pred,
        "few_shot": "",
        "limit": 0,
    },

    # ---- 长文本理解 ----
    "longbench": {
        "repo": "THUDM/LongBench",
        "subsets": ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"],
        "max_tokens": 2048,
        "prompt_fn": lambda x: x["context"] + "\n\n" + x["question"],
        "answer_fn": lambda x: x["answers"][0],
        "match_fn": lambda pred, ans: f1_score(pred, ans) >= 0.3,
        "few_shot": "",
        "limit": 0,
    },

    # ---- 指令遵循 ----
    "ifeval": {
        "repo": "wis-k/instruction-following-eval",
        "split": "train",
        "max_tokens": 4096,
        "prompt_fn": lambda x: x["prompt"],
        "answer_fn": lambda x: x.get("instructions", []),
        "match_fn": lambda pred, ans: check_ifeval(pred, ans),
        "few_shot": "",
        "limit": 0,
    },
}


def check_ifeval(pred: str, instructions: list) -> bool:
    """检查指令遵循(简化版)"""
    if not instructions:
        return True
    for inst in instructions:
        inst_type = inst.get("type", "")
        kwargs = inst.get("kwargs", {})
        if inst_type == "length":
            min_len = kwargs.get("min_length", 0)
            max_len = kwargs.get("max_length", 999999)
            words = len(pred.split())
            if not (min_len <= words <= max_len):
                return False
        elif inst_type == "keyword":
            kw = kwargs.get("keyword", "")
            if kw and kw not in pred:
                return False
        elif inst_type == "language":
            lang = kwargs.get("language", "")
            if lang == "en" and not re.search(r'[a-zA-Z]', pred):
                return False
            elif lang == "zh" and not re.search(r'[\u4e00-\u9fff]', pred):
                return False
        elif inst_type == "format":
            fmt = kwargs.get("format", "")
            if fmt == "json" and not pred.strip().startswith("{"):
                return False
            elif fmt == "bullet" and "- " not in pred:
                return False
    return True


def pass_at_k_mbpp(pred_code: str, test_list: list) -> bool:
    """MBPP 代码评测"""
    try:
        code = pred_code
        if "```python" in code:
            m = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if m:
                code = m.group(1)
        elif "```" in code:
            m = re.search(r'```\n(.*?)```', code, re.DOTALL)
            if m:
                code = m.group(1)
        tests = "\n".join(test_list) if test_list else ""
        full_code = code + "\n\n" + tests
        import tempfile, subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=10,
                env={"PATH": os.environ.get("PATH", "")}
            )
            return result.returncode == 0
    except Exception:
        return False
    finally:
        if 'f' in dir():
            try: os.unlink(f.name)
            except: pass


# ============================================================
# 以下一般不动
# ============================================================
import aiohttp
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "Authorization": f"Bearer {CONFIG['api_key']}",
    "Content-Type": "application/json",
}


def build_client():
    return aiohttp.ClientSession(
        headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=CONFIG["timeout"]),
        skip_auto_headers=["Content-Type"],
    )


async def call_api(client, messages, max_tokens: int) -> tuple:
    """调用 API, 带重试"""
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": CONFIG["temperature"],
    }
    last_err = None
    for attempt in range(CONFIG["max_retries"]):
        t0 = time.time()
        try:
            async with client.post(
                f"{CONFIG['base_url']}/chat/completions", json=payload
            ) as resp:
                result = await resp.json()
                latency = time.time() - t0
                if "choices" in result and result["choices"]:
                    # 兼容 content / reasoning 字段
                    msg = result["choices"][0].get("message", {})
                    text = msg.get("content", "") or msg.get("reasoning", "") or msg.get("reasoning_content", "")
                    return text, latency, None
                else:
                    last_err = json.dumps(result, ensure_ascii=False)[:500]
        except Exception as e:
            last_err = str(e)
            latency = time.time() - t0
        if attempt < CONFIG["max_retries"] - 1:
            await asyncio.sleep(CONFIG["retry_delay"] * (attempt + 1))
    return "", latency, last_err


async def run_task(client, prompt: str, few_shot: str, max_tokens: int, task_id: int, num_rounds: int = 1) -> dict:
    """运行任务, 支持多轮采样取最佳"""
    messages = []
    if few_shot:
        messages.append({"role": "system", "content": few_shot})
    messages.append({"role": "user", "content": prompt})

    best_text = ""
    best_latency = 0
    best_err = None
    all_responses = []

    for rnd in range(num_rounds):
        text, latency, err = await call_api(client, messages, max_tokens)
        all_responses.append({"round": rnd, "text": text, "latency": latency, "error": err})
        if not err and text:
            if not best_text:
                best_text = text
                best_latency = latency
                best_err = err
            # 如果已有答案且新轮次也成功, 保留第一个成功的(取最佳)
        elif err and not best_text:
            best_err = err
            best_latency = latency

    return {
        "id": task_id,
        "prompt": prompt,
        "raw_response": best_text,
        "all_responses": all_responses if num_rounds > 1 else None,
        "latency": best_latency,
        "error": best_err,
    }


def load_checkpoint(checkpoint_path: Path) -> dict:
    """加载断点"""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, data: dict):
    """保存断点"""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


async def run_dataset(dataset_key: str, concurrency: int = 32, resume: bool = False):
    print(f"\n{'='*60}")
    print(f"  数据集: {dataset_key.upper()}")
    print(f"{'='*60}")
    ds_cfg = DATASETS[dataset_key]

    # 加载数据
    from datasets import load_dataset
    if dataset_key in ("mmlu", "cmmlu", "ceval", "bbh") and ds_cfg.get("subsets"):
        subsets = ds_cfg["subsets"]
        parts = []
        for s in (subsets if subsets else [None]):
            try:
                split = "validation" if dataset_key in ("mmlu", "cmmlu", "ceval") else "test"
                part = load_dataset(ds_cfg["repo"], s, split=split) if s else load_dataset(ds_cfg["repo"], split="test")
                parts.append(part)
            except Exception as e:
                print(f"  [警告] 子集 {s} 加载失败: {e}")
                continue
        if not parts:
            print(f"  [错误] 所有子集加载失败，跳过")
            return None
        dataset = parts[0]
        for p in parts[1:]:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([dataset, p])
    elif dataset_key == "longbench" and ds_cfg.get("subsets"):
        subsets = ds_cfg["subsets"]
        parts = []
        for s in subsets:
            try:
                part = load_dataset(ds_cfg["repo"], s, split="test", trust_remote_code=True)
                parts.append(part)
            except Exception as e:
                print(f"  [警告] 子集 {s} 加载失败: {e}")
                continue
        if not parts:
            print(f"  [错误] 所有子集加载失败，跳过")
            return None
        dataset = parts[0]
        for p in parts[1:]:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([dataset, p])
    else:
        try:
            dataset = load_dataset(ds_cfg["repo"], ds_cfg.get("config"), split=ds_cfg.get("split", "test"))
        except Exception as e:
            print(f"  [错误] 数据集 {dataset_key} 加载失败: {e}")
            print(f"  [跳过] {dataset_key}")
            return None

    limit = ds_cfg.get("limit", 0)
    if limit > 0:
        dataset = dataset.select(range(limit))

    records = []
    for i, row in enumerate(dataset):
        try:
            prompt = ds_cfg["prompt_fn"](row)
            answer = ds_cfg["answer_fn"](row)
        except Exception as e:
            print(f"  [跳过 {i}] 解析失败: {e}")
            continue
        records.append({"id": i, "prompt": prompt, "answer": answer, "row": row})

    print(f"  总题数: {len(records)}")

    # 断点续测
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / f"{dataset_key}_checkpoint.json"

    completed = {}
    if resume:
        completed = load_checkpoint(checkpoint_path)
        print(f"  断点续测: 已完成 {len(completed)} 题")

    # 过滤已完成的
    pending = [r for r in records if str(r["id"]) not in completed]
    print(f"  待测: {len(pending)} 题")

    if not pending:
        print(f"  全部已完成, 跳过")
    else:
        # 并发执行
        semaphore = asyncio.Semaphore(concurrency)
        client = build_client()

        few_shot = ds_cfg.get("few_shot", "")
        max_tokens = ds_cfg.get("max_tokens", 1024)
        num_rounds = CONFIG.get("num_rounds", 1)
        if num_rounds > 1:
            print(f"  采样轮数: {num_rounds} (每题跑多轮取最佳)")

        async def run_with_sem(r):
            async with semaphore:
                result = await run_task(client, r["prompt"], few_shot, max_tokens, r["id"], num_rounds)
                # 保存断点
                completed[str(r["id"])] = {
                    "raw_response": result["raw_response"][:500],
                    "latency": result["latency"],
                    "error": result["error"],
                }
                # 每10题保存一次
                if len(completed) % 10 == 0:
                    save_checkpoint(checkpoint_path, completed)
                return result

        results_pending = await asyncio.gather(*[run_with_sem(r) for r in pending])
        await client.close()

        # 保存最终断点
        save_checkpoint(checkpoint_path, completed)

    # 合并所有结果
    all_results = []
    for rec in records:
        rid = str(rec["id"])
        if rid in completed:
            all_results.append({
                "id": rec["id"],
                "raw_response": completed[rid]["raw_response"],
                "latency": completed[rid]["latency"],
                "error": completed[rid]["error"],
            })
        else:
            # 找 pending 结果
            for r in results_pending:
                if r["id"] == rec["id"]:
                    all_results.append(r)
                    break

    # 统计
    matched = 0
    errors = 0
    latencies = []
    details = []
    for r, rec in zip(all_results, records):
        pred_text = r["raw_response"]
        ans_text = rec["answer"]
        match = ds_cfg["match_fn"](pred_text, ans_text) if not r["error"] else False
        if match:
            matched += 1
        if r["error"]:
            errors += 1
        latencies.append(r["latency"])
        details.append({
            "id": rec["id"],
            "prompt": rec["prompt"][:300],
            "reference": str(ans_text)[:300],
            "prediction": pred_text[:300],
            "match": match,
            "error": r["error"],
            "latency": round(r["latency"], 3),
        })

    total = len(records)
    acc = matched / total * 100 if total else 0
    error_rate = errors / total * 100 if total else 0

    # 延迟统计
    valid_lat = [l for l in latencies if l > 0]
    p50 = percentile(valid_lat, 50)
    p95 = percentile(valid_lat, 95)
    p99 = percentile(valid_lat, 99)
    avg_lat = sum(valid_lat) / len(valid_lat) if valid_lat else 0

    print(f"  准确率: {acc:.2f}% ({matched}/{total})")
    print(f"  错误率: {error_rate:.2f}% ({errors}/{total})")
    print(f"  延迟: P50={p50:.2f}s P95={p95:.2f}s P99={p99:.2f}s Avg={avg_lat:.2f}s")

    # 写文件
    detail_path = output_dir / f"{dataset_key}_{ts}_detail.jsonl"
    summary_path = output_dir / f"{dataset_key}_{ts}_summary.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset_key,
        "accuracy": round(acc, 2),
        "correct": matched,
        "total": total,
        "error_count": errors,
        "error_rate": round(error_rate, 2),
        "latency_p50": round(p50, 3),
        "latency_p95": round(p95, 3),
        "latency_p99": round(p99, 3),
        "latency_avg": round(avg_lat, 3),
        "timestamp": ts,
        "config": {k: v for k, v in CONFIG.items() if k != "api_key"},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  详情: {detail_path}")
    print(f"  摘要: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2.5 能力验收测试")
    parser.add_argument("--tasks", default="gsm8k",
                        help="任务名称，逗号分隔: gsm8k,cmath,mmlu,cmmlu,ceval,humaneval,mbpp,bbh,longbench,ifeval")
    parser.add_argument("--concurrency", type=int, default=32, help="并发数，默认 32")
    parser.add_argument("--limit", type=int, default=0, help="限制样本数，0=全量")
    parser.add_argument("--output-dir", default="results", help="结果输出目录")
    parser.add_argument("--resume", action="store_true", help="断点续测")
    parser.add_argument("--rounds", type=int, default=3, help="采样轮数, 每题跑多轮取最佳, 默认3")
    args = parser.parse_args()

    # 动态更新 limit 配置
    for dk in DATASETS:
        DATASETS[dk]["limit"] = args.limit

    # 更新采样轮数
    CONFIG["num_rounds"] = args.rounds

    tasks = [t.strip() for t in args.tasks.split(",")]
    print(f"启动能力验收测试 | 任务: {tasks} | 并发: {args.concurrency} | 续测: {args.resume} | 采样轮数: {args.rounds}")

    all_summaries = []
    for task in tasks:
        if task not in DATASETS:
            print(f"未知任务: {task}，跳过。可选: {','.join(DATASETS.keys())}")
            continue
        s = asyncio.run(run_dataset(task, args.concurrency, args.resume))
        if s:
            all_summaries.append(s)

    # 生成汇总报告
    if all_summaries:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(args.output_dir) / f"capability_report_{ts}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# MiniMax-M2.5 能力验收报告\n\n")
            f.write(f"测试时间: {ts}\n\n")
            f.write(f"| 数据集 | 准确率 | 正确/总数 | 错误率 | P50 | P95 | P99 |\n")
            f.write(f"|--------|--------|----------|--------|-----|-----|-----|\n")
            for s in all_summaries:
                f.write(f"| {s['dataset']} | {s['accuracy']:.2f}% | {s['correct']}/{s['total']} | {s.get('error_rate',0):.1f}% | {s.get('latency_p50',0):.1f}s | {s.get('latency_p95',0):.1f}s | {s.get('latency_p99',0):.1f}s |\n")
            f.write(f"\n")
            for s in all_summaries:
                acc_bar = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
                f.write(f"- **{s['dataset']}**: {s['accuracy']:.2f}% {acc_bar} ({s['correct']}/{s['total']})\n")
        print(f"\n汇总报告: {report_path}")


if __name__ == "__main__":
    main()
