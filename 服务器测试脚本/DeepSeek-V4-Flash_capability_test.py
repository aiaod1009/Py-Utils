#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-V4-Flash 能力验收测试脚本
============================================================
首次使用前需要安装依赖（服务器上执行）：
    pip3 install datasets aiohttp -q

使用方法：
    # 单任务
    python3 DeepSeek-V4-Flash_capability_test.py --tasks gsm8k

    # 多任务（逗号分隔）
    python3 DeepSeek-V4-Flash_capability_test.py --tasks gsm8k,ceval,cmmlu

    # 指定并发数（默认 32）
    python3 DeepSeek-V4-Flash_capability_test.py --tasks gsm8k --concurrency 16

    # 限制样本数（用于快速验证）
    python3 DeepSeek-V4-Flash_capability_test.py --tasks gsm8k --limit 50

    # 全任务 + 全量
    python3 DeepSeek-V4-Flash_capability_test.py --tasks gsm8k,mmlu,cmmlu,ceval,humaneval,longbench --concurrency 32
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================
# ★ 配置区（按需修改）★
# ============================================================
CONFIG = {
    "base_url": "http://192.168.202.3:8002",
    "api_key": "sk-8c946543fe47d863eda8630b9aade65f",
    "model": "/model/DeepSeek-V4-Flash",
    "max_tokens": 8192,
    "temperature": 0.0,
    "timeout": 300,
}

# ============================================================
# 测试数据集配置
# ============================================================
DATASETS = {
    "gsm8k": {
        "repo": "openai/gsm8k",
        "split": "test",
        "prompt_fn": lambda x: x["question"],
        "answer_fn": lambda x: x["answer"].split("#### ")[-1].strip(),
        "match_fn": lambda pred, ans: (
            pred.replace(",", "").replace("$", "").strip().rstrip(".")
            == ans.replace(",", "").replace("$", "").strip().rstrip(".")
        ),
        "limit": 0,
        "cot": True,
    },
    "mmlu": {
        "repo": "cais/mmlu",
        "subsets": ["college_computer_science", "college_mathematics", "mathematics", "physics", "chemistry", "biology"],
        "prompt_fn": lambda x: (
            f"Question: {x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "Answer: Let me think step by step. The answer is"
        ),
        "answer_fn": lambda x: x["answer"],
        "match_fn": lambda pred, ans: pred.strip().startswith(ans.strip()),
        "limit": 0,
        "cot": True,
    },
    "cmmlu": {
        "repo": "RUCAIBox/cmmlu",
        "subsets": None,
        "prompt_fn": lambda x: (
            f"以下是中国高中/大学程度的题目，请推理出正确答案。\n"
            f"题目：{x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "答案："
        ),
        "answer_fn": lambda x: x["answer"],
        "match_fn": lambda pred, ans: pred.strip().startswith(ans.strip()),
        "limit": 0,
        "cot": True,
    },
    "ceval": {
        "repo": "RUCAIBox/ceval",
        "subsets": None,
        "prompt_fn": lambda x: (
            f"请阅读以下选择题并给出答案。\n"
            f"题目：{x['question']}\n"
            + "".join(f"{opt}. {x['choices'][i]}\n" for i, opt in enumerate("ABCD"))
            + "答案是："
        ),
        "answer_fn": lambda x: x["answer"],
        "match_fn": lambda pred, ans: pred.strip().startswith(ans.strip()),
        "limit": 0,
        "cot": True,
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "split": "test",
        "prompt_fn": lambda x: x["prompt"],
        "answer_fn": lambda x: x["canonical_solution"],
        "match_fn": lambda pred, ans: True,
        "limit": 0,
        "cot": False,
    },
    "longbench": {
        "repo": "THUDM/LongBench",
        "subsets": ["qasper", "multi_field_qa", "hotpotqa", "2wiki_mqa", "musique"],
        "prompt_fn": lambda x: x["context"] + "\n\n" + x["question"],
        "answer_fn": lambda x: x["answers"][0],
        "match_fn": lambda pred, ans: (
            pred.replace(" ", "").strip()[:50] == ans.replace(" ", "").strip()[:50]
        ),
        "limit": 0,
        "cot": False,
    },
}

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


async def call_api(client, messages, task_id: int) -> tuple[str, float, Optional[str]]:
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "max_tokens": CONFIG["max_tokens"],
        "temperature": CONFIG["temperature"],
    }
    t0 = time.time()
    try:
        # vLLM V1 引擎标准路径
        async with client.post(f"{CONFIG['base_url']}/v1/chat/completions", json=payload) as resp:
            result = await resp.json()
            latency = time.time() - t0
            if "choices" in result and result["choices"]:
                text = result["choices"][0]["message"]["content"]
                return text, latency, None
            else:
                return "", latency, json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return "", time.time() - t0, str(e)


async def run_task(client, prompt: str, task_id: int, cot: bool) -> dict:
    messages = [{"role": "user", "content": prompt}]
    text, latency, err = await call_api(client, messages, task_id)
    return {
        "id": task_id,
        "prompt": prompt,
        "raw_response": text,
        "latency": latency,
        "error": err,
    }


async def run_dataset(dataset_key: str, concurrency: int = 32):
    print(f"\n{'='*60}")
    print(f"  数据集: {dataset_key.upper()}")
    print(f"{'='*60}")
    ds_cfg = DATASETS[dataset_key]

    # 加载数据
    from datasets import load_dataset
    if dataset_key in ("mmlu", "cmmlu", "ceval") and ds_cfg.get("subsets"):
        subsets = ds_cfg["subsets"]
        parts = []
        for s in subsets:
            try:
                part = load_dataset(ds_cfg["repo"], s, split="val")
                keep_cols = [c for c in part.column_names if c in ["question", "answer", "choices"]]
                parts.append(part.select_columns(keep_cols))
            except Exception as e:
                print(f"  [警告] 子集 {s} 加载失败: {e}")
                continue
        if not parts:
            print(f"  [错误] 所有子集加载失败，跳过")
            return None
        dataset = parts[0]
        for p in parts[1:]:
            dataset = dataset.concatenate_datasets([dataset, p])
    elif dataset_key == "longbench" and ds_cfg.get("subsets"):
        subsets = ds_cfg["subsets"]
        parts = []
        for s in subsets:
            try:
                part = load_dataset(ds_cfg["repo"], s, split="test")
                keep_cols = [c for c in part.column_names if c in ["context", "question", "answers"]]
                parts.append(part.select_columns(keep_cols))
            except Exception as e:
                print(f"  [警告] 子集 {s} 加载失败: {e}")
                continue
        if not parts:
            print(f"  [错误] 所有子集加载失败，跳过")
            return None
        dataset = parts[0]
        for p in parts[1:]:
            dataset = dataset.concatenate_datasets([dataset, p])
    else:
        try:
            dataset = load_dataset(ds_cfg["repo"], split=ds_cfg.get("split", "test"))
        except Exception as e:
            print(f"  [错误] 数据集 {dataset_key} 加载失败: {e}")
            print(f"  [提示] 服务器可能无法访问 HuggingFace，换个能联网的机器试试")
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

    # 并发执行
    semaphore = asyncio.Semaphore(concurrency)
    client = build_client()

    async def run_with_sem(r):
        async with semaphore:
            return await run_task(client, r["prompt"], r["id"], ds_cfg.get("cot", True))

    results = await asyncio.gather(*[run_with_sem(r) for r in records])
    await client.close()

    # 统计
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    matched = 0
    details = []
    for r, rec in zip(results, records):
        pred_text = r["raw_response"]
        ans_text = rec["answer"]
        match = ds_cfg["match_fn"](pred_text, ans_text) if not r["error"] else False
        if match:
            matched += 1
        details.append({
            "id": rec["id"],
            "prompt": rec["prompt"][:200],
            "reference": ans_text[:200],
            "prediction": pred_text[:200],
            "match": match,
            "error": r["error"],
            "latency": r["latency"],
        })

    acc = matched / len(records) * 100 if records else 0
    print(f"  准确率: {acc:.2f}% ({matched}/{len(records)})")

    # 写文件
    detail_path = output_dir / f"{dataset_key}_{ts}_detail.jsonl"
    summary_path = output_dir / f"{dataset_key}_{ts}_summary.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset_key,
        "accuracy": acc,
        "correct": matched,
        "total": len(records),
        "timestamp": ts,
        "config": {k: v for k, v in CONFIG.items() if k != "api_key"},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  详情: {detail_path}")
    print(f"  摘要: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash 能力验收测试")
    parser.add_argument("--tasks", default="gsm8k",
                        help="任务名称，逗号分隔，如: gsm8k,ceval,cmmlu,mmlu,humaneval,longbench")
    parser.add_argument("--concurrency", type=int, default=32, help="并发数，默认 32")
    parser.add_argument("--limit", type=int, default=0, help="限制样本数，0=全量，默认 0")
    parser.add_argument("--output-dir", default="results", help="结果输出目录，默认 results")
    args = parser.parse_args()

    # 动态更新 limit 配置
    for dk in DATASETS:
        DATASETS[dk]["limit"] = args.limit

    tasks = args.tasks.split(",")
    print(f"启动能力验收测试 | 任务: {tasks} | 并发: {args.concurrency}")

    all_summaries = []
    for task in tasks:
        task = task.strip()
        if task not in DATASETS:
            print(f"未知任务: {task}，跳过")
            continue
        s = asyncio.run(run_dataset(task, args.concurrency))
        if s:
            all_summaries.append(s)

    # 生成汇总报告
    if all_summaries:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(args.output_dir) / f"capability_report_{ts}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# DeepSeek-V4-Flash 能力验收报告\n\n")
            f.write(f"测试时间: {ts}\n\n")
            for s in all_summaries:
                acc_bar = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
                f.write(f"- **{s['dataset']}**: {s['accuracy']:.2f}% {acc_bar} ({s['correct']}/{s['total']})\n")
        print(f"\n汇总报告: {report_path}")


if __name__ == "__main__":
    main()
