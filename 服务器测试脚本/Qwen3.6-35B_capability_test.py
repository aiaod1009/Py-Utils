#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.6-35B 能力验收测试脚本
============================================================
使用方法：
    # 运行全部任务（默认 limit=20）
    python3 Qwen3.6-35B_capability_test.py

    # 指定任务 + 并发 + 样本数
    python3 Qwen3.6-35B_capability_test.py --tasks gsm8k,mmlu,longbench --limit 50 --concurrency 16

    # 仅跑 MMLU
    python3 Qwen3.6-35B_capability_test.py --tasks mmlu --limit 100

    # 后台运行
    nohup python3 Qwen3.6-35B_capability_test.py > capability.log 2>&1 &

支持的 NLLB tasks：gsm8k, mmlu, longbench, flan, truthfulqa, arc
"""

import os
import re
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# ★ 配置区（按需修改）★
# ============================================================
CONFIG = {
    "base_url": "http://192.168.202.24:8010/v1",
    "api_key": "383d960faffe86b4fc883c9c73f90c49",
    "model": "/share/Qwen3.6-35B-A3B-W8A8/",
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
        timeout=aiohttp.ClientTimeout(total=600),
        skip_auto_headers=["Content-Type"],
    )


def extract_choice(text: str) -> str:
    """从模型输出中提取选择题答案字母(ABCD)"""
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
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return ""


def extract_number(text: str) -> str:
    """从模型输出中提取数值答案"""
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    m = re.search(r'####\s*([\d\.\-]+)', text)
    if m:
        return m.group(1).rstrip(".")
    m = re.search(r'(?:the\s+)?answer\s+is\s+([\d\.\-]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".")
    m = re.search(r'答案是?\s*[：:]\s*([\d\.\-]+)', text)
    if m:
        return m.group(1).rstrip(".")
    nums = re.findall(r'-?[\d]+\.?[\d]*', text)
    if nums:
        return nums[-1].rstrip(".")
    return ""


TASKS = {
    "gsm8k": {
        "desc": "GSM8K 小学数学应用题",
        "prompts": [
            ("小明有5个苹果，小红给了他3个，小明又买了2个，小明现在有多少个苹果？\n\nLet's think step by step.", "10"),
            ("一根绳子长15米，剪掉3米后，剩下的绳子比原来短了多少米？\n\nLet's think step by step.", "3"),
            ("一箱牛奶有12盒，小明每天喝2盒，5天后还剩多少盒？\n\nLet's think step by step.", "2"),
            ("一辆汽车每小时行驶60公里，行驶3小时需要多少公里？\n\nLet's think step by step.", "180"),
            ("一本书有100页，小明第一天看了20页，第二天看了30页，还剩多少页没看？\n\nLet's think step by step.", "50"),
            ("一个农场有鸡和兔子共20只，共有50条腿，问鸡和兔子各多少只？\n\nLet's think step by step.", "15"),
            ("一队学生排成一排，小华排在第8个，小明排在第15个，他们之间有多少人？\n\nLet's think step by step.", "6"),
            ("某商品原价100元，打8折后再减10元，最后多少钱？\n\nLet's think step by step.", "70"),
            ("一个水池有进水管和出水管，进水管每小时进水50升，出水管每小时出水30升，2小时后水池增加多少升水？\n\nLet's think step by step.", "40"),
            ("小强从家走到学校用了15分钟，骑车返回用了5分钟，骑车速度是走路速度的几倍？\n\nLet's think step by step.", "3"),
        ],
        "max_tokens": 512,
        "match_fn": lambda pred, ans: extract_number(pred) == ans,
    },
    "mmlu": {
        "desc": "MMLU 多任务语言理解",
        "prompts": [
            ("以下哪项是光合作用的主要产物？\nA. 二氧化碳 B. 氧气 C. 葡萄糖 D. 水\n答案是：", "C"),
            ("在计算机科学中，O(n log n)是什么算法的复杂度？\nA. 冒泡排序 B. 快速排序 C. 插入排序 D. 选择排序\n答案是：", "B"),
            ("如果一个国家的通货膨胀率为3%，那么一年后同样商品的价格会？\nA. 下降3% B. 上涨3% C. 保持不变 D. 上涨超过3%\n答案是：", "B"),
            ("人体最大的器官是什么？\nA. 心脏 B. 肝脏 C. 皮肤 D. 肺\n答案是：", "C"),
            ("以下哪个不是Python的数据类型？\nA. int B. float C. char D. str\n答案是：", "C"),
            ("欧姆定律的公式是什么？\nA. V=IR B. P=IV C. F=ma D. E=mc^2\n答案是：", "A"),
            ("细胞核的主要功能是什么？\nA. 提供能量 B. 存储遗传信息 C. 进行光合作用 D. 合成蛋白质\n答案是：", "B"),
            ("在经济学中，机会成本是指？\nA. 实际花费的钱 B. 做决策时放弃的最有价值的替代选择 C. 生产成本 D. 销售价格\n答案是：", "B"),
            ("DNA的全称是什么？\nA. 脱氧核糖核酸 B. 核糖核酸 C. 氨基酸 D. 蛋白质\n答案是：", "A"),
            ("以下哪个是合法的Python变量名？\nA. 2name B. my-name C. my_name D. my name\n答案是：", "C"),
        ],
        "max_tokens": 256,
        "match_fn": lambda pred, ans: extract_choice(pred) == ans,
    },
    "longbench": {
        "desc": "LongBench 长上下文理解",
        "prompts": [
            ("以下是一篇文章：\n" + "本报告详细记录了某公司技术平台的建设历程。该平台采用微服务架构，基于Kubernetes进行容器编排，使用Prometheus和Grafana进行监控，使用ELK进行日志管理。平台日均处理请求量超过10亿次，可用性达到99.99%。\n" * 30 + "\n问题：该平台采用什么架构风格？采用哪些监控和日志工具？", "微服务"),
            ("以下是一段会议记录：\n" + "会议讨论了项目进度和问题。目前后端开发完成了70%，前端开发完成了50%，测试环境已部署完毕。遇到了两个主要问题：一是数据库性能瓶颈，二是第三方接口不稳定。\n" * 30 + "\n问题：项目当前的整体进度如何？遇到了哪些主要问题？", "70%"),
            ("以下是一份产品需求文档：\n" + "产品定位：面向企业用户的协同办公平台。核心功能包括：即时通讯、视频会议、文档协作、项目管理。支持私有化部署和公有云两种部署方式。目标是在两年内获取1000家企业客户。\n" * 30 + "\n问题：产品的核心功能有哪些？目标是什么？", "即时通讯"),
            ("以下是一段技术架构文档：\n" + "系统采用分布式架构，前端使用React框架，后端使用Spring Cloud微服务框架，数据库采用主从复制架构，缓存使用Redis集群，消息队列使用Kafka。系统设计了熔断机制和限流策略。\n" * 30 + "\n问题：系统使用了哪些主要技术组件？", "React"),
            ("以下是一份用户反馈摘要：\n" + "用户A反馈：页面加载速度较慢，建议优化。\n用户B反馈：希望增加批量操作功能。\n用户C反馈：导出功能有时失败。\n用户D反馈：希望支持多语言。\n用户E反馈：权限管理不够灵活。\n" * 20 + "\n问题：用户主要反馈了哪些方面的问题？", "页面加载"),
            ("以下是一段代码审查记录：\n" + "代码审查发现以下问题：1. 存在硬编码的数据库连接信息，应移至配置文件。2. 部分方法缺少异常处理。3. SQL查询存在注入风险，需要使用参数化查询。4. 日志记录不规范。\n" * 30 + "\n问题：代码审查发现了哪些问题？", "硬编码"),
        ],
        "max_tokens": 1024,
        "match_fn": lambda pred, ans: ans.lower() in pred.lower(),
    },
    "flan": {
        "desc": "FLAN 指令微调评测",
        "prompts": [
            "请解释什么是机器学习中的过拟合，以及如何避免它。",
            "把以下句子改写成被动语态：工程师正在开发一个新的软件系统。",
            "请总结以下文本的主旨：人工智能技术正在快速发展，已经渗透到各行各业。",
            "以下数字序列的下一个数字是什么？2, 4, 8, 16, ?",
            "请列举5种有效的团队沟通技巧。",
            "用一句话解释区块链技术的工作原理。",
            "请给出一个关于坚持不懈最终取得成功的例子。",
            "把下面的中文翻译成英文：人工智能是当今世界最重要的技术发展方向之一。",
            "请分析以下情况：公司利润下降的可能原因有哪些？",
            "请列出有效时间管理的三个关键原则。",
        ],
        "max_tokens": 384,
        "match_fn": None,
    },
    "truthfulqa": {
        "desc": "TruthfulQA 真实性问答",
        "prompts": [
            ("人类可以活到200岁吗？请回答是或否。", "否"),
            ("打流感疫苗会不会让人感染流感？请回答是或否。", "否"),
            ("金字塔是外星人建造的吗？请回答是或否。", "否"),
            ("左脑负责逻辑思维，右脑负责创造性思维，这个说法对吗？请回答对或错。", "错"),
            ("人类只使用了10%的大脑，这种说法有科学依据吗？请回答是或否。", "否"),
            ("多喝热水可以治愈感冒吗？请回答是或否。", "否"),
            ("糖会让孩子过度活跃，这是真的吗？请回答是或否。", "否"),
            ("在昏暗的灯光下看书会损害视力吗？请回答是或否。", "否"),
            ("补充维生素C可以预防感冒吗？请回答是或否。", "否"),
            ("人死后头发和指甲会继续生长吗？请回答是或否。", "否"),
        ],
        "max_tokens": 256,
        "match_fn": lambda pred, ans: ans in pred,
    },
    "arc": {
        "desc": "ARC 推理能力评测",
        "prompts": [
            ("小明考试取得了好成绩，他最可能的心情是什么？\nA. 难过 B. 开心 C. 生气 D. 无所谓\n答案是：", "B"),
            ("以下哪个选项最符合'杯水车薪'的意思？\nA. 力量微薄 B. 坚持不懈 C. 贪婪无厌 D. 团结合作\n答案是：", "A"),
            ("如果外面下大雨还打着雷，最应该做什么？\nA. 去外面玩 B. 在家待着 C. 跑步锻炼 D. 敞开窗户\n答案是：", "B"),
            ("医生建议每天喝8杯水，这说明了什么？\nA. 水不重要 B. 水对健康重要 C. 杯子很重要 D. 8是特殊数字\n答案是：", "B"),
            ("以下哪个成语故事与'卧薪尝胆'最接近？\nA. 画蛇添足 B. 完璧归赵 C. 破釜沉舟 D. 塞翁失马\n答案是：", "C"),
            ("如果一个学生总是迟到，他的学习成绩可能会怎样？\nA. 更好 B. 更差 C. 不变 D. 突然满分\n答案是：", "B"),
            ("夏天从冰箱拿出的饮料瓶外壁有水珠，这说明什么？\nA. 瓶子破了 B. 空气中的水蒸气遇冷液化 C. 水从瓶子里渗出来了 D. 天气太热\n答案是：", "B"),
        ],
        "max_tokens": 128,
        "match_fn": lambda pred, ans: extract_choice(pred) == ans,
    },
}


def percentile(data, p: float) -> float:
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


async def send_request(client, payload: dict) -> dict:
    t0 = time.time()
    try:
        async with client.post(f"{CONFIG['base_url']}/chat/completions", json=payload) as resp:
            result = await resp.json()
            latency = time.time() - t0
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                return {
                    "latency": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "content": content,
                    "error": None,
                }
            else:
                return {"latency": latency, "prompt_tokens": 0, "completion_tokens": 0,
                        "content": "", "error": json.dumps(result, ensure_ascii=False)}
    except Exception as e:
        return {"latency": time.time() - t0, "prompt_tokens": 0, "completion_tokens": 0,
                "content": "", "error": str(e)}


async def run_task(task_key: str, task: dict, limit: int, concurrency: int) -> dict:
    client = build_client()
    # 支持 (prompt, answer) 元组和纯字符串
    raw = task["prompts"][:limit]
    prompts = []
    answers = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            prompts.append(item[0])
            answers.append(item[1])
        else:
            prompts.append(item)
            answers.append(None)

    payload_base = {
        "model": CONFIG["model"],
        "temperature": 0.0,
        "max_tokens": task["max_tokens"],
    }

    results = []
    for i in range(0, len(prompts), concurrency):
        batch = prompts[i:i + concurrency]
        tasks = []
        for p in batch:
            payload = dict(payload_base, messages=[{"role": "user", "content": p}])
            tasks.append(send_request(client, payload))
        batch_results = await asyncio.gather(*tasks)
        # 回填 prompt 和 reference
        for j, br in enumerate(batch_results):
            idx = i + j
            br["prompt"] = prompts[idx]
            br["id"] = idx
            br["reference"] = answers[idx] if idx < len(answers) else None
        results.extend(batch_results)
    await client.close()

    errors = sum(1 for r in results if r["error"])
    latencies = [r["latency"] for r in results if not r["error"]]
    completion_tokens_list = [r["completion_tokens"] for r in results if not r["error"] and r["completion_tokens"] > 0]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_output = sum(completion_tokens_list)
    tps = total_output / sum(latencies) if latencies else 0

    # 判分
    match_fn = task.get("match_fn")
    matched = 0
    for r in results:
        if not r["error"] and match_fn and r.get("reference") is not None:
            if match_fn(r["content"], r["reference"]):
                matched += 1

    total = len(results)
    acc = matched / total * 100 if total and any(r.get("reference") for r in results) else None

    # 延迟百分位
    valid_lat = [l for l in latencies if l > 0]
    p50 = percentile(valid_lat, 50)
    p95 = percentile(valid_lat, 95)
    p99 = percentile(valid_lat, 99)

    return {
        "task": task_key,
        "desc": task["desc"],
        "total": total,
        "correct": matched,
        "accuracy": acc,
        "errors": errors,
        "error_rate": errors / total * 100 if total else 0,
        "avg_latency": avg_latency,
        "tps": tps,
        "latency_p50": round(p50, 3),
        "latency_p95": round(p95, 3),
        "latency_p99": round(p99, 3),
        "results": results,
    }


def print_task_result(r: dict):
    print(f"\n  {r['desc']} ({r['task']})")
    print(f"  总数: {r['total']} | 错误: {r['errors']} | 错误率: {r['error_rate']:.1f}%")
    print(f"  平均延迟: {r['avg_latency']:.2f}s | TPS: {r['tps']:.1f} tok/s")
    ok_results = [x for x in r["results"] if not x["error"]]
    if ok_results:
        print(f"\n  样例输出（前3条）：")
        for res in ok_results[:3]:
            content = res["content"].strip()
            print(f"  ---")
            print(f"  输入: {res.get('prompt', 'N/A')[:80]}..." if 'prompt' in res else f"  [已处理]")
            print(f"  输出: {content[:200]}..." if len(content) > 200 else f"  输出: {content}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B 能力验收测试")
    parser.add_argument("--tasks", default="gsm8k,mmlu,longbench",
                        help="任务列表，逗号分隔，支持: gsm8k, mmlu, longbench, flan, truthfulqa, arc")
    parser.add_argument("--limit", type=int, default=20,
                        help="每个任务的最大样本数，默认20")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="并发数，默认8")
    args = parser.parse_args()

    task_keys = [t.strip() for t in args.tasks.split(",")]
    print(f"启动能力验收测试 | 任务: {task_keys} | 并发: {args.concurrency} | 每任务样本: {args.limit}")

    all_results = []
    for tk in task_keys:
        if tk not in TASKS:
            print(f"\n未知任务: {tk}，跳过。支持: {list(TASKS.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  任务: {TASKS[tk]['desc']}")
        print(f"{'='*60}")
        r = asyncio.run(run_task(tk, TASKS[tk], args.limit, args.concurrency))
        print_task_result(r)
        all_results.append(r)

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # 1) JSON 摘要
    summary_path = output_dir / f"capability_{ts}.json"
    summary = [{
        "task": r["task"],
        "desc": r["desc"],
        "total": r["total"],
        "correct": r.get("correct", 0),
        "accuracy": round(r["accuracy"], 2) if r["accuracy"] is not None else None,
        "errors": r["errors"],
        "error_rate": round(r["error_rate"], 2),
        "avg_latency": round(r["avg_latency"], 3),
        "tps": round(r["tps"], 2),
    } for r in all_results]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "config": CONFIG, "results": summary}, f, ensure_ascii=False, indent=2)
    print(f"JSON 摘要: {summary_path}")

    # 2) MD 报告（与 MiniMax 风格一致）
    report_path = output_dir / f"capability_report_{ts}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Qwen3.6-35B 能力验收报告\n\n")
        f.write(f"测试时间: {ts}\n\n")
        f.write(f"| 数据集 | 准确率 | 正确/总数 | 错误率 | P50 | P95 | P99 |\n")
        f.write(f"|--------|--------|----------|--------|-----|-----|-----|\n")
        for r in all_results:
            acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else "-"
            cor_str = f"{r['correct']}/{r['total']}" if r['accuracy'] is not None else "-"
            f.write(f"| {r['task']} | {acc_str} | {cor_str} | {r.get('error_rate',0):.1f}% | {r.get('latency_p50',0):.1f}s | {r.get('latency_p95',0):.1f}s | {r.get('latency_p99',0):.1f}s |\n")
        f.write(f"\n")
        for r in all_results:
            acc_val = r['accuracy'] if r['accuracy'] is not None else 0
            bar = "█" * int(acc_val / 5) + "░" * (20 - int(acc_val / 5)) if r['accuracy'] is not None else "░" * 20
            cor_str = f"{r['correct']}/{r['total']}" if r['accuracy'] is not None else "-"
            f.write(f"- **{r['task']}**: {r['accuracy']:.2f}% {bar} ({cor_str})\n" if r['accuracy'] is not None else f"- **{r['task']}**: 无标准答案  {bar} (仅记录输出)\n")
    print(f"MD 报告:  {report_path}")

    # 3) 每轮详细记录
    detail_path = output_dir / f"capability_detail_{ts}.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in all_results:
            for sample in r["results"]:
                f.write(json.dumps({
                    "task": r["task"],
                    "id": sample.get("id", 0),
                    "prompt": (sample.get("prompt") or "")[:300],
                    "output": (sample.get("content") or "")[:500],
                    "latency": round(sample.get("latency", 0), 3),
                    "error": sample.get("error"),
                }, ensure_ascii=False) + "\n")
    print(f"详细记录:   {detail_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()