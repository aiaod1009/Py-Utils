"""
对比测试脚本 - 中转(sts-token) vs 上游(宇讯/tokeneasy)

在 4K / 16K / 64K / 128K 四种上下文长度下，对两个端点进行对比测试，
统计 TTFT、TPOT、TPS，并评估任务完成度，生成 Markdown + JSON 报告。

用法: python compare_test.py
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict

import requests

# ============================================================
# 端点配置
# ============================================================
ENDPOINTS: dict[str, dict] = {
    "中转(sts-token)": {
        "base_url": "https://www.sts-token.com/v1",
        "api_key": "sk-fd95ae896fba619b9890674c8cabaaa93b0474f855b258c7ae5a98471970d1c8",
        "model": "DeepSeek-V4-Pro",
    },
    "上游(宇讯/tokeneasy)": {
        "base_url": "https://api.tokeneasy.ai",
        "api_key": "sk-te-v1-tcd7Yot4bvhceYMjdZCnWtdNdu4wl6Tu78fi50wu",
        "model": "deepSeek-V4-Pro-0813",
    },
}

# ============================================================
# 测试场景
# ============================================================
CONDITIONS: list[dict] = [
    {"name": "4K",   "input_tokens": 3072,  "max_tokens": 1024, "samples": 3},
    {"name": "16K",  "input_tokens": 14000, "max_tokens": 2048, "samples": 3},
    {"name": "64K",  "input_tokens": 62000, "max_tokens": 2048, "samples": 2},
    {"name": "128K", "input_tokens": 126000,"max_tokens": 2048, "samples": 2},
]

# ============================================================
# 背景文本生成 (复用 full_test.py 的中文语料, ~1 token ≈ 1.4 chars)
# ============================================================
BASE_TEXT = (
    "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，"
    "它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "自1956年达特茅斯会议正式提出人工智能概念以来，该领域经历了多次起伏，"
    "从早期的符号主义到专家系统，再到统计机器学习和深度学习的崛起。"
    "近年来，随着深度学习技术的突破、大数据基础设施的发展以及GPU/TPU等计算能力的提升，"
    "人工智能在图像识别、自然语言处理、语音识别、自动驾驶、医疗诊断等领域取得了显著进展。"
    "大型语言模型（LLM）如GPT系列、GLM系列、Claude系列等的出现，更是将人工智能推向了一个新的高度，"
    "这些模型通过对海量文本数据的学习，展现出了强大的语言理解和生成能力。"
    "在自然语言处理领域，预训练-微调范式已成为主流，BERT、GPT、T5等模型架构层出不穷。"
    "Transformer架构的自注意力机制使得模型能够捕捉长距离依赖关系，这是传统RNN/LSTM难以做到的。"
    "强化学习与人类反馈（RLHF）技术的引入，使得大语言模型能够更好地对齐人类意图和价值观。"
    "然而，人工智能的发展也面临着诸多挑战，包括数据隐私保护、算法公平性与偏见、能源消耗、"
    "以及如何确保AI系统的安全性、可控性和可解释性等问题。"
    "在未来，人工智能将继续深刻影响各行各业，从医疗健康到教育培训，"
    "从金融科技到智能制造，从智慧城市到环境保护，AI的应用前景广阔无垠。"
    "同时，我们也需要关注AI伦理和治理，确保技术的发展真正造福人类社会。"
    "量子计算与人工智能的结合可能带来革命性突破，量子机器学习有望解决经典计算难以处理的问题。"
    "边缘计算与联邦学习则让AI能力延伸到更多终端设备，同时保护用户数据隐私。"
    "多模态学习使得AI能够同时处理文本、图像、音频、视频等多种信息形式，"
    "这为实现更接近人类的感知和理解能力奠定了基础。"
    "AI Agent技术的发展使得大模型能够自主规划、调用工具、执行多步骤任务，"
    "这标志着人工智能从被动应答向主动执行的重要转变。"
    "在实际应用中，AI辅助编程工具如GitHub Copilot、Cursor等极大地提高了开发者的生产效率。"
    "AI在科学研究中的应用也日益广泛，从蛋白质结构预测到新材料发现，从气候模拟到药物研发。"
    "计算机视觉领域，目标检测、语义分割、图像生成（如Stable Diffusion、DALL-E）等技术日趋成熟。"
    "语音技术方面，语音识别、语音合成、声纹识别等已广泛应用于智能音箱、客服系统等场景。"
    "推荐系统借助深度学习，在电商、短视频、社交媒体等平台实现了精准的个性化推荐。"
    "知识图谱与大型语言模型的结合，有望解决大模型的幻觉问题和知识更新问题。"
    "检索增强生成（RAG）技术通过引入外部知识库，显著提升了模型回答的准确性和时效性。"
    "提示工程（Prompt Engineering）和上下文学习（In-Context Learning）成为高效使用大模型的关键技能。"
    "模型压缩技术如量化、剪枝、蒸馏等，使得大模型能够在资源受限的设备上高效运行。"
)


def gen_text(target_tokens: int) -> str:
    """生成约 target_tokens 个 token 的中文背景文本"""
    chars_needed = int(target_tokens * 1.4)
    base_chars = len(BASE_TEXT)
    repeats = max(1, chars_needed // base_chars + 1)
    return BASE_TEXT * repeats


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def headers_of(ep: dict) -> dict:
    return {
        "Authorization": f"Bearer {ep['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "Compare-Test/1.0",
    }


# ============================================================
# 单样本测试结果
# ============================================================
@dataclass
class SampleResult:
    ttft: float = 0.0           # 首token延迟 (秒)
    total_latency: float = 0.0  # 总耗时 (秒)
    output_tokens: int = 0      # 输出tokens
    input_tokens: int = 0       # 输入tokens (来自usage)
    tpot: float = 0.0           # 每输出token耗时 (秒, 不含TTFT)
    tps: float = 0.0            # 输出吞吐 tokens/s
    content_len: int = 0        # 输出字符数
    task_completed: bool = False
    completion_score: float = 0.0  # 0.0 ~ 1.0
    error: str = ""
    notes: str = ""


# ============================================================
# 任务完成度评估: 让模型执行结构化任务并打分
# ============================================================
TASK_INSTRUCTION = (
    "\n\n=== 任务 ===\n"
    "请仔细阅读以上技术背景资料，完成以下任务：\n"
    "1. 用中文写一段不少于200字的总结，概述人工智能的核心技术与发展趋势；\n"
    "2. 在总结末尾，另起一行，用格式【关键指标：关键词1、关键词2、关键词3】列出文中提到的3个技术关键词；\n"
    "3. 简要说明Transformer架构的一个核心优势。\n"
    "请直接输出内容，不要复述原文。"
)


def eval_completion(content: str) -> tuple[bool, float, str]:
    """评估任务完成度, 返回 (是否完成, 得分0-1, 说明)"""
    if not content:
        return False, 0.0, "输出为空"
    score = 0.0
    notes_parts = []
    # 1. 总结长度 >= 200字
    if len(content) >= 200:
        score += 0.3
        notes_parts.append("总结达标")
    else:
        notes_parts.append(f"总结过短({len(content)}字)")
    # 2. 关键指标行
    if "【关键指标" in content or "关键指标" in content:
        score += 0.3
        notes_parts.append("含关键指标行")
    else:
        notes_parts.append("缺少关键指标行")
    # 3. Transformer 相关
    if "Transformer" in content or "自注意力" in content or "注意力机制" in content:
        score += 0.2
        notes_parts.append("提及Transformer")
    else:
        notes_parts.append("未提及Transformer")
    # 4. 输出非空且结构化
    if len(content) > 100 and ("。" in content or "\n" in content):
        score += 0.2
        notes_parts.append("输出结构化")
    score = min(score, 1.0)
    return score >= 0.8, score, "；".join(notes_parts)


# ============================================================
# 执行测试: 双请求模式
#   1) 非流式请求: 测总延迟 / 准确usage / 内容(用于任务完成度评估)
#   2) 流式请求(小max_tokens): 仅测TTFT
# 这样即使中转缓冲式流式, TPOT/TPS 仍可由非流式总延迟准确得出
# ============================================================
def _build_prompt(condition: dict) -> str:
    context = gen_text(condition["input_tokens"])
    return context + TASK_INSTRUCTION


def _non_stream_request(ep: dict, condition: dict, prompt: str) -> SampleResult:
    """非流式请求: 获取总延迟、usage、内容"""
    res = SampleResult()
    session = requests.Session()
    session.trust_env = False
    payload = {
        "model": ep["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": condition["max_tokens"],
        "stream": False,
        "enable_thinking": False,
    }
    t0 = time.perf_counter()
    try:
        resp = session.post(
            chat_url(ep["base_url"]),
            headers=headers_of(ep),
            json=payload,
            timeout=900,
        )
        lat = time.perf_counter() - t0
        if resp.status_code != 200:
            res.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return res
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        res.total_latency = lat
        res.input_tokens = usage.get("prompt_tokens", 0)
        res.output_tokens = usage.get("completion_tokens", 0)
        res.content_len = len(content)
        if res.output_tokens == 0 and content:
            # usage未返回, 用字符数估算 (中文~1.4字符/token)
            res.output_tokens = max(1, int(len(content) / 1.4))
            res.notes = "usage无completion_tokens,按字符估算"
        # TPOT = 总延迟 / 输出tokens (含TTFT, 与 full_test.py 口径一致)
        if res.output_tokens > 0 and lat > 0:
            res.tpot = lat / res.output_tokens
            res.tps = res.output_tokens / lat
        # 任务完成度评估
        completed, score, notes = eval_completion(content)
        res.task_completed = completed
        res.completion_score = score
        if res.notes:
            res.notes += "；" + notes
        else:
            res.notes = notes
    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:150]}"
    return res


def _stream_ttft(ep: dict, condition: dict, prompt: str) -> tuple[float, str]:
    """流式请求(小max_tokens)测量TTFT, 返回 (ttft秒, 错误信息)"""
    session = requests.Session()
    session.trust_env = False
    payload = {
        "model": ep["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": min(100, condition["max_tokens"]),
        "stream": True,
        "enable_thinking": False,
    }
    t0 = time.perf_counter()
    try:
        resp = session.post(
            chat_url(ep["base_url"]),
            headers=headers_of(ep),
            json=payload,
            stream=True,
            timeout=900,
        )
        if resp.status_code != 200:
            return 0.0, f"HTTP {resp.status_code}: {resp.text[:150]}"
        first_line_sample = ""
        for line in resp.iter_lines():
            if not line:
                continue
            d = line.decode("utf-8", errors="ignore")
            if not d.startswith("data:"):
                continue
            payload_str = d[5:].strip()
            if payload_str == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content")
            if token:
                return (time.perf_counter() - t0, "")
            first_line_sample = payload_str[:80]
        return (0.0, f"未收到内容token,首块样例: {first_line_sample}")
    except Exception as e:
        return (0.0, f"{type(e).__name__}: {str(e)[:120]}")


def run_sample(ep: dict, condition: dict) -> SampleResult:
    prompt = _build_prompt(condition)
    # 1) 非流式: 总延迟 + usage + 内容
    res = _non_stream_request(ep, condition, prompt)
    if res.error:
        return res
    # 2) 流式: 仅测TTFT
    ttft, err = _stream_ttft(ep, condition, prompt)
    if err:
        res.notes = (res.notes + "；" if res.notes else "") + f"TTFT测量失败: {err}"
        # TTFT 取总延迟的近似 (无法精确测量首token)
        res.ttft = res.total_latency
    else:
        res.ttft = ttft
        # 检测缓冲式流式: 若TTFT接近总延迟的80%以上, 说明中转可能缓冲
        if res.total_latency > 0 and ttft > res.total_latency * 0.8 and res.output_tokens > 50:
            res.notes = (res.notes + "；" if res.notes else "") + "疑似缓冲式流式(TTFT≈总延迟)"
    return res


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("中转(sts-token) vs 上游(宇讯/tokeneasy) 对比测试")
    print("=" * 70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # results[endpoint][condition] = list[SampleResult]
    results: dict[str, dict[str, list[SampleResult]]] = {name: {} for name in ENDPOINTS}

    for ep_name, ep in ENDPOINTS.items():
        print(f"\n{'─'*60}\n端点: {ep_name}  (model={ep['model']})\n{'─'*60}")
        for cond in CONDITIONS:
            cn = cond["name"]
            samples: list[SampleResult] = []
            print(f"\n  [{cn}] 输入~{cond['input_tokens']}tokens, 输出上限{cond['max_tokens']}, 样本={cond['samples']}")
            for i in range(cond["samples"]):
                if i > 0:
                    time.sleep(3)  # 样本间小延迟, 避免限流
                print(f"    样本 {i+1}/{cond['samples']} ...", end=" ", flush=True)
                t_start = time.perf_counter()
                sr = run_sample(ep, cond)
                elapsed = time.perf_counter() - t_start
                if sr.error:
                    print(f"FAIL ({elapsed:.1f}s): {sr.error[:80]}")
                else:
                    print(f"OK ({elapsed:.1f}s) TTFT={sr.ttft*1000:.0f}ms TPOT={sr.tpot*1000:.1f}ms "
                          f"TPS={sr.tps:.1f} out={sr.output_tokens} 完成度={sr.completion_score*100:.0f}%")
                samples.append(sr)
            results[ep_name][cn] = samples

    generate_report(results)
    print("\n测试完成。")


def generate_report(results: dict[str, dict[str, list[SampleResult]]]):
    ep_names = list(ENDPOINTS.keys())
    lines: list[str] = []
    lines.append("# 中转 vs 上游 对比测试报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- 测试配置 ----
    lines.append("## 一、测试配置\n")
    lines.append("| 端点 | Base URL | 模型 |")
    lines.append("|------|----------|------|")
    for name, ep in ENDPOINTS.items():
        lines.append(f"| {name} | `{ep['base_url']}` | `{ep['model']}` |")
    lines.append("")
    lines.append("## 二、测试场景\n")
    lines.append("| 场景 | 目标输入tokens | 输出上限max_tokens | 采样次数 |")
    lines.append("|------|----------------|---------------------|----------|")
    for c in CONDITIONS:
        lines.append(f"| {c['name']} | {c['input_tokens']} | {c['max_tokens']} | {c['samples']} |")
    lines.append("")
    lines.append("> 任务设计：输入大段中文技术背景 + 结构化任务（写200字总结 + 列关键指标 + 说明Transformer优势），"
                 "用于评估模型在长上下文下的理解与指令遵循能力。\n")

    # ---- 详细对比 ----
    lines.append("## 三、各场景详细对比\n")
    for cond in CONDITIONS:
        cn = cond["name"]
        lines.append(f"### {cn} 上下文 (输入~{cond['input_tokens']} tokens, 输出上限 {cond['max_tokens']} tokens)\n")
        lines.append("| 端点 | 实际输入tokens | TTFT均值(ms) | TTFT P95(ms) | TPOT均值(ms) | TPS均值 | 输出tokens | 任务完成度 | 错误 |")
        lines.append("|------|---------------|-------------|-------------|-------------|---------|-----------|-----------|------|")
        for ep_name in ep_names:
            samples = results[ep_name][cn]
            ok = [s for s in samples if not s.error]
            if not ok:
                err = samples[0].error[:50] if samples else "无样本"
                lines.append(f"| {ep_name} | - | - | - | - | - | - | - | {err} |")
                continue
            in_toks = statistics.mean([s.input_tokens for s in ok]) if ok[0].input_tokens else 0
            ttfts = [s.ttft * 1000 for s in ok]
            tpots = [s.tpot * 1000 for s in ok if s.tpot > 0]
            tps_vals = [s.tps for s in ok if s.tps > 0]
            outs = [s.output_tokens for s in ok]
            scores = [s.completion_score for s in ok]
            ttft_mean = statistics.mean(ttfts)
            ttft_p95 = pct(ttfts, 95)
            tpot_mean = statistics.mean(tpots) if tpots else 0
            tps_mean = statistics.mean(tps_vals) if tps_vals else 0
            out_mean = statistics.mean(outs)
            score_mean = statistics.mean(scores)
            err = "; ".join(s.error[:30] for s in samples if s.error)
            in_str = f"{in_toks:.0f}" if in_toks else "~"
            lines.append(f"| {ep_name} | {in_str} | {ttft_mean:.0f} | {ttft_p95:.0f} | {tpot_mean:.1f} | "
                         f"{tps_mean:.1f} | {out_mean:.0f} | {score_mean*100:.0f}% | {err} |")
        lines.append("")

    # ---- 汇总对比 ----
    lines.append("## 四、汇总对比\n")
    lines.append("### 4.1 TTFT 首token延迟对比 (ms, 越低越好)\n")
    lines.append("| 场景 | " + " | ".join(ep_names) + " | 差值(中转-上游) |")
    lines.append("|------|" + "------|" * (len(ep_names) + 1))
    for cond in CONDITIONS:
        cn = cond["name"]
        vals = {}
        for ep_name in ep_names:
            ok = [s for s in results[ep_name][cn] if not s.error]
            vals[ep_name] = statistics.mean([s.ttft * 1000 for s in ok]) if ok else None
        row = [cn]
        for ep_name in ep_names:
            v = vals[ep_name]
            row.append(f"{v:.0f}" if v is not None else "FAIL")
        diff = ""
        if all(vals.values()):
            diff_val = list(vals.values())[0] - list(vals.values())[1]
            diff = f"{diff_val:+.0f}"
        row.append(diff)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### 4.2 TPOT 每输出token耗时对比 (ms, 越低越好, 含TTFT的总延迟/输出tokens)\n")
    lines.append("| 场景 | " + " | ".join(ep_names) + " | 差值(中转-上游) |")
    lines.append("|------|" + "------|" * (len(ep_names) + 1))
    for cond in CONDITIONS:
        cn = cond["name"]
        vals = {}
        for ep_name in ep_names:
            ok = [s for s in results[ep_name][cn] if not s.error and s.tpot > 0]
            vals[ep_name] = statistics.mean([s.tpot * 1000 for s in ok]) if ok else None
        row = [cn]
        for ep_name in ep_names:
            v = vals[ep_name]
            row.append(f"{v:.1f}" if v is not None else "FAIL")
        diff = ""
        if all(vals.values()):
            diff_val = list(vals.values())[0] - list(vals.values())[1]
            diff = f"{diff_val:+.1f}"
        row.append(diff)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### 4.3 TPS 输出吞吐对比 (tokens/s, 越高越好)\n")
    lines.append("| 场景 | " + " | ".join(ep_names) + " |")
    lines.append("|------|" + "------|" * len(ep_names))
    for cond in CONDITIONS:
        cn = cond["name"]
        row = [cn]
        for ep_name in ep_names:
            ok = [s for s in results[ep_name][cn] if not s.error and s.tps > 0]
            row.append(f"{statistics.mean([s.tps for s in ok]):.1f}" if ok else "FAIL")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### 4.4 任务完成度对比 (得分%, 越高越好)\n")
    lines.append("| 场景 | " + " | ".join(ep_names) + " |")
    lines.append("|------|" + "------|" * len(ep_names))
    for cond in CONDITIONS:
        cn = cond["name"]
        row = [cn]
        for ep_name in ep_names:
            ok = [s for s in results[ep_name][cn] if not s.error]
            row.append(f"{statistics.mean([s.completion_score for s in ok])*100:.0f}%" if ok else "FAIL")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- 模型性能与任务完成度分析 ----
    lines.append("## 五、模型性能与任务完成度分析\n")
    # 计算整体平均
    overall = {}
    for ep_name in ep_names:
        all_ok = [s for cond in CONDITIONS for s in results[ep_name][cond["name"]] if not s.error]
        if all_ok:
            overall[ep_name] = {
                "ttft": statistics.mean([s.ttft * 1000 for s in all_ok]),
                "tpot": statistics.mean([s.tpot * 1000 for s in all_ok if s.tpot > 0]),
                "tps": statistics.mean([s.tps for s in all_ok if s.tps > 0]),
                "score": statistics.mean([s.completion_score for s in all_ok]),
                "count": len(all_ok),
            }
    for ep_name in ep_names:
        if ep_name in overall:
            o = overall[ep_name]
            lines.append(f"- **{ep_name}**：成功{o['count']}次，平均TTFT={o['ttft']:.0f}ms，"
                         f"平均TPOT={o['tpot']:.1f}ms，平均TPS={o['tps']:.1f}，"
                         f"平均任务完成度={o['score']*100:.0f}%")
    lines.append("")

    # ---- 结论 ----
    lines.append("## 六、结论\n")
    if len(overall) == 2:
        a, b = ep_names
        oa, ob = overall[a], overall[b]
        lines.append("### 6.1 延迟性能\n")
        if oa["ttft"] < ob["ttft"]:
            lines.append(f"- **TTFT**：{a}({oa['ttft']:.0f}ms) 快于 {b}({ob['ttft']:.0f}ms)，"
                         f"差距 {abs(oa['ttft']-ob['ttft']):.0f}ms ({abs(oa['ttft']-ob['ttft'])/max(oa['ttft'],ob['ttft'])*100:.1f}%)")
        else:
            lines.append(f"- **TTFT**：{b}({ob['ttft']:.0f}ms) 快于 {a}({oa['ttft']:.0f}ms)，"
                         f"差距 {abs(oa['ttft']-ob['ttft']):.0f}ms ({abs(oa['ttft']-ob['ttft'])/max(oa['ttft'],ob['ttft'])*100:.1f}%)")
        if oa["tpot"] < ob["tpot"]:
            lines.append(f"- **TPOT**：{a}({oa['tpot']:.1f}ms) 优于 {b}({ob['tpot']:.1f}ms)")
        else:
            lines.append(f"- **TPOT**：{b}({ob['tpot']:.1f}ms) 优于 {a}({oa['tpot']:.1f}ms)")
        if oa["tps"] > ob["tps"]:
            lines.append(f"- **TPS吞吐**：{a}({oa['tps']:.1f}) 高于 {b}({ob['tps']:.1f})")
        else:
            lines.append(f"- **TPS吞吐**：{b}({ob['tps']:.1f}) 高于 {a}({oa['tps']:.1f})")
        lines.append("")
        lines.append("### 6.2 任务完成度\n")
        if oa["score"] > ob["score"]:
            lines.append(f"- 任务完成度：{a}({oa['score']*100:.0f}%) 略优于 {b}({ob['score']*100:.0f}%)")
        elif oa["score"] < ob["score"]:
            lines.append(f"- 任务完成度：{b}({ob['score']*100:.0f}%) 略优于 {a}({oa['score']*100:.0f}%)")
        else:
            lines.append(f"- 任务完成度：两者持平({oa['score']*100:.0f}%)")
        lines.append("")
        lines.append("### 6.3 长上下文表现\n")
        lines.append("- 4K/16K 短上下文：两者应均能正常完成任务，主要对比延迟；")
        lines.append("- 64K/128K 长上下文：观察是否出现超时、截断、任务遗忘或质量下降；")
        # 检查长上下文是否失败
        for cond in CONDITIONS[2:]:  # 64K, 128K
            cn = cond["name"]
            for ep_name in ep_names:
                samples = results[ep_name][cn]
                fails = [s for s in samples if s.error]
                if fails:
                    lines.append(f"- {cn}场景下 **{ep_name}** 出现 {len(fails)}/{len(samples)} 次失败：{fails[0].error[:60]}")
        lines.append("")
        lines.append("### 6.4 综合评价\n")
        # 综合判断
        ttft_win = a if oa["ttft"] < ob["ttft"] else b
        tpot_win = a if oa["tpot"] < ob["tpot"] else b
        tps_win = a if oa["tps"] > ob["tps"] else b
        score_win = a if oa["score"] >= ob["score"] else b
        lines.append(f"- 延迟最优：**{ttft_win}**（TTFT）/ **{tpot_win}**（TPOT）")
        lines.append(f"- 吞吐最优：**{tps_win}**")
        lines.append(f"- 任务完成最优：**{score_win}**")
        lines.append("")
        lines.append("> 注：中转端点经过一层转发，理论上TTFT会略高于上游（多一跳网络+转发开销）；"
                     "若中转TPOT/TPS接近上游，说明转发层未对吞吐造成明显瓶颈。"
                     "若长上下文场景中转出现失败而上游正常，需排查中转的请求体大小限制或超时配置。")
    lines.append("")
    lines.append("---")
    lines.append("*报告由 compare_test.py 自动生成*")

    report = "\n".join(lines)
    with open("compare_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # JSON 原始数据
    json_data = {
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "endpoints": ENDPOINTS,
        "conditions": CONDITIONS,
        "results": {
            ep_name: {
                cn: [asdict(s) for s in samples]
                for cn, samples in results[ep_name].items()
            }
            for ep_name in ep_names
        },
    }
    with open("compare_report.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: compare_report.md, compare_report.json")
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
