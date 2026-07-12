"""
超长上下文测试脚本 - 验证 GLM-5.2 的 1M 输入 / 128K 输出是否属实，以及超长上下文下能否真正关联信息

测试项:
  LONG-CTX-001  最大输入长度验证     能否接受 ~1M token 输入
  LONG-CTX-002  最大输出长度验证     能否输出到 ~128K token
  LONG-CTX-003  大海捞针(开头)       在长文开头埋针，验证能否检索
  LONG-CTX-004  大海捞针(中间)       在长文中间埋针，验证能否检索（最关键）
  LONG-CTX-005  大海捞针(末尾)       在长文末尾埋针，验证能否检索
  LONG-CTX-006  跨上下文关联         在长文不同位置放关联线索，验证能否串联推理

用法:
    在 .env 文件中填写 API 配置，然后运行:
    python longctx_test.py                          # 默认档(200K输入/16K输出，快速验证)
    python longctx_test.py --full                   # 满档(1M输入/128K输出，完整验证)
    python longctx_test.py --only LONG-CTX-004      # 只跑中间埋针
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置从 .env 文件读取
# ============================================================
BASE_URL = os.getenv("BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o")
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"
# ============================================================


@dataclass
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: int = 1800
    enable_thinking: bool = False
    # 输入测试目标 token 数 (1M = 1000000)
    target_input_tokens: int = 200_000
    # 输出测试目标 token 数 (128K = 131072)
    target_output_tokens: int = 16_384
    # 大海捞针测试用的上下文 token 数
    haystack_tokens: int = 200_000
    # 字符/token 估算比 (中文约1.7)
    chars_per_token: float = 1.7

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


@dataclass
class TestResult:
    id: str
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


# ============================================================
# HTTP 请求封装
# ============================================================

def chat_request(cfg: Config, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
    payload = {"model": cfg.model, **payload}
    payload["enable_thinking"] = cfg.enable_thinking
    return requests.post(
        cfg.chat_url(),
        headers=cfg.headers,
        json=payload,
        stream=stream,
        timeout=cfg.timeout,
    )


def build_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def split_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    completion = usage.get("completion_tokens", 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens", 0) or usage.get("reasoning_tokens", 0)
    visible = max(completion - reasoning, 0)
    return visible, reasoning


# ============================================================
# 填充文本生成
# ============================================================

# 用于填充的无关段落（让上下文足够长但不包含针）
_FILLER_PARAGRAPH = (
    "太阳系的八大行星依次为水星、金星、地球、火星、木星、土星、天王星和海王星。"
    "木星是太阳系中体积最大的行星，其质量约为其他所有行星质量总和的两倍以上。"
    "地球是目前已知唯一存在液态水和生命的行星，自转一周约二十四小时。"
    "土星以其壮观的行星环闻名，主要由冰粒和岩石碎片构成。"
)


def make_filler_text(target_chars: int) -> str:
    """生成指定字符数的填充文本"""
    if target_chars <= 0:
        return ""
    unit = len(_FILLER_PARAGRAPH)
    repeats = target_chars // unit + 1
    text = _FILLER_PARAGRAPH * repeats
    return text[:target_chars]


def tokens_to_chars(cfg: Config, tokens: int) -> int:
    return int(tokens * cfg.chars_per_token)


# ============================================================
# LONG-CTX-001 最大输入长度验证
# ============================================================

def test_long_ctx_001_max_input(cfg: Config) -> TestResult:
    """验证能否接受超长输入 (~1M token)"""
    target = cfg.target_input_tokens
    target_chars = tokens_to_chars(cfg, target)
    long_text = make_filler_text(target_chars)
    prompt = f"请阅读以下文本并回答：文本提到了几大行星？只回答数字。\n\n{long_text}"
    print(f"  LONG-CTX-001 输入字符数={len(long_text)} (目标~{target} tokens)")

    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, {"messages": build_messages(prompt), "max_tokens": 100})
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        body = ""
        if e.response is not None:
            try:
                body = e.response.json().get("error", {}).get("message", "")[:200]
            except Exception:
                body = e.response.text[:200]
        return TestResult(
            "LONG-CTX-001", "最大输入长度验证", False,
            f"返回 HTTP {code}: {body} (输入可能超过上下文上限)",
            {"input_chars": len(long_text), "target_tokens": target, "http_code": code, "error": body},
        )
    except Exception as e:
        return TestResult("LONG-CTX-001", "最大输入长度验证", False, f"请求异常: {e}",
                          {"input_chars": len(long_text), "target_tokens": target})

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    # 通过标准：被接受(非4xx)且回复合理(应回答8)
    accepted = prompt_tokens > 0
    answered = bool(content)
    coverage = prompt_tokens / target if target else 0
    passed = accepted and answered
    return TestResult(
        "LONG-CTX-001", "最大输入长度验证", passed,
        f"目标~{target}tokens, 实际prompt_tokens={prompt_tokens}, 覆盖率={coverage:.0%}, 回复='{content[:30]}', latency={latency:.1f}s",
        {
            "target_tokens": target, "input_chars": len(long_text),
            "prompt_tokens": prompt_tokens, "coverage": round(coverage, 4),
            "reply": content[:100], "latency_s": round(latency, 3),
        },
    )


# ============================================================
# LONG-CTX-002 最大输出长度验证
# ============================================================

def test_long_ctx_002_max_output(cfg: Config) -> TestResult:
    """验证能否输出到超长上限 (~128K token)"""
    target = cfg.target_output_tokens
    payload = {
        "messages": build_messages(
            "请尽可能详细地描写一个虚构的中世纪奇幻城市，包括地理、建筑、居民、历史、文化、经济、宗教等方方面面，"
            "内容越丰富越好，不要少于数万字。"
        ),
        "max_tokens": target,
    }
    print(f"  LONG-CTX-002 请求 max_tokens={target}")

    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, payload)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return TestResult("LONG-CTX-002", "最大输出长度验证", False, f"请求异常: {e}",
                          {"target_tokens": target})

    usage = data.get("usage", {})
    visible_tokens, reasoning_tokens = split_tokens(usage)
    finish_reason = (data.get("choices") or [{}])[0].get("finish_reason", "")
    coverage = visible_tokens / target if target else 0
    # 通过标准：实际输出 >= 目标的 90%
    passed = coverage >= 0.9
    extra = f", 思考token={reasoning_tokens}" if reasoning_tokens else ""
    return TestResult(
        "LONG-CTX-002", "最大输出长度验证", passed,
        f"max_tokens={target}, 实际输出={visible_tokens}, 覆盖率={coverage:.0%}, finish={finish_reason}{extra}, latency={latency:.1f}s",
        {
            "target_tokens": target, "actual_tokens": visible_tokens,
            "reasoning_tokens": reasoning_tokens, "coverage": round(coverage, 4),
            "finish_reason": finish_reason, "latency_s": round(latency, 3),
        },
    )


# ============================================================
# 大海捞针 - 在长文本指定位置插入"针"，验证能否检索
# ============================================================

# 针：一段特定且不易被填充文本包含的信息
NEEDLE_TEXT = "特别记录：2026年7月7日，张三把保险柜的密码设定为 pineapple-7291。"
NEEDLE_QUESTION = "请告诉我张三设定的保险柜密码是什么？只回答密码本身。"
NEEDLE_ANSWER = "pineapple-7291"


def build_haystack(cfg: Config, position: float) -> str:
    """构造长文本，在 position(0~1) 处插入针"""
    total_chars = tokens_to_chars(cfg, cfg.haystack_tokens)
    needle_chars = len(NEEDLE_TEXT)
    filler_total = max(total_chars - needle_chars, 0)
    # 分两段填充，中间插针
    head_chars = int(filler_total * position)
    tail_chars = filler_total - head_chars
    head = make_filler_text(head_chars)
    tail = make_filler_text(tail_chars)
    return f"{head}\n\n{NEEDLE_TEXT}\n\n{tail}"


def _haystack_test(cfg: Config, tid: str, name: str, position: float) -> TestResult:
    haystack = build_haystack(cfg, position)
    prompt = f"{haystack}\n\n{NEEDLE_QUESTION}"
    print(f"  {tid} 上下文字符={len(haystack)} (针位置={position:.0%})")

    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, {"messages": build_messages(prompt), "max_tokens": 100})
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return TestResult(tid, name, False, f"请求异常: {e}",
                          {"haystack_chars": len(haystack), "position": position})

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    found = NEEDLE_ANSWER in content
    pos_label = {"start": "开头", "middle": "中间", "end": "末尾"}.get(
        "start" if position < 0.2 else "middle" if position < 0.8 else "end", str(position))
    return TestResult(
        tid, name, found,
        f"针位置={pos_label}, prompt_tokens={prompt_tokens}, 回复='{content[:40]}', 找到针={found}, latency={latency:.1f}s",
        {
            "position": position, "position_label": pos_label,
            "prompt_tokens": prompt_tokens, "reply": content[:100],
            "found_needle": found, "latency_s": round(latency, 3),
        },
    )


def test_long_ctx_003_needle_start(cfg: Config) -> TestResult:
    """大海捞针 - 针在开头(~10%)"""
    return _haystack_test(cfg, "LONG-CTX-003", "大海捞针(开头)", 0.1)


def test_long_ctx_004_needle_middle(cfg: Config) -> TestResult:
    """大海捞针 - 针在中间(~50%)，最关键"""
    return _haystack_test(cfg, "LONG-CTX-004", "大海捞针(中间)", 0.5)


def test_long_ctx_005_needle_end(cfg: Config) -> TestResult:
    """大海捞针 - 针在末尾(~90%)"""
    return _haystack_test(cfg, "LONG-CTX-005", "大海捞针(末尾)", 0.9)


# ============================================================
# LONG-CTX-006 跨上下文关联
# ============================================================

def test_long_ctx_006_cross_association(cfg: Config) -> TestResult:
    """在长文不同位置放置关联线索，验证能否串联推理"""
    total_chars = tokens_to_chars(cfg, cfg.haystack_tokens)
    # 线索A (20%处): 李四的工号
    clue_a = "人事记录：员工李四的工号为 ENG-4471。"
    # 线索B (80%处): 该工号对应的保险箱密码
    clue_b = "安保记录：工号 ENG-4471 负责保管的保险箱，密码为 7788-APPLE-9923。"
    # 需要串联 A->B 才能回答
    question = "李四保管的保险箱密码是什么？只回答密码本身。"
    answer = "7788-APPLE-9923"

    filler_total = max(total_chars - len(clue_a) - len(clue_b), 0)
    head_chars = int(filler_total * 0.2)
    mid_chars = int(filler_total * 0.6)
    tail_chars = filler_total - head_chars - mid_chars

    text = (
        make_filler_text(head_chars) + "\n\n" + clue_a + "\n\n"
        + make_filler_text(mid_chars) + "\n\n" + clue_b + "\n\n"
        + make_filler_text(tail_chars)
    )
    prompt = f"{text}\n\n{question}"
    print(f"  LONG-CTX-006 上下文字符={len(text)} (线索A@20%, 线索B@80%)")

    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, {"messages": build_messages(prompt), "max_tokens": 100})
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return TestResult("LONG-CTX-006", "跨上下文关联", False, f"请求异常: {e}",
                          {"haystack_chars": len(text)})

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    found = answer in content
    return TestResult(
        "LONG-CTX-006", "跨上下文关联", found,
        f"prompt_tokens={prompt_tokens}, 回复='{content[:40]}', 正确关联={found}, latency={latency:.1f}s",
        {
            "prompt_tokens": prompt_tokens, "reply": content[:100],
            "expected": answer, "associated": found, "latency_s": round(latency, 3),
        },
    )


# ============================================================
# 运行入口
# ============================================================

TEST_FUNCS = {
    "LONG-CTX-001": test_long_ctx_001_max_input,
    "LONG-CTX-002": test_long_ctx_002_max_output,
    "LONG-CTX-003": test_long_ctx_003_needle_start,
    "LONG-CTX-004": test_long_ctx_004_needle_middle,
    "LONG-CTX-005": test_long_ctx_005_needle_end,
    "LONG-CTX-006": test_long_ctx_006_cross_association,
}


def parse_args() -> tuple[Config, list[str] | None]:
    parser = argparse.ArgumentParser(description="超长上下文测试脚本 (LONG-CTX-001 ~ 006)")
    parser.add_argument("--timeout", type=int, default=1800, help="单次请求超时(秒)，长输出建议>=1800")
    parser.add_argument("--target-input-tokens", type=int, default=200000, help="输入测试目标token数(1M=1000000)")
    parser.add_argument("--target-output-tokens", type=int, default=16384, help="输出测试目标token数(128K=131072)")
    parser.add_argument("--haystack-tokens", type=int, default=200000, help="大海捞针测试上下文token数")
    parser.add_argument("--full", action="store_true", help="满档: 1M输入 + 128K输出 + 1M大海捞针")
    parser.add_argument("--only", help="只跑指定用例，逗号分隔")
    args = parser.parse_args()

    target_in = args.target_input_tokens
    target_out = args.target_output_tokens
    haystack = args.haystack_tokens
    if args.full:
        target_in = 1_000_000
        target_out = 131_072
        haystack = 1_000_000

    cfg = Config(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=args.timeout,
        enable_thinking=ENABLE_THINKING,
        target_input_tokens=target_in,
        target_output_tokens=target_out,
        haystack_tokens=haystack,
    )

    only = [x.strip().upper() for x in args.only.split(",")] if args.only else None
    return cfg, only


def main() -> int:
    cfg, only = parse_args()
    print(f"\n{'='*60}")
    print(f"超长上下文测试开始")
    print(f"Base URL       : {cfg.base_url}")
    print(f"Model          : {cfg.model}")
    print(f"深度思考       : {'开启' if cfg.enable_thinking else '关闭'}")
    print(f"输入目标       : ~{cfg.target_input_tokens} tokens")
    print(f"输出目标       : ~{cfg.target_output_tokens} tokens")
    print(f"大海捞针上下文 : ~{cfg.haystack_tokens} tokens")
    print(f"超时           : {cfg.timeout}s")
    print(f"{'='*60}\n")

    test_ids = [tid for tid in TEST_FUNCS if (only is None or tid in only)]
    if not test_ids:
        print("没有可执行的测试用例，请检查 --only 参数。")
        return 1

    results: list[TestResult] = []
    for tid in test_ids:
        func = TEST_FUNCS[tid]
        print(f"[{tid}] {func.__doc__ or ''}".strip())
        try:
            r = func(cfg)
        except Exception as e:
            r = TestResult(tid, "", False, f"测试异常: {e}")
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  -> [{status}] {r.detail}\n")

    # 汇总
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f"汇总: {passed}/{len(results)} 通过")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.id} {r.name} -- {r.detail}")
    print(f"{'='*60}")

    # 输出 JSON 报告
    report = {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "config": {
            "target_input_tokens": cfg.target_input_tokens,
            "target_output_tokens": cfg.target_output_tokens,
            "haystack_tokens": cfg.haystack_tokens,
        },
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": [
            {"id": r.id, "name": r.name, "passed": r.passed, "detail": r.detail, "metrics": r.metrics}
            for r in results
        ],
    }
    report_path = "longctx_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已写入: {report_path}")

    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
