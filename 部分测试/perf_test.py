"""
性能测试脚本 - 对应 test.json 中 "3. 性能测试" (PERF-001 ~ PERF-007)

用法:
    在 .env 文件中填写 API 配置，然后运行:
    python perf_test.py
    只跑部分用例:
    python perf_test.py --only PERF-001,PERF-003
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    timeout: int = 120
    # 是否开启深度思考
    enable_thinking: bool = False
    # 并发测试档位 (PERF-004 / PERF-005)
    concurrency_levels: list[int] = field(default_factory=lambda: [5, 10, 20])
    # 最大输出长度测试 (PERF-006)
    max_output_tokens: int = 8192
    # 最大输入长度测试 (PERF-007)
    max_input_chars: int = 120_000
    # 短文本 TTFT 阈值 (PERF-001), 以 p95 为准
    ttft_threshold_s: float = 2.0
    # TPOT 阈值 (PERF-002), 以 p95 为准, 单位秒/token
    tpot_threshold_s: float = 0.1
    # TPS 阈值 (PERF-003)
    tps_threshold: float = 30.0
    # TTFT / TPOT 采样次数 (取 p95)
    sample_count: int = 10

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_url(self) -> str:
        # BASE_URL 已含 /v1 时只拼 /chat/completions，否则补 /v1/chat/completions
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


# ============================================================
# 结果
# ============================================================

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
    """发起一次 chat completions 请求"""
    payload = {"model": cfg.model, **payload}
    # 关闭/开启深度思考 (部分模型如 glm/deepseek 支持此参数)
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
    """从 usage 中拆出 (可见输出token, 思考token)。
    部分 API 把 reasoning_tokens 计入 completion_tokens，需要剔除才能算准 TPOT/TPS。"""
    completion = usage.get("completion_tokens", 0)
    # reasoning_tokens 可能在 completion_tokens_details 里，也可能直接平铺
    reasoning = 0
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens", 0) or usage.get("reasoning_tokens", 0)
    visible = max(completion - reasoning, 0)
    return visible, reasoning


def percentile(values: list[float], pct: float) -> float:
    """简单百分位: pct 取 0~100，如 95 表示 p95"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


# ============================================================
# PERF-001 首Token响应时间 (TTFT) - 多次采样取 p95
# ============================================================

def test_perf_001_ttft(cfg: Config) -> TestResult:
    """测量流式请求首个 token 到达时间 (p95)"""
    payload = {
        "messages": build_messages("用一句话介绍太阳系。"),
        "stream": True,
        "max_tokens": 100,
    }
    samples: list[float] = []
    errors: list[str] = []
    for _ in range(cfg.sample_count):
        t0 = time.perf_counter()
        ttft = None
        try:
            resp = chat_request(cfg, payload, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore")
                if decoded.startswith("data:") and "[DONE]" not in decoded:
                    ttft = time.perf_counter() - t0
                    break
        except Exception as e:
            errors.append(str(e))
            continue
        if ttft is not None:
            samples.append(ttft)

    if not samples:
        return TestResult("PERF-001", "首Token响应时间(TTFT)", False,
                          f"无有效样本, 错误: {errors[:3]}")

    p95 = percentile(samples, 95)
    p50 = percentile(samples, 50)
    passed = p95 < cfg.ttft_threshold_s
    return TestResult(
        "PERF-001", "首Token响应时间(TTFT)", passed,
        f"p95={p95:.3f}s, p50={p50:.3f}s, 样本={len(samples)}, 失败={len(errors)}, 阈值p95<{cfg.ttft_threshold_s}s",
        {
            "p95_s": round(p95, 3),
            "p50_s": round(p50, 3),
            "samples": len(samples),
            "failures": len(errors),
            "all_samples": [round(s, 3) for s in samples],
        },
    )


# ============================================================
# PERF-002 端到端响应时间 / TPOT - 多次采样取 p95
# ============================================================

def test_perf_002_latency(cfg: Config) -> TestResult:
    """测量非流式请求 TPOT (time per output token, p95)"""
    payload = {
        "messages": build_messages("写一篇 200 字左右的短文，主题是春天。"),
        "max_tokens": 512,
    }
    tpot_samples: list[float] = []
    latency_samples: list[float] = []
    errors: list[str] = []
    for _ in range(cfg.sample_count):
        try:
            t0 = time.perf_counter()
            resp = chat_request(cfg, payload)
            latency = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            errors.append(str(e))
            continue

        usage = data.get("usage", {})
        visible_tokens, reasoning_tokens = split_tokens(usage)
        if visible_tokens == 0 or latency == 0:
            continue
        tpot_samples.append(latency / visible_tokens)
        latency_samples.append(latency)

    if not tpot_samples:
        return TestResult("PERF-002", "端到端响应时间(TPOT)", False,
                          f"无有效样本, 错误: {errors[:3]}")

    p95_tpot = percentile(tpot_samples, 95)
    p50_tpot = percentile(tpot_samples, 50)
    p95_latency = percentile(latency_samples, 95)
    passed = p95_tpot < cfg.tpot_threshold_s
    return TestResult(
        "PERF-002", "端到端响应时间(TPOT)", passed,
        f"TPOT p95={p95_tpot:.4f}s/tok, p50={p50_tpot:.4f}s/tok, latency p95={p95_latency:.3f}s, 样本={len(tpot_samples)}, 失败={len(errors)}, 阈值p95<{cfg.tpot_threshold_s}s/tok",
        {
            "tpot_p95_s": round(p95_tpot, 4),
            "tpot_p50_s": round(p50_tpot, 4),
            "latency_p95_s": round(p95_latency, 3),
            "samples": len(tpot_samples),
            "failures": len(errors),
        },
    )


# ============================================================
# PERF-003 吞吐量 (TPS)
# ============================================================

def test_perf_003_tps(cfg: Config) -> TestResult:
    """输出 token 数 / 耗时 = TPS"""
    payload = {
        "messages": build_messages("写一篇 500 字左右的文章，主题是人工智能的发展。"),
        "max_tokens": 1024,
    }
    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, payload)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return TestResult("PERF-003", "吞吐量(TPS)", False, f"请求异常: {e}")

    usage = data.get("usage", {})
    visible_tokens, reasoning_tokens = split_tokens(usage)
    if visible_tokens == 0 or latency == 0:
        return TestResult("PERF-003", "吞吐量(TPS)", False, "无法计算 TPS (tokens 或 latency 为 0)")

    tps = visible_tokens / latency
    passed = tps > cfg.tps_threshold
    extra = f", 思考token={reasoning_tokens}" if reasoning_tokens else ""
    return TestResult(
        "PERF-003", "吞吐量(TPS)", passed,
        f"TPS={tps:.2f}, 可见tokens={visible_tokens}{extra}, latency={latency:.3f}s, 阈值>{cfg.tps_threshold}",
        {"tps": round(tps, 2), "visible_tokens": visible_tokens, "reasoning_tokens": reasoning_tokens, "latency_s": round(latency, 3)},
    )


# ============================================================
# PERF-004 并发能力
# ============================================================

def _single_request(cfg: Config) -> tuple[bool, float, str]:
    """单次并发请求，返回 (是否成功, 耗时, 错误信息)"""
    payload = {
        "messages": build_messages("用一句话介绍地球。"),
        "max_tokens": 50,
    }
    t0 = time.perf_counter()
    try:
        resp = chat_request(cfg, payload)
        resp.raise_for_status()
        return True, time.perf_counter() - t0, ""
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        return False, time.perf_counter() - t0, f"HTTP {code}"
    except Exception as e:
        return False, time.perf_counter() - t0, str(e)


def test_perf_004_concurrency(cfg: Config) -> TestResult:
    """测试能否支撑并发请求"""
    level = cfg.concurrency_levels[-1]  # 取最高档
    try:
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(_single_request, cfg) for _ in range(level)]
            results = [f.result() for f in as_completed(futures)]
    except Exception as e:
        return TestResult("PERF-004", "并发能力", False, f"并发异常: {e}")

    ok = sum(1 for r in results if r[0])
    fail = level - ok
    success_rate = ok / level
    # 统计失败原因
    error_counter: dict[str, int] = {}
    for r in results:
        if not r[0] and r[2]:
            error_counter[r[2]] = error_counter.get(r[2], 0) + 1
    error_summary = ", ".join(f"{k}x{v}" for k, v in error_counter.items()) or "无"
    # 通过标准：成功率 >= 95%
    passed = success_rate >= 0.95
    return TestResult(
        "PERF-004", "并发能力", passed,
        f"并发={level}, 成功={ok}, 失败={fail}, 成功率={success_rate:.1%}, 失败原因: {error_summary}",
        {"concurrency": level, "success": ok, "fail": fail, "success_rate": round(success_rate, 4), "errors": error_counter},
    )


# ============================================================
# PERF-005 并发响应时间
# ============================================================

def test_perf_005_concurrency_latency(cfg: Config) -> TestResult:
    """不同并发档位下的响应时间"""
    level_results: list[dict[str, Any]] = []
    all_passed = True

    for level in cfg.concurrency_levels:
        try:
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [pool.submit(_single_request, cfg) for _ in range(level)]
                results = [f.result() for f in as_completed(futures)]
        except Exception as e:
            level_results.append({"concurrency": level, "error": str(e)})
            all_passed = False
            continue

        latencies = [r[1] for r in results if r[0]]
        error_counter: dict[str, int] = {}
        for r in results:
            if not r[0] and r[2]:
                error_counter[r[2]] = error_counter.get(r[2], 0) + 1
        error_summary = ", ".join(f"{k}x{v}" for k, v in error_counter.items()) or "无"

        if not latencies:
            level_results.append({"concurrency": level, "success": 0, "error": error_summary})
            all_passed = False
            continue

        avg = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = percentile(latencies, 95)
        level_results.append({
            "concurrency": level,
            "success": len(latencies),
            "fail": level - len(latencies),
            "errors": error_counter,
            "avg_s": round(avg, 3),
            "p50_s": round(p50, 3),
            "p95_s": round(p95, 3),
        })
        # 通过标准：p95 不超过 avg 的 5 倍（响应时间合理增长而非雪崩）
        if p95 > avg * 5:
            all_passed = False

    detail = "; ".join(
        f"{lr['concurrency']}并发: ok={lr.get('success','?')}/{lr['concurrency']}, avg={lr.get('avg_s','?')}s p95={lr.get('p95_s','?')}s"
        for lr in level_results
    )
    return TestResult(
        "PERF-005", "并发响应时间", all_passed,
        detail,
        {"levels": level_results},
    )


# ============================================================
# PERF-006 最大输出长度
# ============================================================

def test_perf_006_max_output(cfg: Config) -> TestResult:
    """测试能否输出到 max_tokens 上限"""
    payload = {
        "messages": build_messages("请尽可能详细地描写一个虚构的中世纪城市，不少于 8000 字。"),
        "max_tokens": cfg.max_output_tokens,
    }
    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, payload)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return TestResult("PERF-006", "最大输出长度", False, f"请求异常: {e}")

    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    finish_reason = (data.get("choices") or [{}])[0].get("finish_reason", "")
    # 通过标准：实际输出 token 接近 max_tokens（>= 90%）且未异常中断
    ratio = completion_tokens / cfg.max_output_tokens if cfg.max_output_tokens else 0
    passed = ratio >= 0.9
    return TestResult(
        "PERF-006", "最大输出长度", passed,
        f"max_tokens={cfg.max_output_tokens}, 实际={completion_tokens}, 覆盖率={ratio:.1%}, finish={finish_reason}, latency={latency:.1f}s",
        {
            "max_tokens": cfg.max_output_tokens,
            "actual_tokens": completion_tokens,
            "coverage": round(ratio, 4),
            "finish_reason": finish_reason,
            "latency_s": round(latency, 3),
        },
    )


# ============================================================
# PERF-007 最大输入长度
# ============================================================

def test_perf_007_max_input(cfg: Config) -> TestResult:
    """测试长文本输入能否被正常处理"""
    long_text = ("这是一段用于测试长上下文能力的文本。" * (cfg.max_input_chars // 20 + 1))[:cfg.max_input_chars]
    payload = {
        "messages": build_messages(f"请阅读以下文本并总结一句话：\n\n{long_text}"),
        "max_tokens": 100,
    }
    try:
        t0 = time.perf_counter()
        resp = chat_request(cfg, payload)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        # 4xx 说明超长被拒
        code = e.response.status_code if e.response is not None else 0
        return TestResult(
            "PERF-007", "最大输入长度", False,
            f"返回 HTTP {code}，输入可能超过上下文上限",
            {"input_chars": len(long_text), "http_code": code},
        )
    except Exception as e:
        return TestResult("PERF-007", "最大输入长度", False, f"请求异常: {e}")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    passed = bool(content) and prompt_tokens > 0
    return TestResult(
        "PERF-007", "最大输入长度", passed,
        f"input_chars={len(long_text)}, prompt_tokens={prompt_tokens}, latency={latency:.1f}s, 回复长度={len(content)}",
        {
            "input_chars": len(long_text),
            "prompt_tokens": prompt_tokens,
            "latency_s": round(latency, 3),
            "reply_len": len(content),
        },
    )


# ============================================================
# 运行入口
# ============================================================

TEST_FUNCS = {
    "PERF-001": test_perf_001_ttft,
    "PERF-002": test_perf_002_latency,
    "PERF-003": test_perf_003_tps,
    "PERF-004": test_perf_004_concurrency,
    "PERF-005": test_perf_005_concurrency_latency,
    "PERF-006": test_perf_006_max_output,
    "PERF-007": test_perf_007_max_input,
}


def parse_args() -> tuple[Config, list[str] | None]:
    parser = argparse.ArgumentParser(description="API 性能测试脚本 (PERF-001 ~ PERF-007)")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时(秒)")
    parser.add_argument("--concurrency", type=str, default="5,10,20", help="并发档位，逗号分隔")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="PERF-006 测试的最大输出 token")
    parser.add_argument("--max-input-chars", type=int, default=120000, help="PERF-007 测试的最大输入字符数")
    parser.add_argument("--ttft-threshold", type=float, default=2.0, help="PERF-001 TTFT p95 阈值(秒)")
    parser.add_argument("--tpot-threshold", type=float, default=0.1, help="PERF-002 TPOT p95 阈值(秒/token)")
    parser.add_argument("--tps-threshold", type=float, default=30.0, help="PERF-003 TPS 阈值")
    parser.add_argument("--sample-count", type=int, default=10, help="TTFT/TPOT 采样次数")
    parser.add_argument("--only", help="只跑指定用例，逗号分隔，如 PERF-001,PERF-003")
    args = parser.parse_args()

    cfg = Config(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=args.timeout,
        enable_thinking=ENABLE_THINKING,
        concurrency_levels=[int(x) for x in args.concurrency.split(",") if x.strip()],
        max_output_tokens=args.max_output_tokens,
        max_input_chars=args.max_input_chars,
        ttft_threshold_s=args.ttft_threshold,
        tpot_threshold_s=args.tpot_threshold,
        tps_threshold=args.tps_threshold,
        sample_count=args.sample_count,
    )

    only = [x.strip().upper() for x in args.only.split(",")] if args.only else None
    return cfg, only


def main() -> int:
    cfg, only = parse_args()
    print(f"\n{'='*60}")
    print(f"性能测试开始")
    print(f"Base URL : {cfg.base_url}")
    print(f"Model    : {cfg.model}")
    print(f"深度思考 : {'开启' if cfg.enable_thinking else '关闭'}")
    print(f"并发档位 : {cfg.concurrency_levels}")
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
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": [
            {"id": r.id, "name": r.name, "passed": r.passed, "detail": r.detail, "metrics": r.metrics}
            for r in results
        ],
    }
    report_path = "perf_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已写入: {report_path}")

    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
