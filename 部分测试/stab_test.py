"""
稳定性测试脚本 - 对应 test.json 中 "4. 稳定性测试" (STAB-001 ~ STAB-005)

用法:
    在 .env 文件中填写 API 配置，然后运行:
    python stab_test.py
    只跑部分用例:
    python stab_test.py --only STAB-001,STAB-003
    调整压测时长:
    python stab_test.py --long-duration 600 --sustain-duration 300
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
    enable_thinking: bool = False
    # STAB-001 连续请求次数
    continuous_count: int = 100
    # STAB-002 长时间运行时长(秒)。规范要求 24h=86400，默认 300 便于快速验证
    long_duration_s: int = 300
    # STAB-004 持续并发压测
    sustain_concurrency: int = 20
    sustain_duration_s: int = 600
    # STAB-004 错误率阈值 <0.1%
    error_rate_threshold: float = 0.001
    # STAB-005 限流探测并发数
    rate_limit_burst: int = 50

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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _do_one(cfg: Config) -> tuple[bool, float, str]:
    """单次请求，返回 (成功, 耗时, 错误信息)"""
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


# ============================================================
# STAB-001 连续请求稳定性
# ============================================================

def test_stab_001_continuous(cfg: Config) -> TestResult:
    """连续 N 次请求，验证无错误/崩溃"""
    count = cfg.continuous_count
    ok = 0
    fail = 0
    latencies: list[float] = []
    error_counter: dict[str, int] = {}

    for i in range(count):
        success, latency, err = _do_one(cfg)
        if success:
            ok += 1
            latencies.append(latency)
        else:
            fail += 1
            error_counter[err] = error_counter.get(err, 0) + 1
        # 进度
        if (i + 1) % 20 == 0:
            print(f"  STAB-001 进度: {i+1}/{count}, 成功={ok}, 失败={fail}")

    error_rate = fail / count
    error_summary = ", ".join(f"{k}x{v}" for k, v in error_counter.items()) or "无"
    passed = fail == 0
    p95 = percentile(latencies, 95) if latencies else 0
    return TestResult(
        "STAB-001", "连续请求稳定性", passed,
        f"共{count}次, 成功={ok}, 失败={fail}, 错误率={error_rate:.2%}, latency p95={p95:.3f}s, 失败原因: {error_summary}",
        {
            "count": count, "success": ok, "fail": fail,
            "error_rate": round(error_rate, 4),
            "latency_p95_s": round(p95, 3),
            "errors": error_counter,
        },
    )


# ============================================================
# STAB-002 长时间运行稳定性
# ============================================================

def test_stab_002_long_run(cfg: Config) -> TestResult:
    """持续请求指定时长，验证无崩溃"""
    duration = cfg.long_duration_s
    deadline = time.perf_counter() + duration
    ok = 0
    fail = 0
    rounds = 0
    error_counter: dict[str, int] = {}
    latencies: list[float] = []

    while time.perf_counter() < deadline:
        rounds += 1
        success, latency, err = _do_one(cfg)
        if success:
            ok += 1
            latencies.append(latency)
        else:
            fail += 1
            error_counter[err] = error_counter.get(err, 0) + 1
        if rounds % 20 == 0:
            elapsed = time.perf_counter() - (deadline - duration)
            print(f"  STAB-002 进度: {elapsed:.0f}/{duration}s, 轮次={rounds}, 成功={ok}, 失败={fail}")

    error_rate = fail / rounds if rounds else 1
    error_summary = ", ".join(f"{k}x{v}" for k, v in error_counter.items()) or "无"
    # 通过标准：无崩溃，错误率低
    passed = error_rate == 0
    p95 = percentile(latencies, 95) if latencies else 0
    return TestResult(
        "STAB-002", "长时间运行稳定性", passed,
        f"时长={duration}s, 轮次={rounds}, 成功={ok}, 失败={fail}, 错误率={error_rate:.2%}, latency p95={p95:.3f}s, 失败原因: {error_summary}",
        {
            "duration_s": duration, "rounds": rounds, "success": ok, "fail": fail,
            "error_rate": round(error_rate, 4),
            "latency_p95_s": round(p95, 3),
            "errors": error_counter,
        },
    )


# ============================================================
# STAB-003 错误恢复
# ============================================================

def test_stab_003_error_recovery(cfg: Config) -> TestResult:
    """发送异常请求后，验证服务能否自动恢复正常"""
    url = cfg.chat_url()
    headers = cfg.headers

    # 构造几类异常请求
    abnormal_cases = [
        ("畸形JSON", headers, "not-a-valid-json{", "raw"),
        ("无效模型名", headers, json.dumps({"model": "this-model-not-exist", "messages": build_messages("hi"), "enable_thinking": cfg.enable_thinking}), "raw"),
        ("空messages", headers, json.dumps({"model": cfg.model, "messages": [], "enable_thinking": cfg.enable_thinking}), "raw"),
        ("超长字段", headers, json.dumps({"model": cfg.model, "messages": build_messages("x" * 200000), "enable_thinking": cfg.enable_thinking}), "raw"),
    ]

    abnormal_results: list[dict[str, Any]] = []
    for name, hdr, body, _ in abnormal_cases:
        try:
            resp = requests.post(url, headers=hdr, data=body, timeout=cfg.timeout)
            code = resp.status_code
            abnormal_results.append({"case": name, "status": code, "ok": 200 <= code < 500})
        except Exception as e:
            abnormal_results.append({"case": name, "status": -1, "ok": False, "error": str(e)})
        time.sleep(0.5)

    # 发一个正常请求，验证是否恢复
    time.sleep(1)
    success, latency, err = _do_one(cfg)
    recovered = success
    passed = recovered
    abnormal_summary = "; ".join(f"{r['case']}={r['status']}" for r in abnormal_results)
    return TestResult(
        "STAB-003", "错误恢复", passed,
        f"异常请求: [{abnormal_summary}]; 恢复请求: {'成功' if recovered else '失败'}({latency:.2f}s){', '+err if err else ''}",
        {
            "abnormal_cases": abnormal_results,
            "recovered": recovered,
            "recovery_latency_s": round(latency, 3),
        },
    )


# ============================================================
# STAB-004 并发稳定性 (持续压测)
# ============================================================

def test_stab_004_sustain_concurrency(cfg: Config) -> TestResult:
    """持续并发压测指定时长，验证错误率 <0.1%"""
    concurrency = cfg.sustain_concurrency
    duration = cfg.sustain_duration_s
    deadline = time.perf_counter() + duration
    ok = 0
    fail = 0
    total = 0
    error_counter: dict[str, int] = {}
    latencies: list[float] = []

    def worker():
        nonlocal ok, fail, total
        local_ok = 0
        local_fail = 0
        local_lat: list[float] = []
        local_err: dict[str, int] = {}
        while time.perf_counter() < deadline:
            success, latency, err = _do_one(cfg)
            total += 1
            if success:
                local_ok += 1
                local_lat.append(latency)
            else:
                local_fail += 1
                local_err[err] = local_err.get(err, 0) + 1
        return local_ok, local_fail, local_lat, local_err

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        last_report = time.perf_counter()
        for f in as_completed(futures):
            lok, lfail, llat, lerr = f.result()
            ok += lok
            fail += lfail
            latencies.extend(llat)
            for k, v in lerr.items():
                error_counter[k] = error_counter.get(k, 0) + v
            now = time.perf_counter()
            if now - last_report > 10:
                print(f"  STAB-004 进度: 已完成={ok+fail}, 成功={ok}, 失败={fail}")
                last_report = now

    error_rate = fail / total if total else 1
    error_summary = ", ".join(f"{k}x{v}" for k, v in error_counter.items()) or "无"
    passed = error_rate < cfg.error_rate_threshold
    p95 = percentile(latencies, 95) if latencies else 0
    return TestResult(
        "STAB-004", "并发稳定性", passed,
        f"并发={concurrency}, 时长={duration}s, 总请求={total}, 成功={ok}, 失败={fail}, 错误率={error_rate:.4%}(阈值<{cfg.error_rate_threshold:.2%}), latency p95={p95:.3f}s, 失败原因: {error_summary}",
        {
            "concurrency": concurrency, "duration_s": duration,
            "total": total, "success": ok, "fail": fail,
            "error_rate": round(error_rate, 5),
            "latency_p95_s": round(p95, 3),
            "errors": error_counter,
        },
    )


# ============================================================
# STAB-005 服务限流
# ============================================================

def test_stab_005_rate_limit(cfg: Config) -> TestResult:
    """超并发突发请求，验证是否返回 429"""
    burst = cfg.rate_limit_burst
    payload = {
        "messages": build_messages("用一个词回答：你好。"),
        "max_tokens": 10,
    }

    def one():
        t0 = time.perf_counter()
        try:
            resp = chat_request(cfg, payload)
            return resp.status_code, time.perf_counter() - t0
        except Exception as e:
            return -1, time.perf_counter() - t0

    status_counter: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=burst) as pool:
        futures = [pool.submit(one) for _ in range(burst)]
        for f in as_completed(futures):
            code, _ = f.result()
            status_counter[code] = status_counter.get(code, 0) + 1

    rate_limited = status_counter.get(429, 0)
    has_other_success = status_counter.get(200, 0)
    summary = ", ".join(f"HTTP {k}x{v}" for k, v in sorted(status_counter.items()))
    # 通过标准：出现了 429 限流响应（说明限流机制生效）
    passed = rate_limited > 0
    return TestResult(
        "STAB-005", "服务限流", passed,
        f"突发={burst}, 响应分布: {summary}, 429限流={rate_limited}个",
        {
            "burst": burst,
            "status_distribution": status_counter,
            "rate_limited_count": rate_limited,
        },
    )


# ============================================================
# 运行入口
# ============================================================

TEST_FUNCS = {
    "STAB-001": test_stab_001_continuous,
    "STAB-002": test_stab_002_long_run,
    "STAB-003": test_stab_003_error_recovery,
    "STAB-004": test_stab_004_sustain_concurrency,
    "STAB-005": test_stab_005_rate_limit,
}


def parse_args() -> tuple[Config, list[str] | None]:
    parser = argparse.ArgumentParser(description="API 稳定性测试脚本 (STAB-001 ~ STAB-005)")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时(秒)")
    parser.add_argument("--continuous-count", type=int, default=100, help="STAB-001 连续请求次数")
    parser.add_argument("--long-duration", type=int, default=300, help="STAB-002 长时间运行时长(秒)，规范要求86400(24h)")
    parser.add_argument("--sustain-concurrency", type=int, default=20, help="STAB-004 持续并发数")
    parser.add_argument("--sustain-duration", type=int, default=600, help="STAB-004 持续压测时长(秒)")
    parser.add_argument("--rate-limit-burst", type=int, default=50, help="STAB-005 限流探测并发数")
    parser.add_argument("--only", help="只跑指定用例，逗号分隔，如 STAB-001,STAB-003")
    args = parser.parse_args()

    cfg = Config(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=args.timeout,
        enable_thinking=ENABLE_THINKING,
        continuous_count=args.continuous_count,
        long_duration_s=args.long_duration,
        sustain_concurrency=args.sustain_concurrency,
        sustain_duration_s=args.sustain_duration,
        rate_limit_burst=args.rate_limit_burst,
    )

    only = [x.strip().upper() for x in args.only.split(",")] if args.only else None
    return cfg, only


def main() -> int:
    cfg, only = parse_args()
    print(f"\n{'='*60}")
    print(f"稳定性测试开始")
    print(f"Base URL : {cfg.base_url}")
    print(f"Model    : {cfg.model}")
    print(f"深度思考 : {'开启' if cfg.enable_thinking else '关闭'}")
    print(f"连续请求 : {cfg.continuous_count}次")
    print(f"长时运行 : {cfg.long_duration_s}s (规范24h=86400)")
    print(f"持续并发 : {cfg.sustain_concurrency}并发 / {cfg.sustain_duration_s}s")
    print(f"限流探测 : {cfg.rate_limit_burst}并发")
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
    report_path = "stab_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已写入: {report_path}")

    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
