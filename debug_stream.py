"""调试脚本：检查流式响应结构，看看 content 是否走 reasoning_content 字段"""
import json
import time
import requests
import os

# 加载 .env
env = {}
env_path = os.path.join(os.path.dirname(__file__), ".env")
with open(env_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

MAIGU_BASE_URL = env.get("MAIGU_BASE_URL", "https://ka.ent-pilot.com")
MAIGU_API_KEY = env.get("MAIGU_API_KEY", "")
STS_BASE_URL = env.get("STS_BASE_URL", "https://www.sts-token.com/v1")
STS_API_KEY = env.get("STS_API_KEY", "")


def build_chat_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def debug_call(endpoint_name, base_url, api_key, model, prompt, max_tokens, enable_thinking):
    url = build_chat_url(base_url)
    print(f"\n=== {endpoint_name} / {model} / enable_thinking={enable_thinking} ===")
    print(f"URL: {url}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "enable_thinking": enable_thinking,
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    print(f"Status: {resp.status_code}")

    chunk_count = 0
    content_chunks = 0
    reasoning_chunks = 0
    sample_chunks = []
    full_content = ""
    full_reasoning = ""
    usage_data = {}

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        chunk_count += 1
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            if content:
                content_chunks += 1
                full_content += content
            if reasoning:
                reasoning_chunks += 1
                full_reasoning += reasoning
            if chunk_count <= 3 or (content and len(sample_chunks) < 5):
                sample_chunks.append(chunk)
        if "usage" in chunk:
            usage_data = chunk.get("usage", {}) or {}

    print(f"总 chunks: {chunk_count}")
    print(f"content chunks: {content_chunks}  reasoning chunks: {reasoning_chunks}")
    print(f"usage: {json.dumps(usage_data, ensure_ascii=False)}")
    print(f"full_content (前200): {full_content[:200]!r}")
    print(f"full_reasoning (前200): {full_reasoning[:200]!r}")
    if sample_chunks:
        print(f"首 chunk 示例: {json.dumps(sample_chunks[0], ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    prompt = "请用一句话介绍人工智能。"
    # 4K 场景的模型，max_tokens=1024
    # 先测麦谷 glm-5.2 with enable_thinking=False
    debug_call("麦谷上游", MAIGU_BASE_URL, MAIGU_API_KEY, "glm-5.2", prompt, 1024, False)
    # 再测 STS glm-5.2 with enable_thinking=False
    debug_call("STS中转", STS_BASE_URL, STS_API_KEY, "glm-5.2", prompt, 1024, False)
    # 再测麦谷 glm-5.1
    debug_call("麦谷上游", MAIGU_BASE_URL, MAIGU_API_KEY, "glm-5.1", prompt, 1024, False)
