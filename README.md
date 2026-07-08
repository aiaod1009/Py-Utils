# API 测试套件

针对 OpenAI 兼容 API 的自动化测试脚本集合，覆盖性能、稳定性、超长上下文等维度。

## 项目结构

```
test/
├── .env                # API 配置（Base URL、Key、Model）
├── .gitignore          # Git 忽略规则
├── test.json           # 测试用例清单（9 大类 50+ 用例）
├── perf_test.py        # 性能测试脚本 (PERF-001 ~ 007)
├── stab_test.py        # 稳定性测试脚本 (STAB-001 ~ 005)
├── longctx_test.py     # 超长上下文测试脚本 (LONG-CTX-001 ~ 006)
├── *_report.json       # 测试报告（自动生成，不提交 Git）
```

## 快速开始

### 1. 安装依赖

```bash
pip install python-dotenv requests
```

### 2. 配置 API

编辑 `.env` 文件，填入你的 API 信息：

```env
BASE_URL=https://your-api.example.com/v1
API_KEY=sk-your-key-here
MODEL=gpt-4o
ENABLE_THINKING=false
```

### 3. 运行测试

```bash
# 性能测试
python perf_test.py

# 稳定性测试
python stab_test.py

# 超长上下文测试（默认档 200K 输入）
python longctx_test.py

# 超长上下文测试（满档 1M 输入 / 128K 输出）
python longctx_test.py --full
```

## 测试脚本详解

### perf_test.py — 性能测试

覆盖 `test.json` 中 "3. 性能测试" 的 7 个用例：

| 用例 | 测试内容 | 指标 |
|------|---------|------|
| PERF-001 | 首 Token 响应时间 (TTFT) | p95 < 2s |
| PERF-002 | 端到端响应时间 (TPOT) | p95 < 0.1s/token |
| PERF-003 | 吞吐量 (TPS) | > 30 tokens/s |
| PERF-004 | 并发能力 | 成功率 ≥ 95% |
| PERF-005 | 并发响应时间 | p95 合理增长 |
| PERF-006 | 最大输出长度 | 覆盖率 ≥ 90% |
| PERF-007 | 最大输入长度 | 能接受超长输入 |

**常用参数：**

```bash
python perf_test.py --concurrency 5,10,20    # 并发档位
python perf_test.py --sample-count 20        # TTFT/TPOT 采样次数
python perf_test.py --only PERF-001,PERF-003 # 只跑指定用例
```

### stab_test.py — 稳定性测试

覆盖 `test.json` 中 "4. 稳定性测试" 的 5 个用例：

| 用例 | 测试内容 | 通过标准 |
|------|---------|---------|
| STAB-001 | 连续请求稳定性 | 100 次请求 0 失败 |
| STAB-002 | 长时间运行 | 持续 N 秒无崩溃 |
| STAB-003 | 错误恢复 | 异常请求后能恢复正常 |
| STAB-004 | 并发稳定性 | 错误率 < 0.1% |
| STAB-005 | 服务限流 | 能触发 429 响应 |

**常用参数：**

```bash
python stab_test.py --continuous-count 100   # 连续请求次数
python stab_test.py --long-duration 300      # 长时运行时长(秒)
python stab_test.py --sustain-concurrency 20 # 持续并发数
python stab_test.py --sustain-duration 600   # 持续压测时长(秒)
```

### longctx_test.py — 超长上下文测试

验证模型声称的上下文上限是否属实，以及能否真正"看到"超长文本中的信息：

| 用例 | 测试内容 | 说明 |
|------|---------|------|
| LONG-CTX-001 | 最大输入长度验证 | 能否接受 ~1M token 输入 |
| LONG-CTX-002 | 最大输出长度验证 | 能否输出到 ~128K token |
| LONG-CTX-003 | 大海捞针(开头) | 针在长文开头，验证检索 |
| LONG-CTX-004 | 大海捞针(中间) | **最关键**，验证中间不遗忘 |
| LONG-CTX-005 | 大海捞针(末尾) | 针在长文末尾，验证检索 |
| LONG-CTX-006 | 跨上下文关联 | 不同位置的线索能否串联推理 |

**常用参数：**

```bash
python longctx_test.py --full                        # 满档 1M/128K
python longctx_test.py --target-input-tokens 500000  # 自定义输入目标
python longctx_test.py --only LONG-CTX-004           # 只跑中间埋针
```

## 测试报告

每次运行后自动生成 JSON 报告：

- `perf_report.json` — 性能测试结果
- `stab_report.json` — 稳定性测试结果
- `longctx_report.json` — 超长上下文测试结果

报告包含：
- 总体通过率
- 每个用例的通过状态、详细描述、具体指标

报告文件已加入 `.gitignore`，不会提交到 Git。

## test.json 测试清单

完整的测试用例定义，包含 9 大类：

1. 基础连通性测试 (CONN-001 ~ 004)
2. 功能性测试 (FUNC-001 ~ 008)
3. 性能测试 (PERF-001 ~ 007) — 由 `perf_test.py` 实现
4. 稳定性测试 (STAB-001 ~ 005) — 由 `stab_test.py` 实现
5. 大模型/长任务连续性测试 (LONG-001 ~ 010)
6. 准确性测试 (ACC-001 ~ 005)
7. 成本测试 (COST-001 ~ 004)
8. 错误处理测试 (ERR-001 ~ 004)
9. 协议兼容性测试 (COMP-001 ~ 002)

目前已实现第 3、4 类的自动化脚本，其余类别可按需扩展。

## 注意事项

- `.env` 文件包含 API Key，请勿提交到公开仓库
- 超长上下文测试（满档）耗时较长，建议先用默认档验证
- 并发测试可能触发限流（429），属于正常现象
- 深度思考（thinking）功能默认关闭，避免干扰性能指标