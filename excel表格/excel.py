import json
import pandas as pd

# 测试数据
test_categories = [
    {
        "category": "1. 基础连通性测试",
        "tests": [
            {"id": "CONN-001", "item": "API端点连通性", "description": "验证API地址可访问", "method": "curl/请求访问", "expected": "返回200/成功响应", "priority": "P0"},
            {"id": "CONN-002", "item": "Token认证", "description": "验证Bearer Token鉴权", "method": "带正确Token请求", "expected": "正常返回", "priority": "P0"},
            {"id": "CONN-003", "item": "鉴权失败测试", "description": "无Token/错误Token", "method": "不带Token请求", "expected": "返回401未授权", "priority": "P0"},
            {"id": "CONN-004", "item": "模型列表查询", "description": "获取可用模型列表", "method": "GET /v1/models", "expected": "返回模型信息", "priority": "P0"},
        ]
    },
    {
        "category": "2. 功能性测试",
        "tests": [
            {"id": "FUNC-001", "item": "基础对话", "description": "单轮问答功能", "method": "POST /v1/chat/completions", "expected": "正常返回回复", "priority": "P0"},
            {"id": "FUNC-002", "item": "多轮对话", "description": "上下文保持能力", "method": "连续多轮对话", "expected": "上下文正确关联", "priority": "P0"},
            {"id": "FUNC-003", "item": "流式输出", "description": "Server-Sent Events", "method": "stream=true", "expected": "SSE流式返回", "priority": "P0"},
            {"id": "FUNC-004", "item": "系统提示词", "description": "System Prompt设置", "method": "设置system消息", "expected": "按角色设定回复", "priority": "P1"},
            {"id": "FUNC-005", "item": "JSON输出", "description": "结构化JSON响应", "method": "response_format=json", "expected": "返回有效JSON", "priority": "P1"},
            {"id": "FUNC-006", "item": "工具调用-Tool Call", "description": "函数调用能力", "method": "tools参数调用", "expected": "正确触发工具调用", "priority": "P1"},
            {"id": "FUNC-007", "item": "长文本处理", "description": "长上下文理解", "method": "输入超长文本", "expected": "正确理解处理", "priority": "P1"},
            {"id": "FUNC-008", "item": "图片多模态(如有)", "description": "图像理解能力", "method": "传入图片URL/base64", "expected": "正确理解图片", "priority": "P1"},
        ]
    },
    {
        "category": "3. 性能测试",
        "tests": [
            {"id": "PERF-001", "item": "首Token响应时间(TTFT)", "description": "首个token出现时间", "method": "time to first token", "expected": "<2s (短文本)", "priority": "P0"},
            {"id": "PERF-002", "item": "端到端响应时间", "description": "完整请求响应时间", "method": "request_latency", "expected": "根据输出长度评估", "priority": "P0"},
            {"id": "PERF-003", "item": "吞吐量(TPS)", "description": "Tokens Per Second", "method": "输出token数/耗时", "expected": ">30 TPS", "priority": "P0"},
            {"id": "PERF-004", "item": "并发能力", "description": "支持同时请求数", "method": "多线程并发测试", "expected": "达到声称并发数", "priority": "P1"},
            {"id": "PERF-005", "item": "并发响应时间", "description": "高并发下响应时间", "method": "10/50/100并发", "expected": "响应时间合理增长", "priority": "P1"},
            {"id": "PERF-006", "item": "最大输出长度", "description": "单次最大输出token", "method": "max_tokens参数", "expected": "达到声称上限", "priority": "P1"},
            {"id": "PERF-007", "item": "最大输入长度", "description": "单次最大输入token", "method": "长文本输入测试", "expected": "支持声称上下文", "priority": "P1"},
        ]
    },
    {
        "category": "4. 稳定性测试",
        "tests": [
            {"id": "STAB-001", "item": "连续请求稳定性", "description": "连续100次请求", "method": "循环请求", "expected": "无错误/崩溃", "priority": "P0"},
            {"id": "STAB-002", "item": "24小时长时间运行", "description": "持续请求测试", "method": "压测24小时", "expected": "无内存泄漏/崩溃", "priority": "P1"},
            {"id": "STAB-003", "item": "错误恢复", "description": "异常请求后恢复", "method": "发送异常请求", "expected": "自动恢复", "priority": "P1"},
            {"id": "STAB-004", "item": "并发稳定性", "description": "高并发持续压测", "method": "100并发压测10分钟", "expected": "无错误率<0.1%", "priority": "P1"},
            {"id": "STAB-005", "item": "服务限流", "description": "超过限流阈值", "method": "超并发请求", "expected": "返回429", "priority": "P1"},
        ]
    },
    {
        "category": "5. 大模型/长任务连续性测试 [Agent监控]",
        "tests": [
            {"id": "LONG-001", "item": "超长输出任务", "description": "生成超长文本(50K+ tokens)", "method": "Agent监控脚本: max_tokens=100000", "expected": "稳定输出不中断", "priority": "P0"},
            {"id": "LONG-002", "item": "长时间推理任务", "description": "单次请求耗时>5分钟", "method": "Agent: 监控推理时长", "expected": "完成不超时", "priority": "P0"},
            {"id": "LONG-003", "item": "复杂代码生成", "description": "生成完整项目代码", "method": "Agent: 代码生成任务", "expected": "代码完整可运行", "priority": "P1"},
            {"id": "LONG-004", "item": "长对话上下文", "description": "30+轮对话保持上下文", "method": "Agent: 多轮对话监控", "expected": "上下文不丢失", "priority": "P1"},
            {"id": "LONG-005", "item": "文档摘要任务", "description": "长文档摘要处理", "method": "Agent: 输入长文档", "expected": "正确摘要", "priority": "P1"},
            {"id": "LONG-006", "item": "Agent多步骤任务", "description": "多轮Tool Call任务链", "method": "Agent: 自主调用工具", "expected": "任务链完整执行", "priority": "P1"},
            {"id": "LONG-007", "item": "断点续传测试", "description": "长任务中断后恢复", "method": "Agent: 模拟中断重试", "expected": "从断点继续", "priority": "P1"},
            {"id": "LONG-008", "item": "资源监控-内存", "description": "长任务内存使用监控", "method": "Agent: 监控内存峰值", "expected": "无内存溢出", "priority": "P1"},
            {"id": "LONG-009", "item": "资源监控-GPU", "description": "GPU利用率监控", "method": "Agent: nvidia-smi监控", "expected": "GPU利用率合理", "priority": "P1"},
            {"id": "LONG-010", "item": "长任务成本核算", "description": "长任务Token计费", "method": "Agent: 对比计费明细", "expected": "计费准确", "priority": "P1"},
        ]
    },
    {
        "category": "6. 准确性测试",
        "tests": [
            {"id": "ACC-001", "item": "数学推理", "description": "数学计算能力", "method": "数学题测试集", "expected": "正确率>80%", "priority": "P1"},
            {"id": "ACC-002", "item": "代码生成", "description": "代码编写能力", "method": "编程题测试集", "expected": "可执行/逻辑正确", "priority": "P1"},
            {"id": "ACC-003", "item": "知识准确性", "description": "知识问答正确性", "method": "标准问答测试集", "expected": "准确率达标", "priority": "P1"},
            {"id": "ACC-004", "item": "指令遵循", "description": "按要求执行能力", "method": "复杂指令测试", "expected": "正确执行", "priority": "P1"},
            {"id": "ACC-005", "item": "内容安全性", "description": "敏感内容过滤", "method": "违规测试用例", "expected": "正确拒绝/脱敏", "priority": "P0"},
        ]
    },
    {
        "category": "7. 成本测试",
        "tests": [
            {"id": "COST-001", "item": "输入Token计费", "description": "验证输入token计费", "method": "请求后对比计费", "expected": "计费准确", "priority": "P0"},
            {"id": "COST-002", "item": "输出Token计费", "description": "验证输出token计费", "method": "请求后对比计费", "expected": "计费准确", "priority": "P0"},
            {"id": "COST-003", "item": "计费明细查询", "description": "获取计费详情API", "method": "usage API", "expected": "返回详细计费", "priority": "P1"},
            {"id": "COST-004", "item": "免费额度测试", "description": "验证免费额度", "method": "新账号测试", "expected": "免费额正确", "priority": "P1"},
        ]
    },
    {
        "category": "8. 错误处理测试",
        "tests": [
            {"id": "ERR-001", "item": "无效模型名", "description": "错误模型参数", "method": "model参数错误", "expected": "返回400错误", "priority": "P0"},
            {"id": "ERR-002", "item": "参数格式错误", "description": "JSON格式错误", "method": "发送畸形JSON", "expected": "返回400错误", "priority": "P0"},
            {"id": "ERR-003", "item": "超长输入", "description": "超过最大输入限制", "method": "输入超长文本", "expected": "返回错误", "priority": "P1"},
            {"id": "ERR-004", "item": "服务端错误", "description": "500等服务器错误", "method": "触发异常", "expected": "返回正确错误码", "priority": "P1"},
        ]
    },
    {
        "category": "9. 协议兼容性测试",
        "tests": [
            {"id": "COMP-001", "item": "OpenAI兼容", "description": "OpenAI API兼容", "method": "标准OpenAI SDK", "expected": "正常调用", "priority": "P0"},
            {"id": "COMP-002", "item": "SDK支持", "description": "各语言SDK测试", "method": "Python/Go/JS SDK", "expected": "正常工作", "priority": "P1"},
        ]
    }
]

# 整理数据
rows = []
for cat in test_categories:
    category = cat["category"]
    for test in cat["tests"]:
        rows.append({
            "分类": category,
            "用例ID": test["id"],
            "测试项": test["item"],
            "测试描述": test["description"],
            "测试方法": test["method"],
            "预期结果": test["expected"],
            "优先级": test["priority"],
            "测试结果": "",  # 留空给手动测试填写
            "备注": ""      # 留空给手动测试填写
        })

# 生成DataFrame并保存
df = pd.DataFrame(rows)
df.to_excel("LLM_API测试用例.xlsx", index=False, engine='openpyxl')
print("Excel文件已生成: LLM_API测试用例.xlsx")