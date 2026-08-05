# DeepSeek-V4-Flash 服务访问问题排查

## 现象

运行 `DeepSeek-V4-Flash_capability_test.py` 时报错：

```
ClientConnectorError: Cannot connect to host 192.168.202.3:8002
```

脚本启动后预检失败，无法正常测试模型能力。

---

## 排查过程

### 1. 容器是否运行

```bash
docker ps
```

| 结果 |
|------|
| ✅ 容器 `DeepSeek-V4-Flash-W8A8` 正常运行，状态 Up 3 weeks |

### 2. 容器内 vLLM 服务是否正常

```bash
docker exec DeepSeek-V4-Flash-W8A8 python3 -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8002/v1/models').read().decode()[:500])"
```

返回结果：

```json
{
  "object": "list",
  "data": [{
    "id": "/model/DeepSeek-V4-flash",
    "object": "model",
    "max_model_len": 262144
    ...
  }]
}
```

| 结果 |
|------|
| ✅ 容器内 vLLM 服务正常，API 可正常调用 |

### 3. 容器端口映射检查

```bash
docker port DeepSeek-V4-Flash-W8A8
```

| 结果 |
|------|
| ❌ 无任何输出，说明没有端口映射 |

### 4. 宿主机连通性测试

在宿主机（192.168.202.3）上执行：

```bash
curl http://localhost:8002/v1/models
```

| 结果 |
|------|
| ❌ 连接失败 |

### 5. vLLM 启动参数检查

```bash
docker exec DeepSeek-V4-Flash-W8A8 ps aux | grep vllm
```

启动命令为：

```
vllm serve /model/DeepSeek-V4-flash \
  --trust-remote-code \
  --port 8002 \
  -tp 8 \
  ...
```

启动命令中指定了 `--port 8002`，但未看到 `--host 0.0.0.0`。不过 vLLM 默认监听 `0.0.0.0`，所以当前问题主要是端口映射缺失。

---

## 根因

**容器启动时缺少 `-p 8002:8002` 端口映射参数。**

- vLLM 服务在容器内正常监听 `127.0.0.1:8002`
- 但由于未将容器端口映射到宿主机，外部机器无法访问
- 脚本在宿主机或其它机器上运行时自然连接不上

---

## 修复方案

停掉并删除当前容器，重新创建时加上端口映射：

```bash
# 1. 停掉当前容器
docker stop DeepSeek-V4-Flash-W8A8

# 2. 删除当前容器
docker rm DeepSeek-V4-Flash-W8A8

# 3. 重新启动（注意加上 -p 8002:8002）
docker run -d \
  --gpus all \
  --name DeepSeek-V4-Flash-W8A8 \
  -p 8002:8002 \
  -v /model:/model \
  -v /share:/share \
  <镜像ID或名称> \
  /bin/bash -c "tail -f /dev/null"
```

> **注意**：以上命令仅为示例，请根据原始启动命令补充完整的 `-v` 挂载和其它参数。

### 启动 vLLM 服务（容器内）

进容器启动 vLLM 时，可显式指定 `--host 0.0.0.0` 避免后续问题：

```bash
docker exec DeepSeek-V4-Flash-W8A8 bash -c "
nohup vllm serve /model/DeepSeek-V4-flash \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8002 \
  --kv-cache-dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  -tp 8 \
  > vllm.log 2>&1 &
"
```

### 验证

```bash
# 宿主机上验证
curl http://localhost:8002/v1/models

# 其它机器上验证（替换为实际IP）
curl http://192.168.202.3:8002/v1/models
```

如果返回正常 JSON，说明修复成功，可以重新运行测试脚本。

---

## 相关文件

- 测试脚本：`DeepSeek-V4-Flash_capability_test.py`
- 参考脚本：`MiniMax-M2.5_capability_test.py`（可正常运行的对比脚本）
- 运行日志：`bench.log`
