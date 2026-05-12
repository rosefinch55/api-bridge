# API Bridge

Claude Code CLI 协议桥接服务。接收 Anthropic 格式请求，转发到任意后端 API。

## 两种模式

### 1. OpenAI 兼容 → Anthropic（以 CSU 为例）

适用于 OpenAI 兼容接口（DeepSeek、Qwen、vLLM 等）。

```
Claude Code  →  apibridge  →  CSU (OpenAI 协议)
(Anthropic)     转换层        (DeepSeek / Qwen)
```

自动处理：
- 消息格式转换（system / tool_use / tool_result / 图片）
- 工具定义转换（input_schema ↔ parameters）
- 流式 SSE 增量同步

### 2. Anthropic 直传（以小米为例）

适用于原生 Anthropic 兼容接口（小米 mimo、第三方代理等）。

```
Claude Code  →  apibridge  →  小米 mimo (Anthropic 协议)
(Anthropic)     透传           (mimo-v2.5 / mimo-v2.5-pro)
```

请求直接转发，不做格式转换。

## 快速启动

```bash
# CSU DeepSeek（OpenAI 转换模式）
start-csu.bat

# CSU Qwen（OpenAI 转换模式）
start-csu-qwen.bat

# 小米 mimo v2.5（直传模式，无需启动 bridge）
start-mimo.bat

# 小米 mimo v2.5-pro（直传模式，无需启动 bridge）
start-xiaomi.bat
```

CSU 系列脚本会自动启动 bridge 服务（端口 4000），然后启动 Claude Code。
小米系列直连，不经过 bridge。

## 如何替换后端

编辑 `server.py` 中的两个配置：

### UPSTREAMS — 添加上游

```python
UPSTREAMS = {
    "my-provider": {
        "url": "https://api.example.com/v1/chat/completions",
        "key": "YOUR_API_KEY",
        "protocol": "openai",      # "openai" 需要转换，"anthropic" 直传
    },
}
```

### MODEL_MAP — 映射模型名

```python
MODEL_MAP = {
    "my-model": ("my-provider", "actual-model-name"),
}
```

然后写个 bat 脚本：

```bat
@echo off
start "API Bridge" python server.py
timeout /t 3 >nul
set ANTHROPIC_BASE_URL=http://localhost:4000
set ANTHROPIC_AUTH_TOKEN=YOUR_API_KEY
set ANTHROPIC_MODEL=my-model
claude
```

## 端点

| 路径 | 说明 |
|------|------|
| `GET /` | 健康检查 |
| `POST /v1/messages` | 主代理端点 |
| `GET /v1/models` | 模型列表 |

## 依赖

```bash
pip install fastapi uvicorn httpx
```
