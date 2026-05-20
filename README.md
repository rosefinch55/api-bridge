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
- thinking/reasoning 内容透传

### 2. Anthropic 直传（以小米为例）

适用于原生 Anthropic 兼容接口（小米 mimo、第三方代理等）。

```
Claude Code  →  apibridge  →  小米 mimo (Anthropic 协议)
(Anthropic)     透传           (mimo-v2.5 / mimo-v2.5-pro)
```

请求直接转发，不做格式转换。

## 项目结构

```
├── server.py          # 服务入口（路由 + 启动）
├── config.py          # 配置模块（上游、模型映射、env 加载）
├── convert.py         # 协议转换（Anthropic ↔ OpenAI）
├── stream.py          # 流式响应处理
├── config.yaml        # 上游配置文件
├── .env.example       # 环境变量模板
└── requirements.txt   # Python 依赖
```

## 快速启动

### 方式一：使用 config.yaml

编辑 `config.yaml` 填入你的 API key：

```yaml
upstream:
  url: "https://api.chat.csu.edu.cn/v1/chat/completions"
  key: "sk-xxxxx"
  model: "DeepSeek-V4-Flash"

bridge:
  port: 4000
```

然后启动：

```bash
pip install -r requirements.txt
python server.py
```

### 方式二：使用 .env

复制 `.env.example` 为 `.env`，填入配置后启动。

### 方式三：使用 bat 脚本

```bash
# CSU DeepSeek
start-csu.bat

# CSU Qwen
start-csu-qwen.bat

# 小米 mimo（直传模式）
start-mimo.bat
```

bat 脚本会自动启动 bridge 服务（端口 4000），然后启动 Claude Code。

## 如何替换后端

### config.yaml 方式

直接编辑 `config.yaml` 中的 `upstream` 配置。

### .env 方式

```bash
UPSTREAM_URL=https://api.example.com/v1/chat/completions
UPSTREAM_KEY=sk-xxxxx
UPSTREAM_MODEL=your-model-name
```

### 多上游（进阶）

如果需要多个上游和模型映射，请使用完整版 `config.py`，参考 `MODEL_MAP` 配置。

## 端点

| 路径 | 说明 |
|------|------|
| `GET /` | 健康检查 |
| `POST /v1/messages` | 主代理端点 |
| `GET /v1/models` | 模型列表 |

## 依赖

```bash
pip install -r requirements.txt
```
