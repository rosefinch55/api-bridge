"""协议转换模块：Anthropic ↔ OpenAI 格式互转"""

import json
import uuid
from config import MODEL_MAP, DEFAULT_MODEL, ENABLE_THINKING


def convert_tools(tools: list) -> list:
    """Anthropic tools → OpenAI tools"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
        }
        for tool in tools
    ]


def _convert_system(system) -> dict | None:
    """转换 system 消息"""
    if not system:
        return None
    if isinstance(system, list):
        text = "\n".join(b.get("text", "") for b in system if b.get("type") == "text")
    else:
        text = system
    return {"role": "system", "content": text}


def _convert_assistant_message(content: list) -> dict:
    """转换 assistant 消息（可能包含 tool_use）"""
    text_parts = []
    tool_calls = []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                }
            })
    msg = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_user_message(content: list) -> list[dict]:
    """转换 user 消息（可能包含 text/image/tool_result）"""
    text_parts = []
    image_parts = []
    tool_msgs = []

    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block["text"])
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                media_type = src.get("media_type", "image/png")
                data = src.get("data", "")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"}
                })
        elif btype == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_text = "\n".join(
                    b.get("text", "") for b in result_content if b.get("type") == "text"
                )
            else:
                result_text = str(result_content)
            is_error = block.get("is_error", False)
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": f"[Error] {result_text}" if is_error else result_text,
            })

    result = []
    if image_parts:
        msg_content = []
        if text_parts:
            msg_content.append({"type": "text", "text": "\n".join(text_parts)})
        msg_content.extend(image_parts)
        result.append({"role": "user", "content": msg_content})
    elif text_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    result.extend(tool_msgs)
    return result


def anthropic_to_openai(body: dict) -> dict:
    """Anthropic messages API → OpenAI chat completions API"""
    messages = []

    # system
    sys_msg = _convert_system(body.get("system", ""))
    if sys_msg:
        messages.append(sys_msg)

    # messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "assistant":
                messages.append(_convert_assistant_message(content))
            elif role == "user":
                messages.extend(_convert_user_message(content))
            else:
                messages.append({"role": role, "content": str(content)})
        else:
            messages.append({"role": role, "content": str(content)})

    # 构建请求
    req_model = body.get("model", DEFAULT_MODEL)
    upstream_model = MODEL_MAP.get(req_model, req_model)

    openai_req = {
        "model": upstream_model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
    }

    # 流式时请求返回 usage
    if openai_req["stream"]:
        openai_req["stream_options"] = {"include_usage": True}

    # thinking 模式（仅 DeepSeek）
    upstream_map = MODEL_MAP.get(req_model, (None, None))
    upstream_model_name = upstream_map[1] if upstream_map else req_model
    is_deepseek = upstream_model_name and "DeepSeek" in upstream_model_name
    is_thinking = "-thinking" in req_model
    if not is_thinking:
        thinking = body.get("thinking", {})
        is_thinking = thinking.get("type") == "enabled"
    if ENABLE_THINKING and is_thinking and is_deepseek:
        openai_req["enable_thinking"] = True
        openai_req["budget_tokens"] = body.get("thinking", {}).get("budget_tokens", 4096)

    # 工具
    if "tools" in body:
        openai_req["tools"] = convert_tools(body["tools"])

    # tool_choice
    if "tool_choice" in body:
        tc = body["tool_choice"]
        tc_type = tc.get("type", "auto")
        tc_map = {
            "auto": "auto",
            "any": "required",
            "none": "none",
        }
        if tc_type in tc_map:
            openai_req["tool_choice"] = tc_map[tc_type]
        elif tc_type == "tool":
            openai_req["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}

    # 其他参数
    for key in ("temperature", "top_p"):
        if key in body:
            openai_req[key] = body[key]
    if "stop_sequences" in body:
        openai_req["stop"] = body["stop_sequences"]

    return openai_req


def openai_response_to_anthropic(openai_resp: dict, model: str) -> dict:
    """OpenAI chat completion → Anthropic message"""
    choice = openai_resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")

    content_blocks = []

    # reasoning
    reasoning = msg.get("reasoning_content", "")
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    # text
    text = msg.get("content", "")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # tool_calls
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": args,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }

    usage = openai_resp.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    anthropic_usage.update({k: v for k, v in usage.items() if k not in ("prompt_tokens", "completion_tokens")})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason_map.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": anthropic_usage,
    }
