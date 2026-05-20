"""流式响应模块：OpenAI SSE → Anthropic SSE 格式转换"""

import json
import uuid
import httpx
from config import REQUEST_TIMEOUT


def make_sse(event_type: str, data: dict) -> str:
    """生成 SSE 格式行"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def anthropic_passthrough(url: str, body: dict, headers: dict):
    """Anthropic 协议流式透传（mimo 等原生 Anthropic 上游）"""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"


async def openai_to_anthropic_stream(openai_req: dict, model: str, headers: dict, upstream_url: str):
    """OpenAI 流式响应 → Anthropic SSE 格式"""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield make_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    })

    tool_calls_buf = {}
    block_index = 0
    current_block_type = None
    finish_reason = None
    usage_data = {}

    def _switch_block(new_type: str, **kwargs):
        """切换 content block 类型"""
        nonlocal current_block_type, block_index
        if current_block_type == new_type:
            return
        if current_block_type:
            yield make_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
        current_block_type = new_type
        if new_type == "thinking":
            yield make_sse("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            })
        elif new_type == "text":
            yield make_sse("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "text", "text": ""},
            })
        elif new_type == "tool_use":
            yield make_sse("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "tool_use", "id": kwargs["tool_id"], "name": kwargs["name"], "input": {}},
            })

    def _emit_block_switch(new_type: str, **kwargs):
        """同步包装：yield from _switch_block"""
        return _switch_block(new_type, **kwargs)

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", upstream_url, json=openai_req, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # 收集 usage（最后一个空 choices chunk）
                    if chunk.get("usage"):
                        usage_data = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason") or finish_reason

                    # reasoning（DeepSeek reasoning_content / Qwen reasoning）
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        for ev in _switch_block("thinking"):
                            yield ev
                        yield make_sse("content_block_delta", {
                            "type": "content_block_delta", "index": block_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        })

                    # text
                    text = delta.get("content")
                    if text:
                        for ev in _switch_block("text"):
                            yield ev
                        yield make_sse("content_block_delta", {
                            "type": "content_block_delta", "index": block_index,
                            "delta": {"type": "text_delta", "text": text},
                        })

                    # tool_calls
                    for tc_delta in delta.get("tool_calls", []):
                        tc_idx = tc_delta.get("index", 0)
                        if tc_idx not in tool_calls_buf:
                            tc_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                            tc_name = tc_delta.get("function", {}).get("name", "")
                            tool_calls_buf[tc_idx] = {"id": tc_id, "name": tc_name, "arguments": ""}
                            for ev in _switch_block("tool_use", tool_id=tc_id, name=tc_name):
                                yield ev
                        args_chunk = tc_delta.get("function", {}).get("arguments", "")
                        if args_chunk:
                            tool_calls_buf[tc_idx]["arguments"] += args_chunk
                            yield make_sse("content_block_delta", {
                                "type": "content_block_delta", "index": block_index,
                                "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                            })

    except Exception as e:
        for ev in _switch_block("text"):
            yield ev
        yield make_sse("content_block_delta", {
            "type": "content_block_delta", "index": block_index,
            "delta": {"type": "text_delta", "text": f"[Error: {e}]"},
        })

    # 关闭最后一个 block
    if current_block_type:
        yield make_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

    # stop_reason + usage
    stop = "tool_use" if (finish_reason == "tool_calls" and tool_calls_buf) else "end_turn"
    yield make_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        },
    })
    yield make_sse("message_stop", {"type": "message_stop"})
