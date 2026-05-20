"""API Bridge 服务入口（公开版）：单上游，Anthropic 协议 ↔ OpenAI 兼容协议"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from config import (
    UPSTREAM_URL, UPSTREAM_MODEL, PORT, REQUEST_TIMEOUT,
    resolve_model, get_upstream_headers, get_upstream_url,
)
from convert import anthropic_to_openai, openai_response_to_anthropic
from stream import anthropic_passthrough, openai_to_anthropic_stream

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok", "model": UPSTREAM_MODEL}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": UPSTREAM_MODEL, "object": "model", "owned_by": "upstream"}]
    }


@app.post("/v1/messages")
async def proxy(request: Request):
    body = await request.json()
    model = body.get("model", "default")

    upstream_name, upstream_model, protocol = resolve_model(model)
    body["model"] = upstream_model

    print(f"[REQ] model={model} → {upstream_model} stream={body.get('stream')}")

    headers = get_upstream_headers(upstream_name)
    url = get_upstream_url(upstream_name)
    is_stream = body.get("stream", False)

    try:
        openai_req = anthropic_to_openai(body)
        if is_stream:
            return StreamingResponse(
                openai_to_anthropic_stream(openai_req, model, headers, url),
                media_type="text/event-stream",
            )
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(url, json=openai_req, headers=headers)
            anthropic_resp = openai_response_to_anthropic(resp.json(), model)
            return JSONResponse(anthropic_resp)
    except Exception as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            status_code=500,
        )


if __name__ == "__main__":
    print(f"API Bridge (public) running on http://localhost:{PORT}")
    print(f"Upstream: {UPSTREAM_URL} ({UPSTREAM_MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
