"""配置模块（公开版）：单上游 + config.yaml"""

import os
import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_yaml() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()
_upstream = _cfg.get("upstream", {})
_bridge = _cfg.get("bridge", {})

# 功能开关
ENABLE_THINKING = False
REQUEST_TIMEOUT = 300

# 单上游配置
UPSTREAM_URL = os.getenv("UPSTREAM_URL", _upstream.get("url", ""))
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", _upstream.get("key", ""))
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", _upstream.get("model", "DeepSeek-V4-Flash"))

PORT = int(os.getenv("PORT", str(_bridge.get("port", 4000))))

# 公开版只支持单模型，直接透传
MODEL_MAP = {
    "default": ("upstream", UPSTREAM_MODEL),
}
DEFAULT_MODEL = "default"


def resolve_model(model_name: str) -> tuple[str | None, str | None, str | None]:
    """公开版：所有请求都转发到同一个上游"""
    return "upstream", UPSTREAM_MODEL, "openai"


def get_upstream_headers(upstream_name: str) -> dict:
    return {
        "Authorization": f"Bearer {UPSTREAM_KEY}",
        "Content-Type": "application/json",
    }


def get_upstream_url(upstream_name: str) -> str:
    return UPSTREAM_URL
