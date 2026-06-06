"""
llm_client.py
================
LLM 客户端配置文件。

适用于：OpenAI Compatible格式。
"""

import os
import time

# ============================================================
# 本地测试时，通过环境变量接入 OpenAI-compatible API。
# 可参考 .env.example，但本文件不会自动读取 .env。
# ============================================================
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "qwen3-8b"
ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "false").lower() in {"1", "true", "yes"}

# ============================================================
# 以下代码无需修改
# ============================================================

OPENAI_CONFIG = {
    "base_url": BASE_URL,
    "api_key":  API_KEY,
    "model":    MODEL,
    "temperature": 1.0,
    "top_p":       1.0,
    "max_tokens":  8192,
    "enable_thinking": ENABLE_THINKING,
}

_client = None


def _init_client():
    global _client
    if not OPENAI_CONFIG["api_key"]:
        raise RuntimeError(
            "Missing API key. Set LLM_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY before running local evaluation."
        )
    from openai import OpenAI
    _client = OpenAI(
        base_url=OPENAI_CONFIG["base_url"],
        api_key=OPENAI_CONFIG["api_key"],
    )


def call_llm(messages: list[dict], retries: int = 2) -> str:
    """调用 LLM，输入 OpenAI 格式 messages，返回回复文本。"""
    if _client is None:
        _init_client()

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=messages,
                temperature=OPENAI_CONFIG["temperature"],
                top_p=OPENAI_CONFIG["top_p"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                extra_body={"enable_thinking": OPENAI_CONFIG["enable_thinking"]},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise last_err


_tokenizer = None
_tokenizer_loaded = False
_tokenizer_lock = __import__("threading").Lock()


def _load_tokenizer():
    from transformers import AutoTokenizer
    import os
    _dir = os.path.join(os.path.dirname(__file__), "tokenizer")
    return AutoTokenizer.from_pretrained(_dir, trust_remote_code=True)


def count_tokens(text: str) -> int:
    """计算单段文本的 token 数。"""
    global _tokenizer, _tokenizer_loaded
    if not _tokenizer_loaded:
        with _tokenizer_lock:
            if not _tokenizer_loaded:
                _tokenizer = _load_tokenizer()
                _tokenizer_loaded = True
    if not text:
        return 0
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


def count_messages_tokens(messages: list[dict]) -> int:
    """计算 messages 列表的总 token 数（仅计算 content，与本地 runner 一致）。
    可在调用 call_llm 前用于检查是否超出 max_prompt_tokens。
    """
    return count_tokens(" ".join(m.get("content", "") for m in messages))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """将文本截断到至多 max_tokens 个 token（使用真实 tokenizer）。"""
    global _tokenizer, _tokenizer_loaded
    if not _tokenizer_loaded:
        _tokenizer = _load_tokenizer()
        _tokenizer_loaded = True
    if not text:
        return text
    ids = _tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return _tokenizer.decode(ids[:max_tokens])


if __name__ == "__main__":
    print("Testing LLM connection...")
    try:
        result = call_llm([{"role": "user", "content": "Say 'hello' in one word."}])
        print(f"✓ Connected. Response: {result[:100]}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("请检查 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 等环境变量")
