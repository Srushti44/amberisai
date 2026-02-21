"""
deepseek_client.py
==================
Handles AI API calls with streaming.
Auto-detects provider from API key:
  - sk-...   -> DeepSeek
  - gsk_...  -> Groq (FREE, no credit card needed)
"""

import logging
from typing import Generator

logger = logging.getLogger(__name__)


def detect_provider(api_key: str) -> tuple:
    if api_key.startswith("gsk_"):
        return "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"
    else:
        return "https://api.deepseek.com/v1", "deepseek-chat"


def validate_api_key(api_key: str) -> bool:
    if not api_key or len(api_key) < 15:
        return False
    return api_key.startswith("sk-") or api_key.startswith("gsk_")


def stream_deepseek(
    messages: list,
    api_key: str,
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Stream response - auto-detects DeepSeek or Groq from API key format.
    Both are OpenAI-compatible so same SDK works for both.
    """
    try:
        from openai import OpenAI

        base_url, default_model = detect_provider(api_key)
        chosen_model = model or default_model

        client = OpenAI(api_key=api_key, base_url=base_url)

        stream = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    except Exception as e:
        logger.error(f"[DeepSeek] API error: {e}")
        raise


def call_deepseek(
    messages: list,
    api_key: str,
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Non-streaming call. Returns full response string."""
    try:
        from openai import OpenAI
        base_url, default_model = detect_provider(api_key)
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model or default_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    except Exception as e:
        logger.error(f"[DeepSeek] API error: {e}")
        raise