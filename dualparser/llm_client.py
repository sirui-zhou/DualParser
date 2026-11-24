import os
import time

import openai

from .config import get_api_base, get_api_key, get_key_file


LLM_CALL_STATS = {
    'attempts': 0,
    'ok': 0,
    'truncated': 0,
    'retry': 0,
    'gave_up': 0,
    'fallback_raw': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0,
    'usage_missing_requests': 0,
    'elapsed_seconds': 0.0,
}

_API_READY = False


def reset_llm_call_stats():
    for key in LLM_CALL_STATS:
        LLM_CALL_STATS[key] = 0.0 if key == 'elapsed_seconds' else 0


def _read_key_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        api_base = file.readline().strip()
        key_str = file.readline().strip()
    return api_base, key_str


def ensure_api_credentials():
    global _API_READY
    if _API_READY:
        return

    env_api_key = get_api_key()
    env_api_base = get_api_base()
    key_file = get_key_file()

    if env_api_key:
        openai.api_key = env_api_key
        if env_api_base:
            openai.api_base = env_api_base
        _API_READY = True
        return

    if os.path.exists(key_file):
        api_base, api_key = _read_key_file(key_file)
        openai.api_base = api_base
        openai.api_key = api_key
        _API_READY = True
        return

    _API_READY = True


def request_chat_completion(messages, model, temperature=0.0, max_tokens=2048):
    """Send a chat completion request and normalize API-specific failures."""
    ensure_api_credentials()
    LLM_CALL_STATS['attempts'] += 1
    started_at = time.perf_counter()

    _extra = ({'thinking': {'type': 'disabled'}}
              if os.environ.get('DUALPARSER_THINKING', 'off') == 'off' else {})
    try:
        answer = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            max_tokens=max_tokens,
            **_extra
        )
    except Exception:
        LLM_CALL_STATS['usage_missing_requests'] += 1
        raise
    finally:
        LLM_CALL_STATS['elapsed_seconds'] += time.perf_counter() - started_at

    usage = answer.get('usage') or {}
    if not usage:
        LLM_CALL_STATS['usage_missing_requests'] += 1
    prompt_tokens = usage.get('prompt_tokens', 0) or 0
    completion_tokens = usage.get('completion_tokens', 0) or 0
    total_tokens = usage.get('total_tokens')
    LLM_CALL_STATS['prompt_tokens'] += int(prompt_tokens)
    LLM_CALL_STATS['completion_tokens'] += int(completion_tokens)
    LLM_CALL_STATS['total_tokens'] += int(
        total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
    )

    choices = answer.get('choices', [])
    ok_choices = []
    for response in choices:
        finish_reason = response.get('finish_reason')
        content = response.get('message', {}).get('content')
        if finish_reason != 'length' and content is not None:
            ok_choices.append(content)

    if not ok_choices:
        LLM_CALL_STATS['truncated'] += 1
        raise RuntimeError("response truncated by max_tokens (finish_reason=='length')")

    LLM_CALL_STATS['ok'] += 1
    return ok_choices[0]
