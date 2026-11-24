import hashlib
import json
import os
from pathlib import Path


_ENV_NAMES = {
    "t5_model_dir": ("DUALPARSER_T5_MODEL_DIR", "LILAC_T5_MODEL_DIR", "T5_MODEL_DIR"),
    "key_file": ("DUALPARSER_KEY_FILE", "LILAC_KEY_FILE", "OPENAI_KEY_FILE"),
    "api_key": ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    "api_base": ("OPENAI_API_BASE", "DEEPSEEK_API_BASE"),
    "cache_dir": ("DUALPARSER_CACHE_DIR", "LILAC_CACHE_DIR", "DUALPARSER_RUNTIME_DIR"),
    "result_dir": ("DUALPARSER_RESULT_DIR", "LILAC_RESULT_DIR"),
    "log_dir": ("DUALPARSER_LOG_DIR", "LILAC_LOG_DIR"),
}


def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _expand_path(value):
    return os.path.abspath(os.path.expanduser(value)) if value else None


def get_repo_root():
    return Path(__file__).resolve().parents[1]


def get_t5_model_dir():
    env_value = _first_env(*_ENV_NAMES["t5_model_dir"])
    if env_value:
        return _expand_path(env_value)

    candidates = [
        r"C:\t5_best",
        os.path.join(os.path.expanduser("~"), "t5_best"),
        os.path.join(os.getcwd(), "t5_best"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def get_key_file():
    env_value = _first_env(*_ENV_NAMES["key_file"])
    if env_value:
        return _expand_path(env_value)

    default_path = get_repo_root() / "openai_key.txt"
    return str(default_path)


def get_cache_dir():
    env_value = _first_env(*_ENV_NAMES["cache_dir"])
    if env_value:
        return _expand_path(env_value)
    return str((get_repo_root() / "temp" / "dualparser").resolve())


def get_result_dir():
    env_value = _first_env(*_ENV_NAMES["result_dir"])
    if env_value:
        return _expand_path(env_value)
    return str((get_repo_root() / "result").resolve())


def get_log_dir():
    env_value = _first_env(*_ENV_NAMES["log_dir"])
    if env_value:
        return _expand_path(env_value)
    return str((get_repo_root() / "logs" / "dualparser").resolve())


def sha256_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return 'missing'


def package_source_sha():
    here = Path(__file__).resolve().parent
    items = [[p.name, sha256_file(p)] for p in sorted(here.glob('*.py'))]
    return hashlib.sha256(
        json.dumps(items, ensure_ascii=False).encode('utf-8')).hexdigest()


def config_fingerprint(parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def get_prompt_dir():
    env_value = _first_env("DUALPARSER_PROMPT_DIR", "LILAC_PROMPT_DIR")
    if env_value:
        return _expand_path(env_value)
    return str((get_repo_root() / "full_dataset" / "sampled_examples").resolve())


def get_prompt_file(dataset_name, shot):
    if not shot:
        return None
    return os.path.join(get_prompt_dir(), dataset_name, f"{shot}shot.json")


def get_api_key():
    return _first_env(*_ENV_NAMES["api_key"]) or os.environ.get("OPENAI_KEY")


def get_api_base():
    return _first_env(*_ENV_NAMES["api_base"]) or os.environ.get("OPENAI_BASE_URL")
