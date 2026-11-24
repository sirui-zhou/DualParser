import re

PATTERNS = {
    'ipv4': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'ipv6': re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
    'uuid': re.compile(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    ),
    'url': re.compile(r'https?://[^\s]+'),
    'unix_path': re.compile(r'(?:/[\w.\-]+)+/?'),
    'win_path': re.compile(r'[A-Za-z]:\\(?:[\w.\-]+\\?)+'),
    'hex': re.compile(r'\b0x[0-9a-fA-F]+\b'),
    'mac': re.compile(r'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b'),
    'email': re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'),
    'timestamp_iso': re.compile(
        r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'
        r'(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b'
    ),
    'timestamp_common': re.compile(
        r'\b\d{2}[/\-.]\d{2}[/\-.]\d{4}\s+\d{2}:\d{2}:\d{2}'
        r'(?:[.,]\d+)?\b'
    ),
    'timestamp_short': re.compile(r'\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b'),
    'number': re.compile(r'\b\d+\b'),
    'float_num': re.compile(r'\b\d+\.\d+\b'),
}

LEVEL_PATTERN = re.compile(
    r'\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|CRITICAL|FATAL|ALERT|EMERGENCY)\b',
    re.IGNORECASE,
)
REDUNDANT_SPACES = re.compile(r'\s+')


def extract_structured_fields(log):
    fields = {}
    cleaned = log
    level_match = LEVEL_PATTERN.search(cleaned)
    if level_match:
        fields['level'] = level_match.group(1).upper()
    for key in ('timestamp_iso', 'timestamp_common', 'timestamp_short'):
        cleaned = PATTERNS[key].sub('<*>', cleaned)
    return cleaned, fields


def replace_special_values(log):
    """Replace special values with <*> to match the Drain pickle format."""
    cleaned = log
    value_types = (
        'url', 'uuid', 'ipv6', 'ipv4', 'mac', 'email', 'win_path',
        'unix_path', 'hex', 'float_num', 'number',
    )
    for key in value_types:
        cleaned = PATTERNS[key].sub('<*>', cleaned)
    return cleaned


def normalize_log(log):
    cleaned = log.strip()
    cleaned, fields = extract_structured_fields(cleaned)
    cleaned = replace_special_values(cleaned)
    cleaned = REDUNDANT_SPACES.sub(' ', cleaned)
    cleaned = re.sub(r'(<\*>\s*){2,}', '<*> ', cleaned).strip()
    return cleaned, fields


def preprocess_for_parsing(log):
    cleaned, _ = normalize_log(log)
    return cleaned
