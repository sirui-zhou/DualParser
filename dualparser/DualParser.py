import regex as re
import os
import pickle
import pandas as pd
from datetime import datetime
import json
import time
from .config import (get_t5_model_dir, get_cache_dir, get_result_dir,
                     get_prompt_file, config_fingerprint, package_source_sha,
                     sha256_file, get_repo_root)
from .gpt_query import query_template_from_gpt_with_check
from .llm_client import LLM_CALL_STATS, reset_llm_call_stats
from .benchmark_metrics import build_cost_report
from .parsing_cache import ParsingCache
from .lightweight import (LogBucket, LightweightParsers, run_flush,
                          finalize_templates, DEFAULT_THETA)
from .prompt_select import prompt_select
from .preprocess import normalize_log as _compress_log
from .utils import (
    cache_to_file,
    load_pickle,
    load_tuple_list,
    read_json_file,
    save_pickle,
)
try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError:
    torch = None
    T5Tokenizer = None
    T5ForConditionalGeneration = None

_t5_model = None
_t5_tokenizer = None
_t5_device = None
T5_INFERENCE_STATS = {
    "input_tokens": 0,
    "output_tokens": 0,
    "elapsed_seconds": 0.0,
}

_DBG_PROGRESS = os.environ.get('DualParser_DEBUG_PROGRESS', '0') == '1'
_SKIP_VOTE = os.environ.get('DualParser_SKIP_VOTE', '0') == '1'


_FLAN_MODE = os.environ.get('DualParser_USE_FLAN', '0') == '1'


FLAN_INSTRUCTION = (
    "Log parsing task. Given a raw log message, output its template.\n"
    "Template definition: the log message with all dynamic values replaced by <*>.\n"
    "Dynamic values include: timestamps, IPs, PIDs, usernames, file paths, hex addresses, numbers.\n"
    "Keep all static words and punctuation exactly as they appear.\n"
    "Output format: only the template string, nothing else.\n"
)


_QWEN_MODE = os.environ.get('DualParser_USE_QWEN', '0') == '1'
QWEN_BASE_URL = os.environ.get('DualParser_QWEN_BASE', 'http://127.0.0.1:8000/v1')
QWEN_MODEL_NAME = os.environ.get('DualParser_QWEN_MODEL', 'qwen3-8B')
QWEN_INSTRUCTION = (
    "You are a log parsing expert. Convert the log message into a template: "
    "replace dynamic values (timestamps, numbers, IPs, ports, ids, paths, user "
    "names, host names, hex addresses) with <*>, and keep every other word and "
    "punctuation exactly as it appears. "
    "Output the template only, on a single line, with no explanation."
)
_qwen_client = None


def get_qwen_client():
    return None


def query_template_from_qwen(log, candidate_template=None):
    import json
    import urllib.request

    started_at = time.perf_counter()
    if candidate_template:
        user_text = (
            "Log: %s\nCandidate template: %s\n"
            "If the candidate is already correct, reply with the candidate "
            "template unchanged. Otherwise reply with the corrected template "
            "only, on a single line." % (log, candidate_template)
        )
    else:
        user_text = "Log: %s" % log
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": QWEN_INSTRUCTION},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 128,

        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        QWEN_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer dummy"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = (data["choices"][0]["message"]["content"] or "").strip()
    text = text.split("\n")[0].strip().strip("`").strip()
    for prefix in ("Template:", "template:", "模板:", "Template :",
                   "Log:", "log:", "Log message:", "Input:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = " ".join(text.split())
    text = text.replace("*>", "<*>")
    T5_INFERENCE_STATS["elapsed_seconds"] += time.perf_counter() - started_at
    T5_INFERENCE_STATS["input_tokens"] += len(log.split())
    T5_INFERENCE_STATS["output_tokens"] += len(text.split())
    return text, True


def get_t5_model():
    global _t5_model, _t5_tokenizer, _t5_device
    if _t5_model is None:
        model_dir = get_t5_model_dir()
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"T5-small model directory not found: '{model_dir}'. "
                "Set DUALPARSER_T5_MODEL_DIR or LILAC_T5_MODEL_DIR to the correct path."
            )
        print(f"Loading T5-small model from {model_dir}...")
        if _FLAN_MODE:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            _t5_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        else:
            _t5_tokenizer = T5Tokenizer.from_pretrained(model_dir)
            _t5_model = T5ForConditionalGeneration.from_pretrained(model_dir)
        _t5_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _t5_model.to(_t5_device)
        _t5_model.eval()
        print(f"T5-small loaded on {_t5_device}")
    return _t5_model, _t5_tokenizer, _t5_device


def query_template_from_t5(log, candidate_template=None):
    if _QWEN_MODE:
        return query_template_from_qwen(log, candidate_template)
    model, tokenizer, device = get_t5_model()
    if _FLAN_MODE:
        input_text = FLAN_INSTRUCTION + "Log: " + log + "\nTemplate:"
    else:
        input_text = "parselog: " + log
    inputs = tokenizer(
        input_text, max_length=512, truncation=True, return_tensors='pt'
    ).to(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    T5_INFERENCE_STATS["elapsed_seconds"] += time.perf_counter() - started_at
    T5_INFERENCE_STATS["input_tokens"] += int(inputs["attention_mask"].sum().item())
    T5_INFERENCE_STATS["output_tokens"] += int(
        outputs[0].ne(tokenizer.pad_token_id).sum().item()
    )
    template = tokenizer.decode(outputs[0], skip_special_tokens=True)
    template = template.replace("*>", "<*>")
    return template, True

T5_ESCALATE_THETA = float(os.environ.get('DualParser_ACCEPT_THETA', '0.995'))

T5_MIN_LITERAL_MASS = 1


def is_t5_template_usable(t5_template):
    literal_mass = sum(1 for c in t5_template.replace('<*>', '') if c.isalnum())
    return literal_mass >= T5_MIN_LITERAL_MASS


def normalize_wildcards(text):
    text = str(text).replace('\\', '')
    text = re.sub(r'<+\s*\*+\s*>', '<*>', text)
    text = text.replace('<>', '<*>')
    return re.sub(r'\s+', ' ', text).strip()



def verify_template_by_log(template, log):
    tpl = normalize_wildcards(template)
    if not tpl:
        return False, 'empty'
    if not is_t5_template_usable(tpl):
        return False, 'no_literal'

    if '<*>' not in tpl or tpl == log.strip():
        return False, 'no_abstraction'
    frags = tpl.split('<*>')
    pattern = '^' + '(.+?)'.join(re.escape(f) for f in frags) + '$'
    try:
        m = re.match(pattern, log)
    except re.error:
        return False, 'bad_regex'
    if m is None:
        return False, 'no_cover'
    params = list(m.groups())
    if any(p == '' for p in params):
        return False, 'empty_param'
    pos = 0
    for frag in frags:
        if not frag:
            continue
        j = log.find(frag, pos)
        if j < 0:
            return False, 'fabricated'
        pos = j + len(frag)
    rebuilt = [frags[0]]
    for p, f in zip(params, frags[1:]):
        rebuilt.append(p)
        rebuilt.append(f)
    if ''.join(rebuilt) != log:
        return False, 'no_reconstruct'
    return True, 'ok'



def should_escalate(t5_template, drain_candidate=None, theta=None):
    if theta is None:
        theta = T5_ESCALATE_THETA
    if not is_t5_template_usable(t5_template):
        return True
    if drain_candidate and drain_candidate != "(none)":
        from difflib import SequenceMatcher
        sim = SequenceMatcher(None, t5_template, drain_candidate).ratio()
        if sim < theta:
            return True
    return False


def is_t5_quality_good(t5_template, drain_candidate=None):
    return not should_escalate(t5_template, drain_candidate)


def save_results_to_csv(log_file, template_file, cache_file, output_file,
                        output_template_file, retok_rex=None, tpl_canon=True,
                        finalize_stats=None):
    with open(log_file, 'r') as f:
        lines_a = f.readlines()
    with open(template_file, 'r') as f:
        lines_b = f.readlines()
    cache = load_pickle(cache_file)
    total_len = len(lines_a)
    lineids = range(1, total_len + 1)
    eventids = [''] * total_len
    contents = [''] * total_len
    event_templates = [''] * total_len
    templates_set = []
    orphan_count = 0
    print("start writing log structured csv.")
    for (line_a, line_b) in zip(lines_a, lines_b):
        idx_a, str_a = line_a.strip().split(' ', 1)
        idx_b, str_b = line_b.strip().split(' ', 1)
        idx_a, idx_b = int(idx_a), int(idx_b)
        tid = int(str_b)
        if 0 <= tid < len(cache.template_list):
            str_b = cache.template_list[tid]
        else:
            orphan_count += 1
            str_b = str_a
        if idx_a != idx_b:
            print(f"Error in line: {idx_a} {idx_b}")
            return
        contents[idx_a] = str_a
        event_templates[idx_a] = str_b

    event_templates = finalize_templates(contents, event_templates, retok_rex,
                                         canon=tpl_canon, stats=finalize_stats)

    tpl_index = {}
    for i in range(total_len):
        t = event_templates[i]
        eid = tpl_index.get(t)
        if eid is None:
            templates_set.append(t)
            eid = len(templates_set)
            tpl_index[t] = eid
        eventids[i] = f"E{eid}"
    print("end writing log structured csv.")
    if orphan_count:
        print(f"WARNING: {orphan_count} line(s) had no valid template id "
              f"(add_templates returned -1); fell back to the raw log.")

    df = pd.DataFrame({
        'LineId': lineids,
        'EventId': eventids,
        'Content': contents,
        'EventTemplate': event_templates,
    })
    df.to_csv(output_file, index=False)

    template_ids = [f"E{i+1}" for i in range(len(templates_set))]
    df = pd.DataFrame({'EventId': template_ids, 'EventTemplate': templates_set})
    df.to_csv(output_template_file, index=False)


def load_regs():
    regs_common = []
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "common.json"), "r") as fr:
        dic = json.load(fr)
    
    patterns = dic['COMMON']['regex']
    for pattern in patterns:
        regs_common.append(re.compile(pattern))
    return regs_common

class LogParser:

    def __init__(
        self,
        log_format,
        indir='./',
        outdir='./result/',
        rex=[],
        data_type='2k',
        shot=0,
        example_size=0,
        model="gpt-3.5-turbo-0613",
        selection_method="LILAC",
        use_t5=False,
        cold_start=False,
        tpl_canon=True,
    ):
        self.path = indir
        self.df_log = None
        self.log_format = log_format
        self.data_type = data_type
        self.shot = shot
        self.example_size = example_size
        self.selection_method = selection_method
        self.model = model
        self.use_t5 = use_t5
        self.cold_start = cold_start
        self.outdir = outdir
        self.rex = rex or []
        self.tpl_canon = bool(tpl_canon)


    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        dataset_name = logName.split('_')[0]
        cache_root = get_cache_dir()
        output_path = os.path.join(
            cache_root,
            f"dualparser_{self.data_type}_{self.shot}_{self.example_size}_{self.model}",
            dataset_name,
        )
        evaluation_path = (
            self.outdir
            if self.outdir and self.outdir != './result/'
            else get_result_dir()
        )
        evaluation_path = os.path.abspath(evaluation_path)
        output_csv = os.path.join(
            evaluation_path,
            f"{dataset_name}_{self.data_type}.log_structured.csv",
        )
        if os.path.exists(output_csv):
            print(f"{dataset_name} already exists.")
            return
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(evaluation_path, exist_ok=True)
        run_started_at = time.perf_counter()
        reset_llm_call_stats()
        finalize_stats = {}
        T5_INFERENCE_STATS.update({
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_seconds": 0.0,
        })
        self._llm_queried_log_count = 0
        self.stage_stats = {}
        self.load_data()


        if self.cold_start:
            self.drain_results = {}
            print(
                "Cold start mode: batch Drain pre-run disabled, "
                "using cache Drain tree only"
            )
        else:
            drain_pkl = os.path.join(
                get_repo_root(), f"drain_{dataset_name}_{self.data_type}.pkl"
            )
            if os.path.exists(drain_pkl):
                with open(drain_pkl, 'rb') as f:
                    self.drain_results = pickle.load(f)
                print(f"Loaded batch Drain results: {len(self.drain_results)} lines")
            else:
                self.drain_results = {}
                print(f"No Drain results at {drain_pkl}, pure LLM mode")

        self.adaptive_threshold = 0.4

        self.bucket = LogBucket(
            threshold=int(os.environ.get('DualParser_BUCKET_THRESHOLD', 200)),
            rounds=int(os.environ.get('DualParser_BUCKET_ROUNDS', 3)))
        self.lw = LightweightParsers(
            workdir=os.path.join(get_cache_dir(), '_lw_bucket', dataset_name))
        self.theta = DEFAULT_THETA
        self._pending_drain = {}

        regs_common = load_regs()


        cached_tree = os.path.join(output_path, "cached_tree.pkl")
        cached_log = os.path.join(output_path, "cached_log.txt")
        cached_template = os.path.join(output_path, "cached_template.txt")
        _fp_drain = os.path.join(
            get_repo_root(), f"drain_{dataset_name}_{self.data_type}.pkl"
        )
        _fp_prompt = get_prompt_file(dataset_name, self.shot)
        fp_parts = {
            'schema': 1,
            'package_src': package_source_sha(),
            'dataset': dataset_name,
            'data_type': self.data_type,
            'shot': self.shot,
            'example_size': self.example_size,
            'model': self.model,
            'selection_method': self.selection_method,
            'use_t5': bool(self.use_t5),
            'cold_start': bool(self.cold_start),
            'tpl_canon': bool(self.tpl_canon),
            'retok_rex': list(self.rex or []),
            'theta': DEFAULT_THETA,
            't5_escalate_theta': T5_ESCALATE_THETA,
            't5_min_literal_mass': T5_MIN_LITERAL_MASS,
            'bucket_threshold': self.bucket.threshold,
            'bucket_rounds': self.bucket.rounds,
            'skip_vote': bool(_SKIP_VOTE),
            'drain_pkl': sha256_file(_fp_drain),
            'prompt_file': (sha256_file(_fp_prompt) if _fp_prompt else 'none'),
        }
        self.config_fingerprint = config_fingerprint(fp_parts)
        fp_file = os.path.join(output_path, '_config_fingerprint.json')
        _had_cache = any(os.path.exists(_p) for _p in
                         (cached_tree, cached_log, cached_template))
        if os.path.exists(fp_file):
            try:
                with open(fp_file, encoding='utf-8') as _f:
                    _stored = json.load(_f)
            except Exception:
                _stored = {}
            if _stored.get('fingerprint') != self.config_fingerprint:
                _old = _stored.get('parts', {}) or {}
                _diff = ', '.join(
                    k for k in sorted(set(fp_parts) | set(_old))
                    if _old.get(k) != fp_parts.get(k)) or '(no per-field record in the old entry)'
                if os.environ.get('DualParser_ALLOW_FP_MISMATCH') == '1':
                    print('[fingerprint] WARNING: config fingerprint mismatch, but '
                          'DualParser_ALLOW_FP_MISMATCH=1 -- resuming anyway '
                          '(results may mix configurations).')
                else:
                    raise RuntimeError(
                        '[fingerprint] The config fingerprint does not match the one '
                        'recorded in the cache directory; refusing to resume.\n'
                        f'  cache dir   : {output_path}\n'
                        f'  recorded    : {_stored.get("fingerprint")}\n'
                        f'  current     : {self.config_fingerprint}\n'
                        f'  differences : {_diff}\n'
                        '  remedies    : (1) use a new --tag; (2) move that dataset'
                        "'s cache directory away; (3) set "
                        'DualParser_ALLOW_FP_MISMATCH=1 if you are sure it is fine.')
        elif _had_cache:
            if os.environ.get('DualParser_ALLOW_FP_MISMATCH') == '1':
                print('[fingerprint] WARNING: the cache directory holds an old cache '
                      'with no fingerprint record, but DualParser_ALLOW_FP_MISMATCH=1 '
                      '-- resuming anyway (results may mix configurations).')
            else:
                raise RuntimeError(
                    '[fingerprint] The cache directory holds an old cache with no '
                    'fingerprint record, so it cannot be shown to match the current '
                    'code / regex / pickle files; refusing to resume.\n'
                    f'  cache dir : {output_path}\n'
                    f'  current   : {self.config_fingerprint}\n'
                    '  remedies  : (1) move that dataset cache directory away '
                    '(recommended);\n'
                    '              (2) use a new --tag (it changes the output dir, '
                    'not the cache dir);\n'
                    '              (3) set DualParser_ALLOW_FP_MISMATCH=1 if you are '
                    'sure it is fine.')
        with open(fp_file, 'w', encoding='utf-8') as _f:
            json.dump({'fingerprint': self.config_fingerprint,
                       'short': self.config_fingerprint[:12],
                       'dataset': dataset_name,
                       'written_at': datetime.now().isoformat(timespec='seconds'),
                       'parts': fp_parts}, _f, ensure_ascii=False, indent=2)
        print(f'[fingerprint] {dataset_name}: {self.config_fingerprint[:12]}'
              f'  (written to {fp_file})')

        if (
            os.path.exists(cached_tree)
            and os.path.exists(cached_log)
            and os.path.exists(cached_template)
        ):
            cache = load_pickle(cached_tree)
            log_messages = load_tuple_list(cached_log)
            log_templates = load_tuple_list(cached_template)
            idx = log_messages[-1][1] + 1
        else:
            log_messages = []
            log_templates = []
            cache = ParsingCache()
            idx = 0
        _prompt_file = get_prompt_file(dataset_name, self.shot)
        prompt_cases = None if _prompt_file is None else read_json_file(_prompt_file)


        num_query = 0
        total_line = len(self.df_log)
        resume_start_idx = idx
        run_input_line_count = max(0, total_line - idx)
        cache_step = max(1, total_line // 5)
        if idx < total_line:
            for log in self.df_log[idx:]['Content']:
                if _DBG_PROGRESS and idx % 100 == 0:
                    print(
                        f"[dbg] line={idx} tpl={len(cache.template_list)} "
                        f"topkeys={len(cache.template_tree)} "
                        f"{datetime.now().strftime('%H:%M:%S')}",
                        flush=True,
                    )
                queried = self.process_log(
                    cache, [log], log_messages, log_templates, idx,
                    prompt_cases, regs_common, total_line,
                )
                idx += 1
                if queried:
                    num_query += queried
                    print("Query times: ", num_query)
                if idx % cache_step == 0:
                    print("Finished processing line: ", idx)
                    if len(self.bucket) == 0:
                        cache_to_file(log_messages, cached_log)
                        cache_to_file(log_templates, cached_template)
                        save_pickle(cache, cached_tree)                

        while len(self.bucket) > 0:
            num_query += self.flush_bucket(cache, log_messages, log_templates,
                                           prompt_cases, regs_common, total_line)


        if num_query > 0:
            print("Total query: ", num_query)

        log_messages.sort(key=lambda t: t[1])
        log_templates.sort(key=lambda t: t[1])

        cache_to_file(log_messages, cached_log)
        cache_to_file(log_templates, cached_template)
        save_pickle(cache, cached_tree)
        parsed_path = os.path.join(
            evaluation_path, f"{dataset_name}_{self.data_type}.log_structured.csv"
        )
        save_results_to_csv(
            cached_log, cached_template, cached_tree,
            parsed_path,
            os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_templates.csv"),
            retok_rex=self.rex,
            tpl_canon=self.tpl_canon,
            finalize_stats=finalize_stats,
        )
        elapsed_seconds = time.perf_counter() - run_started_at
        api_stats = dict(LLM_CALL_STATS)
        t5_stats = dict(T5_INFERENCE_STATS)

        metrics = {
            "dataset": dataset_name,
            "model": self.model,
            "total_log_lines": total_line,
            "processed_log_lines_this_run": run_input_line_count,
            "resumed_from_line_index": resume_start_idx,
            "llm_queried_log_lines": self._llm_queried_log_count,
            "llm_access_rate": (
                self._llm_queried_log_count / float(run_input_line_count)
                if run_input_line_count else 0.0
            ),
            "api": api_stats,
            "elapsed_seconds": elapsed_seconds,
            "logs_per_second": (
                run_input_line_count / elapsed_seconds if elapsed_seconds else None
            ),
            "t5": t5_stats,
        }
        if "EventTemplate" in self.df_log.columns:
            parsed = pd.read_csv(parsed_path)
            if len(parsed) == len(self.df_log):
                try:
                    from ..utils.evaluator import get_accuracy
                except ImportError:
                    from utils.evaluator import get_accuracy
                _acc = get_accuracy(
                    self.df_log["EventTemplate"], parsed["EventTemplate"]
                )
                ga, fga = _acc[0], _acc[1]
                metrics["template_matching"] = {"GA": ga, "FGA": fga}
            else:
                metrics["template_matching"] = {
                    "available": False,
                    "reason": "ground-truth and parsed row counts differ",
                    "ground_truth_rows": len(self.df_log),
                    "parsed_rows": len(parsed),
                }
        else:
            metrics["template_matching"] = {
                "available": False,
                "reason": "input has no EventTemplate ground truth column",
            }
        metrics["cost"] = build_cost_report(api_stats, t5_stats)
        metrics["config_fingerprint"] = self.config_fingerprint
        metrics["config_fingerprint_short"] = self.config_fingerprint[:12]
        metrics["stages"] = {
            "note": "counters cover only the segment processed in this run; resumed runs exclude skipped lines",
            "lines": {
                "cache_hit": self.stage_stats.get("cache_hit", 0),
                "drain_tpl_hit_cache": self.stage_stats.get("drain_tpl_hit_cache", 0),
                "drain_adopted": self.stage_stats.get("drain_adopted", 0),
                "drain_uncertain": self.stage_stats.get("drain_uncertain", 0),
                "bucket_added": self.stage_stats.get("bucket_added", 0),
                "llm_path_entered": self.stage_stats.get("llm_path_entered", 0),
                "t5_accepted": self.stage_stats.get("t5_accepted", 0),
                "t5_escalated": self.stage_stats.get("t5_escalated", 0),
                "llm_template_written": self.stage_stats.get("llm_template_written", 0),
            },
            "bucket": dict(self.bucket.stats),
            "lightweight": {
                "batches": self.lw.stats.get("batches", 0),
                "rule_counts": dict(self.lw.stats.get("rule_counts", {})),
                "parse_errors": dict(self.lw.stats.get("parse_errors", {})),
            },
            "finalize": dict(finalize_stats),
        }

        with open(
            os.path.join(evaluation_path, f"_evaluation_{dataset_name}.json"),
            "w",
            encoding="utf-8",
        ) as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        with open(
            os.path.join(evaluation_path, f"_llm_calls_{dataset_name}.txt"),
            "w",
            encoding="utf-8",
        ) as calls_file:
            calls_file.write(
                f"{dataset_name}\t{api_stats['attempts']}\t{total_line}\t"
                f"{self._llm_queried_log_count}\t"
                f"{api_stats['prompt_tokens']}\t{api_stats['completion_tokens']}\t"
                f"{api_stats['total_tokens']}\n"
            )

        print(f"[Bucket stats] {self.bucket.stats}")
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


    def _bump(self, key, n=1):
        self.stage_stats[key] = self.stage_stats.get(key, 0) + n


    def process_log(
        self,
        cache,
        logs,
        log_messages,
        log_templates,
        idx,
        prompt_cases,
        regs_common,
        total_line,
    ):

        for log in logs:
            results = cache.match_event(log)
            if results[0] != "NoMatch":
                self._bump("cache_hit")
                log_messages.append((log, idx))
                log_templates.append((results[1], idx))
                return 0


            candidate_drain_tpl = None
            if idx in self.drain_results and self.drain_results[idx]:
                drain_tpl = self.drain_results[idx]
                matched = cache.match_event(drain_tpl)
                if matched[0] != "NoMatch":
                    self._bump("drain_tpl_hit_cache")
                    log_messages.append((log, idx))
                    log_templates.append((matched[1], idx))
                    return 0
                tokens = drain_tpl.split()
                wildcards = tokens.count("<*>")
                wildcard_ratio = wildcards / max(len(tokens), 1)
                if wildcard_ratio < self.adaptive_threshold:
                    self._bump("drain_adopted")
                    print(f"Drain confident at Line-{idx}: {drain_tpl}")
                    template_id = cache.add_templates(drain_tpl, True, results[2])
                    log_messages.append((log, idx))
                    log_templates.append((template_id, idx))
                    return 0
                self._bump("drain_uncertain")
                print(
                    f"Drain uncertain (high wildcard ratio) at Line-{idx}, "
                    "queued to log bucket"
                )
                candidate_drain_tpl = drain_tpl

            self._bump("bucket_added")
            self.bucket.add(idx, log)
            self._pending_drain[idx] = candidate_drain_tpl
            if self.bucket.ready():
                return self.flush_bucket(cache, log_messages, log_templates,
                                         prompt_cases, regs_common, total_line)
            return 0


    def flush_bucket(
        self, cache, log_messages, log_templates, prompt_cases, regs_common, total_line
    ):
        queried = [0]


        def on_resolved(idx, content, template, template_id):
            log_messages.append((content, idx))
            log_templates.append((template_id, idx))
            self._pending_drain.pop(idx, None)


        def on_give_up(idx, content):
            queried[0] += self.query_llm_for(cache, content, idx, log_messages,
                                             log_templates, prompt_cases,
                                             regs_common, total_line)

        if _SKIP_VOTE:

            def on_resolved(idx, content, template, template_id):
                on_give_up(idx, content)

        run_flush(self.bucket, self.lw, cache, on_resolved, on_give_up, theta=self.theta)
        return queried[0]


    def query_llm_for(
        self,
        cache,
        log,
        idx,
        log_messages,
        log_templates,
        prompt_cases,
        regs_common,
        total_line,
    ):
        candidate_drain_tpl = self._pending_drain.pop(idx, None)
        self._bump("llm_path_entered")
        results = cache.match_event(log)
        print("===========================================")
        print(f"Line-{idx}/{total_line}: No match. {log}")


        if os.environ.get('DualParser_COMPRESS_LOG', '0') == '1':
            log_for_model, _ = _compress_log(log)
            if log_for_model != log:
                print(f"[compress] -> {log_for_model}")
        else:
            log_for_model = log
        if prompt_cases is not None:
            examples = prompt_select(prompt_cases, log_for_model, self.example_size, self.selection_method)
        else:
            examples = []
        queried_llm = False

        if self.use_t5:
            t5_template, normal = query_template_from_t5(
                log_for_model, candidate_template=candidate_drain_tpl
            )
            print("t5_new_template: ", t5_template)
            if _QWEN_MODE:
                _usable, _why = verify_template_by_log(t5_template, log_for_model)
                if not _usable:
                    print("Qwen template rejected by verifier: ", _why)
            else:
                _usable = is_t5_quality_good(t5_template, candidate_drain_tpl)
            if not _usable:
                self._bump("t5_escalated")
                queried_llm = True
                self._llm_queried_log_count += 1
                print("T5 quality low, falling back to LLM correction...")
                new_template, normal = query_template_from_gpt_with_check(
                    log_for_model, regs_common, examples, self.model,
                    candidate_template=candidate_drain_tpl,
                    t5_template=t5_template)
                print("llm_corrected_template: ", new_template)
            else:
                self._bump("t5_accepted")
                new_template = t5_template
        else:
            queried_llm = True
            self._llm_queried_log_count += 1
            new_template, normal = query_template_from_gpt_with_check(
                log_for_model,
                regs_common,
                examples,
                self.model,
                candidate_template=candidate_drain_tpl,
            )
            print("queried_new_template: ", new_template)
        self._bump("llm_template_written")
        template_id = cache.add_templates(new_template, normal, results[2])
        log_messages.append((log, idx))
        log_templates.append((template_id, idx))
        print("===========================================")
        return int(queried_llm)


    def load_data(self):
        csv_path = os.path.join(self.path, self.logName+'_structured.csv')
        if os.path.exists(csv_path):
            self.df_log = pd.read_csv(csv_path)
        else:
            headers, regex = self.generate_logformat_regex(self.log_format)
            self.df_log = self.log_to_dataframe(
                os.path.join(self.path, self.logName),
                regex,
                headers,
                self.log_format,
            )


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
