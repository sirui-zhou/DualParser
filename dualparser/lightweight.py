import contextlib
import importlib

from .template_norm import canon_templates
import io as _io
import os
import re
import shutil
import sys

try:
    import regex as _regex
except ImportError:
    import re as _regex

try:
    from .parsing_cache import message_split, lcs_similarity
except ImportError:
    from parsing_cache import message_split, lcs_similarity

_HERE = os.path.dirname(os.path.abspath(__file__))
_LP_MAIN = os.path.join(_HERE, 'vendor')
for _p in (os.path.join(_LP_MAIN, 'logparser'), _LP_MAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PARSER_SPECS = (
    ('A', 'Spell.Spell', 'Spell'),
    ('B', 'IPLoM.IPLoM', 'IPLoM'),
    ('C', 'LFA.LFA', 'LFA'),
)

DEFAULT_THETA = float(os.environ.get('DualParser_Theta', 0.9))

class Fused(object):
    __slots__ = ('template', 'confidence', 'rule')

    def __init__(self, template, confidence, rule):
        self.template = template
        self.confidence = confidence
        self.rule = rule

    def __repr__(self):
        return 'Fused(%r, %.4f, %s)' % (self.template, self.confidence, self.rule)


def _collapse(s):
    out = []
    for tok in s.split(' '):
        if tok == '<*>' and out and out[-1] == '<*>':
            continue
        out.append(tok)
    return ' '.join(out)


def _canon(t):
    if t is None:
        return None
    t = re.sub(r'<[^<>]*>', '<*>', str(t))
    t = _collapse(' '.join(t.split()))
    return t or None


def _rule3(*templates):
    seqs = [message_split(t) for t in templates]
    n = min(len(s) for s in seqs)
    merged = []
    for i in range(n):
        col = set(s[i] for s in seqs)
        merged.append(seqs[0][i] if len(col) == 1 else '<*>')
    extra = max(len(s) for s in seqs) - n
    if extra > 0:
        merged.extend(['<*>'] * extra)
    collapsed = []
    for tok in merged:
        if tok == '<*>' and collapsed and collapsed[-1] == '<*>':
            continue
        collapsed.append(tok)
    return Fused(' '.join(collapsed), 0.6, 'rule3')


def fuse(t_a, t_b, t_c):
    vals = [_canon(t_a), _canon(t_b), _canon(t_c)]
    present = [v for v in vals if v]
    if len(present) < 2:
        return Fused(None, 0.0, 'none')

    if len(present) == 2:
        if present[0] == present[1]:
            return Fused(present[0], 0.8, '2of3')
        return _rule3(present[0], present[1])

    a, b, c = vals
    if a == b == c:
        return Fused(a, 1.0, 'rule1')

    counts = {}
    for t in (a, b, c):
        counts[t] = counts.get(t, 0) + 1
    top = max(counts, key=lambda k: counts[k])
    if counts[top] == 2:
        odd = [t for t in (a, b, c) if t != top][0]
        x = lcs_similarity(message_split(top), message_split(odd))
        return Fused(top, 0.8 + 0.2 * x, 'rule2')

    return _rule3(a, b, c)

_PUNC = "!\"#$%&'()+,-/:;=?@.[]^_`{|}~\\"
_SEP_RE = re.compile("([\\s%s])" % re.escape(_PUNC))
_KEEP_PUNC = '=|()'


def _span_tokens(text):
    out = []
    pos = 0
    for part in _SEP_RE.split(text):
        n = len(part)
        if n == 0:
            continue
        if n == 1 and (part.isspace() or part in _PUNC):
            if part in _KEEP_PUNC:
                out.append((part, pos, pos + 1))
        else:
            out.append(('<*>' if '<*>' in part else part, pos, pos + n))
        pos += n
    return [(t, s, e) for i, (t, s, e) in enumerate(out)
            if not (t == '<*>' and i > 0 and out[i - 1][0] == '<*>')]

_MONTHS = frozenset(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
_WEEKDAYS = frozenset(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
_TZ = frozenset(['am', 'pm', 'utc', 'gmt'])

def _datetime_like(tok):
    if not tok:
        return False
    low = tok.lower()
    if low in _MONTHS or low in _WEEKDAYS or low in _TZ:
        return True
    return tok.isdigit() and len(tok) <= 4


def merge_datetime_runs(tpl):
    out, i, n = [], 0, len(tpl)
    while i < n:
        if tpl[i] != '<*>':
            out.append(tpl[i])
            i += 1
            continue
        j = i
        while j < n and tpl[j] == '<*>':
            j += 1
        if j < n and tpl[j].lower() in (_MONTHS | _WEEKDAYS):
            while j < n and (tpl[j] == '<*>' or _datetime_like(tpl[j])):
                j += 1
            out.append('<*>')
            i = j
            continue
        out.append('<*>')
        i = j
    return out


def render_from_source(source, fused):
    if not source or not fused:
        return None
    src = _span_tokens(source)
    tpl = merge_datetime_runs(message_split(fused))
    if not src or not tpl:
        return None

    pieces, last, i = [], 0, 0
    for j, tok in enumerate(tpl):
        if tok == '<*>':
            nxt = tpl[j + 1] if j + 1 < len(tpl) else None
            if nxt is None:
                k = len(src)
            else:
                k = i
                while k < len(src) and src[k][0] != nxt:
                    k += 1
                if k >= len(src):
                    return None
            if k == i:
                return None
            m, masked = i, False
            while m < k:
                if (src[m][0] == '(' and m + 1 < k and src[m + 1][0] == ')'
                        and not source[src[m][2]:src[m + 1][1]].strip()):
                    pieces.append(source[last:src[m][1]])
                    pieces.append('(<*>)')
                    last = src[m + 1][2]
                    m += 2
                    masked = True
                    continue
                if src[m][0] in _KEEP_PUNC:
                    pieces.append(source[last:src[m][2]])
                    last = src[m][2]
                    m += 1
                    continue
                seg = m
                while m < k and src[m][0] not in _KEEP_PUNC:
                    m += 1
                head = source[last:src[seg][1]]
                cut = max(head.rfind(' '), head.rfind('\t'))
                pieces.append(head[:cut + 1] if cut >= 0 else head)
                pieces.append('<*>')
                last = src[m - 1][2]
                masked = True
            if not masked:
                return None
            i = k
        else:
            while i < len(src) and src[i][0] in _KEEP_PUNC and src[i][0] != tok:
                pieces.append(source[last:src[i][2]])
                last = src[i][2]
                i += 1
            if i >= len(src) or src[i][0] != tok:
                return None
            pieces.append(source[last:src[i][1]])
            pieces.append(source[src[i][1]:src[i][2]])
            last = src[i][2]
            i += 1
    pieces.append(source[last:])
    out = _collapse(' '.join(''.join(pieces).split()))
    return out or None

class LightweightParsers(object):


    def __init__(self, workdir, log_format='<Content>', specs=PARSER_SPECS,
                 keep_workdir=False):
        self.workdir = workdir
        self.log_format = log_format
        self.specs = specs
        self.keep_workdir = keep_workdir
        self.batch_no = 0
        self.stats = {'batches': 0, 'parse_errors': {}, 'rule_counts': {}}


    def _run_one(self, label, modpath, contents, batch_dir):
        indir = os.path.join(batch_dir, label, 'in')
        outdir = os.path.join(batch_dir, label, 'out')
        os.makedirs(indir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        logname = 'lw_batch.log'
        with open(os.path.join(indir, logname), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(contents) + '\n')

        mod = importlib.import_module(modpath)
        parser = mod.LogParser(indir=indir, outdir=outdir, log_format=self.log_format)
        with contextlib.redirect_stdout(_io.StringIO()), \
             contextlib.redirect_stderr(_io.StringIO()):
            parser.parse(logname)

        df = getattr(parser, 'df_log', None)
        if df is None or 'EventTemplate' not in list(df.columns):
            raise RuntimeError('%s produced no df_log["EventTemplate"]' % modpath)

        got = {}
        for c, t in zip(df['Content'].astype(str), df['EventTemplate'].astype(str)):
            got[c] = t
        if not got or any(c not in got for c in contents):
            tpls = [str(x) for x in df['EventTemplate'].tolist()]
            if len(tpls) == len(contents):
                got = {c: t for c, t in zip(contents, tpls)}
            else:
                raise RuntimeError(
                    '%s row count mismatch: batch has %d rows, parser returned %d'
                    % (modpath, len(contents), len(tpls)))
        return got


    def parse_batch(self, contents):
        contents = [c for c in contents if c]
        if not contents:
            return {}

        self.batch_no += 1
        self.stats['batches'] += 1
        batch_dir = os.path.join(self.workdir, 'batch_%05d' % self.batch_no)
        if os.path.isdir(batch_dir):
            shutil.rmtree(batch_dir, ignore_errors=True)
        os.makedirs(batch_dir, exist_ok=True)

        per_parser = []
        for label, modpath, disp in self.specs:
            try:
                per_parser.append(self._run_one(label, modpath, contents, batch_dir))
            except Exception as e:
                self.stats['parse_errors'][disp] = \
                    self.stats['parse_errors'].get(disp, 0) + 1
                print('[lightweight] %s failed on this batch: %s: %s'
                      % (disp, type(e).__name__, str(e)[:160]))
                per_parser.append({})

        if not self.keep_workdir:
            shutil.rmtree(batch_dir, ignore_errors=True)

        out = {}
        for c in contents:
            out[c] = tuple(m.get(c) for m in per_parser)
        return out


    def collaborative(self, contents):
        raw = self.parse_batch(contents)
        out = {}
        for c, triple in raw.items():
            out[c] = fuse(*triple)
        for c in contents:
            f = out.get(c)
            if f is not None:
                self.stats['rule_counts'][f.rule] = \
                    self.stats['rule_counts'].get(f.rule, 0) + 1
        return out

class LogBucket(object):


    def __init__(self, threshold=200, rounds=3):
        self.threshold = threshold
        self.rounds = rounds
        self._items = []
        self._pos = {}
        self.stats = {'added': 0, 'resolved': 0, 'by_cache': 0,
                      'retained': 0, 'gave_up': 0, 'flushes': 0,
                      'rendered': 0, 'render_fallback': 0}


    def add(self, idx, content):
        if idx in self._pos:
            return
        self._pos[idx] = len(self._items)
        self._items.append([idx, content, 0])
        self.stats['added'] += 1


    def put_back(self, idx, content, rnd):
        self._pos[idx] = len(self._items)
        self._items.append([idx, content, rnd + 1])
        self.stats['retained'] += 1


    def __len__(self):
        return len(self._items)


    def ready(self):
        return len(self._items) >= self.threshold


    def take(self):
        batch = [tuple(it) for it in self._items]
        self._items = []
        self._pos = {}
        return batch


def run_flush(bucket, lw, cache, on_resolved, on_give_up, theta=DEFAULT_THETA):
    if not len(bucket):
        return
    bucket.stats['flushes'] += 1
    batch = bucket.take()

    pending = []
    for idx, content, rnd in batch:
        res = cache.match_event(content)
        if res[0] != 'NoMatch':
            bucket.stats['by_cache'] += 1
            bucket.stats['resolved'] += 1
            on_resolved(idx, content, res[0], res[1])
        else:
            pending.append((idx, content, rnd))

    if not pending:
        return

    fused = lw.collaborative([c for _, c, _ in pending])


    if os.environ.get('DualParser_QWEN_FOURTH', '0') == '1':
        try:
            from .DualParser import (query_template_from_qwen,
                                     verify_template_by_log)
        except ImportError:
            from DualParser import (query_template_from_qwen,
                                    verify_template_by_log)
        for idx, content, rnd in pending:
            f = fused.get(content)

            if f is not None and f.template and f.rule == 'rule1':
                continue
            try:
                q = query_template_from_qwen(content, f.template if f else None)
            except Exception:
                continue
            ok, _why = verify_template_by_log(q, content)
            if ok:
                fused[content] = Fused(q, 1.0, 'qwen4')
                bucket.stats['qwen_fourth'] = bucket.stats.get('qwen_fourth', 0) + 1

    written = {}
    tpl_of = {}


    _dump_path = os.environ.get('DualParser_DUMP_FUSION', '')
    if _dump_path:
        import csv as _csv
        _new = not os.path.exists(_dump_path)
        with open(_dump_path, 'a', newline='', encoding='utf-8') as _f:
            _w = _csv.writer(_f)
            if _new:
                _w.writerow(['content', 'fused_template', 'confidence', 'rule',
                             'rendered', 'round'])
            for idx, content, rnd in pending:
                f = fused.get(content)
                if f is None:
                    _w.writerow([content, '', '', 'none', '', rnd])
                    continue
                _r = render_from_source(content, f.template) if f.template else None
                _w.writerow([content, f.template or '', f.confidence, f.rule,
                             _r if _r is not None else (f.template or ''), rnd])
    for idx, content, rnd in pending:
        f = fused.get(content)
        if f is not None and f.template and f.confidence >= theta:
            tpl = render_from_source(content, f.template)
            if tpl is None:
                bucket.stats['render_fallback'] += 1
                tpl = f.template
            else:
                bucket.stats['rendered'] += 1

            tid = cache.add_templates(tpl, True, [])
            if tid is not None and tid != -1:
                written[idx] = tid
                tpl_of[idx] = tpl

    for idx, content, rnd in pending:
        if idx in written:
            bucket.stats['resolved'] += 1
            on_resolved(idx, content, tpl_of[idx], written[idx])
            continue

        res = cache.match_event(content)
        if res[0] != 'NoMatch':
            bucket.stats['by_cache'] += 1
            bucket.stats['resolved'] += 1
            on_resolved(idx, content, res[0], res[1])
            continue

        if rnd + 1 >= bucket.rounds:
            bucket.stats['gave_up'] += 1
            on_give_up(idx, content)
        else:
            bucket.put_back(idx, content, rnd)

_RETOK_DELIMS = set(' \t\n\r\f\v,!;:=|"\'[](){}.-+@#$%&')
_RETOK_NONWS = ''.join(re.escape(c)
                       for c in sorted(_RETOK_DELIMS - set(' \t\n\r\f\v')))
_RETOK_RUN_RE = re.compile(r'<\*>([' + _RETOK_NONWS + r']+)<\*>')


def _retok_tokenize(s):
    out, buf = [], []
    for ch in s:
        if ch in _RETOK_DELIMS:
            if buf:
                out.append(('w', ''.join(buf)))
                buf = []
            out.append(('d', ch))
        else:
            buf.append(ch)
    if buf:
        out.append(('w', ''.join(buf)))
    return out


def _retok_collapse(s):
    while True:
        new = _RETOK_RUN_RE.sub('<*>', s)
        if new == s:
            return s
        s = new


def retemplate_cluster(members_masked, collapse=True):
    if len(members_masked) < 2:
        return None
    rep = _retok_tokenize(members_masked[0])
    n_pos = len(rep)
    if n_pos == 0:
        return None

    col_vals = [set() for _ in range(n_pos)]
    n_used = 0
    for m in members_masked:
        t = _retok_tokenize(m)
        if len(t) != n_pos:
            continue
        if any(t[i][0] != rep[i][0] for i in range(n_pos)):
            continue
        n_used += 1
        for i in range(n_pos):
            col_vals[i].add(t[i][1])
    if n_used < 2:
        return None

    out = ''.join(ch if kind == 'd' or len(col_vals[i]) == 1 else '<*>'
                  for i, (kind, ch) in enumerate(rep))
    return _retok_collapse(out) if collapse else out


def _protect_wildcards(old_tpl, new_tpl):
    old_toks = _retok_tokenize(old_tpl)
    new_toks = _retok_tokenize(new_tpl)
    if len(old_toks) != len(new_toks):
        return new_tpl
    changed = False
    for i in range(len(old_toks)):
        ok, ov = old_toks[i]
        nk, nv = new_toks[i]
        if ok == 'w' and ov == '<*>' and nk == 'w' and nv != '<*>':
            new_toks[i] = ('w', '<*>')
            changed = True
    if not changed:
        return new_tpl
    return _retok_collapse(''.join(t[1] for t in new_toks))


def apply_mask(s, rex):
    if not rex:
        return s
    for r in rex:
        s = _regex.sub(r, '<*>', s)
    return s


def _retok_const_ratio(tpl):
    toks = [t for t in _retok_tokenize(tpl) if t[0] == 'w']
    if not toks:
        return 0.0
    return sum(1 for t in toks if t[1] != '<*>') / len(toks)


def _retok_adaptive_keep(old_tpl, new_tpl, cluster_size):
    if cluster_size < 3:
        return False
    old_ratio = _retok_const_ratio(old_tpl)
    new_ratio = _retok_const_ratio(new_tpl)
    if old_ratio >= 0.20 and new_ratio < old_ratio * 0.70:
        return False
    return True


def retokenize_templates(contents, templates, rex):
    if not rex or len(contents) != len(templates):
        return list(templates)


    def _mask(s):
        return apply_mask(s, rex)

    buckets = {}
    for i, t in enumerate(templates):
        buckets.setdefault(t, []).append(i)

    out = list(templates)
    n_kept = 0
    n_applied = 0
    for _t, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        members = [_mask(contents[i]) for i in idxs]
        v = retemplate_cluster(members)
        if v is None:
            continue
        if not _retok_adaptive_keep(_t, v, len(idxs)):
            n_kept += 1
            continue
        n_applied += 1
        for i in idxs:
            out[i] = v
    return out


def finalize_templates(contents, templates, rex, canon=False, stats=None):
    out = retokenize_templates(contents, templates, rex)
    if stats is not None:
        stats['retok_lines_changed'] = sum(1 for a, b in zip(templates, out) if a != b)
        stats['clusters_before'] = len(set(templates))
        stats['clusters_after_retok'] = len(set(out))
    if canon:
        post = canon_templates(out)
        if stats is not None:
            stats['canon_lines_changed'] = sum(1 for a, b in zip(out, post) if a != b)
            stats['clusters_after_canon'] = len(set(post))
        out = post
    return out
