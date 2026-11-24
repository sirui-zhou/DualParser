import re
import sys

sys.setrecursionlimit(1000000)


def print_tree(move_tree, indent=' '):
    for key, value in move_tree.items():
        if isinstance(value, dict):
            print(f'{indent}|- {key}')
            print_tree(value, indent + '|  ')
        elif isinstance(value, tuple):
            print(f'{indent}|- {key}: tuple')
        else:
            print(f'{indent}|- {key}: {value}')


def lcs_similarity(X, Y):
    m, n = len(X), len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return 2 * c[m][n] / (m + n)

import re as _re
_ALNUM_RE = _re.compile(r'[A-Za-z0-9]')


def _is_punct_shell(template_tokens):
    consts = [t for t in template_tokens if t != '<*>']
    if template_tokens.count('<*>') < 2 or not consts:
        return False
    return not any(_ALNUM_RE.search(t) for t in consts)

class ParsingCache(object):

    def __init__(self):
        self.template_tree = {}
        self.template_list = []


    def add_templates(self, event_template, insert=True, relevant_templates=[]):
        template_tokens = message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1
        if _is_punct_shell(template_tokens):
            return -1
        if insert or len(relevant_templates) == 0:
            id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return id
        max_similarity = 0
        similar_template = None
        for rt in relevant_templates:
            splited_template1, splited_template2 = rt.split(), event_template.split()
            if len(splited_template1) != len(splited_template2):
                continue 
            similarity = lcs_similarity(splited_template1, splited_template2)
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template = rt
        if max_similarity > 0.8:
            success, id = self.modify(similar_template, event_template)
            if not success:
                id = self.insert(event_template, template_tokens, len(self.template_list))
                self.template_list.append(event_template)
            return id
        else:
            id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return id


    def insert(self, event_template, template_tokens, template_id):
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            self.template_tree[start_token] = {}
        move_tree = self.template_tree[start_token]

        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                move_tree[token] = {}
            move_tree = move_tree[token]
            tidx += 1

        move_tree["".join(template_tokens)] = (
            sum(1 for s in template_tokens if s != "<*>"),
            template_tokens.count("<*>"),
            event_template,
            template_id
        )
        return template_id


    def modify(self, similar_template, event_template):
        merged_template = []
        similar_tokens = similar_template.split()
        event_tokens = event_template.split()
        i = 0
        for token in similar_tokens:
            if token == event_tokens[i]:
                merged_template.append(token)
            else:
                merged_template.append("<*>")
            i += 1
        merged_template = " ".join(merged_template)
        success, old_ids = self.delete(similar_template)
        if not success:
            return False, -1
        self.insert(merged_template, message_split(merged_template), old_ids)
        self.template_list[old_ids] = merged_template
        return True, old_ids
        

    def delete(self, event_template):
        template_tokens = message_split(event_template)
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            return False, []
        move_tree = self.template_tree[start_token]

        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                return False, []
            move_tree = move_tree[token]
            tidx += 1
        old_id = move_tree["".join(template_tokens)][3]
        del move_tree["".join(template_tokens)]
        return True, old_id


    def match_event(self, log):
        return tree_match(self.template_tree, log)


    def _preprocess_template(self, template):
        return template


def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')']
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


def message_split(message):
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))

    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' '
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens


def tree_match(match_tree, log_content):
    log_tokens = message_split(log_content)
    template, template_id, parameter_str = match_template(match_tree, log_tokens)
    if template:
        return (template, template_id, parameter_str)
    else:
        return ("NoMatch", "NoMatch", parameter_str)


def _rank(value):
    return (-value[0], value[1])

class _Best(object):
    __slots__ = ('best',)


    def __init__(self):
        self.best = None


    def offer(self, key, value, params):
        if key is None or value is None or params is None:
            return
        if self.best is None or _rank(value) < _rank(self.best[1]):
            self.best = (key, value, params)


def match_template(match_tree, log_tokens):
    _FT_MEMO.clear()
    _FT_MEMO_STATS['calls'] = _FT_MEMO_STATS['hits'] = _FT_MEMO_STATS['misses'] = 0
    best = _Best()
    find_results = find_template(match_tree, log_tokens, best, [], 1)
    relevant_templates = find_results[1]
    if best.best is not None:
        _k, _v, _p = best.best
        return _v[2], _v[3], _p
    return False, False, relevant_templates


def get_all_templates(move_tree):
    result = []
    for key, value in move_tree.items():
        if isinstance(value, tuple):
            result.append(value[2])
        else:
            result = result + get_all_templates(value)
    return result

import os as _os

_FT_MEMO = {}
_FT_MEMO_STATS = {'calls': 0, 'hits': 0, 'misses': 0}
_FT_MEMO_ON = _os.environ.get('DualParser_FT_MEMO', '1') != '0'


def find_template(move_tree, log_tokens, result, parameter_list, depth):
    _FT_MEMO_STATS['calls'] += 1
    if not _FT_MEMO_ON or '<*>' not in move_tree:
        return _find_template_raw(move_tree, log_tokens, result, parameter_list, depth)

    key = (id(move_tree), tuple(log_tokens))
    hit = _FT_MEMO.get(key)
    if hit is not None:
        _FT_MEMO_STATS['hits'] += 1
        ret, b = hit
        if b is not None:
            _k, _v, _suf = b
            result.offer(_k, _v, tuple(parameter_list) + _suf)
        return (ret[0], ret[1])

    _FT_MEMO_STATS['misses'] += 1
    entry_len = len(parameter_list)
    sub_best = _Best()
    ret = _find_template_raw(move_tree, log_tokens, sub_best, parameter_list, depth)
    if sub_best.best is not None:
        _k, _v, _p = sub_best.best
        result.offer(_k, _v, _p)
        b = (_k, _v, tuple(_p)[entry_len:])
    else:
        b = None
    _FT_MEMO[key] = (ret, b)
    return ret


def _find_template_raw(move_tree, log_tokens, result, parameter_list, depth):
    flag = 0
    if len(log_tokens) == 0:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.offer(key, value, tuple(parameter_list))
                flag = 2
        if "<*>" in move_tree:
            parameter_list.append("")
            move_tree = move_tree["<*>"]
            if isinstance(move_tree, tuple):
                flag = 2
            else:
                for key, value in move_tree.items():
                    if isinstance(value, tuple):
                        result.offer(key, value, tuple(parameter_list))
                        flag = 2
    else:
        token = log_tokens[0]

        relevant_templates = []
        
        if token in move_tree:
            find_result = find_template(
                move_tree[token], log_tokens[1:], result, parameter_list, depth + 1
            )
            if find_result[0]:
                flag = 2
            elif flag != 2:
                flag = 1
                relevant_templates = relevant_templates + find_result[1]
        if "<*>" in move_tree:
            if isinstance(move_tree["<*>"], dict):
                next_keys = move_tree["<*>"].keys()
                next_continue_keys = []
                for nk in next_keys:
                    nv = move_tree["<*>"][nk]
                    if not isinstance(nv, tuple):
                        next_continue_keys.append(nk)
                idx = 0
                while idx < len(log_tokens):
                    token = log_tokens[idx]
                    if token in next_continue_keys:
                        parameter_list.append("".join(log_tokens[0:idx]))
                        find_result = find_template(
                            move_tree["<*>"], log_tokens[idx:], result, parameter_list,depth+1
                        )
                        if find_result[0]:
                            flag = 2
                        elif flag != 2:
                            flag = 1
                        if parameter_list:
                            parameter_list.pop()
                    idx += 1
                if idx == len(log_tokens):
                    parameter_list.append("".join(log_tokens[0:idx]))
                    find_result = find_template(
                        move_tree["<*>"], log_tokens[idx + 1 :], result, parameter_list,depth+1
                    )
                    if find_result[0]:
                        flag = 2
                    else:
                        if flag != 2:
                            flag = 1
                        relevant_templates = relevant_templates + find_result[1]
                    if parameter_list:
                        parameter_list.pop()
    if flag == 2:
        return (True, [])
    if flag == 1:
        return (False, relevant_templates)
    if flag == 0:
        if depth >= 2:
            return (False, get_all_templates(move_tree))
        else:
            return (False, [])