# -*- coding: utf-8 -*-
import re
import regex as _regex



def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template."""
    boolean = {}
    default_strings = {}
    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    template = template.strip()
    template = _regex.sub(r'\s+', ' ', template)

    tokens = _regex.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:
        if _regex.match(r'^\d+$', token):
            token = '<*>'
        if _regex.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':
                token = '<*>'
        new_tokens.append(token)
    template = ''.join(new_tokens)

    while True:
        prev = template
        template = _regex.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break
    while True:
        prev = template
        template = _regex.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    for a, b in ((" #<*># ", " <*> "), (" #<*> ", " <*> "),
                 ("<*>:<*>", "<*>"), ("<*>#<*>", "<*>"),
                 ("<*>/<*>", "<*>"), ("<*>@<*>", "<*>"),
                 ("<*>.<*>", "<*>"), (' "<*>" ', " <*> "),
                 (" '<*>' ", " <*> "), ("<*><*>", "<*>")):
        while a in template:
            template = template.replace(a, b)
    return template



def pp_structural(template, max_iter=8):
    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {r'\s', r'\,', r'\!', r'\;', r'\:',
                       r'\=', r'\|', r'\"', r'\'',
                       r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'}
    token_delimiters = path_delimiters.union(
        {r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&'})

    def once(t):
        t = t.strip()
        t = re.sub(r'\s+', ' ', t)

        p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', t)
        new_p = []
        for p in p_tokens:
            if re.match(r'^(\/[^\/]+)+$', p):
                p = '<*>'
            new_p.append(p)
        t = ''.join(new_p)

        tokens = re.split('(' + '|'.join(token_delimiters) + ')', t)
        new_t = []
        for token in tokens:
            for to_replace in boolean.union(default_strings):
                if token.lower() == to_replace.lower():
                    token = '<*>'
            if re.match(r'^\d+$', token):
                token = '<*>'
            if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
                if token != '<*>/<*>':
                    token = '<*>'
            new_t.append(token)
        t = ''.join(new_t)

        for pattern in (r'<\*>\.<\*>', r'<\*><\*>'):
            while True:
                prev = t
                t = re.sub(pattern, '<*>', t)
                if prev == t:
                    break
        return t

    cur = template
    for _ in range(max_iter):
        nxt = once(cur)
        if nxt == cur:
            return nxt
        cur = nxt
    return cur


def pp_fixed(template, max_iter=8):
    cur = pp_structural(template)
    for _ in range(max_iter):
        nxt = correct_single_template(cur)
        if nxt == cur:
            return nxt
        cur = nxt
    return cur


def canon_templates(templates):
    seen = {}
    out = []
    for t in templates:
        v = seen.get(t)
        if v is None:
            v = pp_fixed(t)
            seen[t] = v
        out.append(v)
    return out
