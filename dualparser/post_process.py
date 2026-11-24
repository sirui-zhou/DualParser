import regex as re


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template."""
    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:
        if re.match(r'^\d+$', token):
            token = '<*>'

        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':
                token = '<*>'

        new_tokens.append(token)

    template = ''.join(new_tokens)

    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    replacements = (
        (" #<*># ", " <*> "),
        (" #<*> ", " <*> "),
        ("<*>:<*>", "<*>"),
        ("<*>#<*>", "<*>"),
        ("<*>/<*>", "<*>"),
        ("<*>@<*>", "<*>"),
        ("<*>.<*>", "<*>"),
        (' "<*>" ', " <*> "),
        (" '<*>' ", " <*> "),
        ("<*><*>", "<*>"),
    )
    for old, new in replacements:
        while old in template:
            template = template.replace(old, new)
    return template