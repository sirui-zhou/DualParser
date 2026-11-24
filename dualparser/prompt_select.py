import textdistance
import random


def lcs_distance(x, y):
    seq1 = x.split()
    seq2 = y.split()
    lengths = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))
    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))

def lev_distance(x, y):
    return textdistance.levenshtein.normalized_distance(x, y)


def euc_distance(x, y):
    return textdistance.cosine.normalized_distance(x, y)


def jaccard_distance(x, y):
    return textdistance.jaccard.normalized_distance(x.split(), y.split())


def ratcliff_distance(x, y):
    return textdistance.ratcliff_obershelp.normalized_distance(x, y)


def clean(s):
    """Keep log text unchanged before similarity scoring."""
    return s


def prompt_select(prompts, log, demonstration, selection_method="LILAC"):
    if demonstration == 0:
        return []
    if selection_method == "random":
        print("random selection...")
        result = random.choices(prompts, k=demonstration)
    else:
        log = clean(log)
        for d in prompts:
            d['similarity'] = jaccard_distance(clean(d['query']), log)
        result = sorted(prompts, key=lambda d: d['similarity'])[:demonstration]

    return [
        {
            'query': f"Log message: `{item['query']}`",
            'answer': f"Log template: `{item['answer']}`",
        }
        for item in result
    ]
