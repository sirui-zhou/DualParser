# DualParser

A collaborative log parsing framework that pairs lightweight statistical parsing
with a large language model. Most messages are resolved locally — by a template
cache, a Drain-style tree, and a vote among three statistical parsers
(Spell / IPLoM / LFA). Only the small fraction of messages whose templates remain
ambiguous is sent to the LLM.

The released configuration uses **no labels**: no ground-truth templates are used
for prompting, and no model is trained on any labelled data.

## How it works

```
log line
   |
   +- cache hit ----------------> reuse template
   |
   +- cache miss
        +- log bucket (batched)
             +- Spell / IPLoM / LFA vote
                  +- consensus  --> accept frozen template
                  +- no consensus -> LLM repair
                                       +- retok: re-render from the source line
```

- **Template cache.** Extracted templates are indexed and reused, so repeated
  events never trigger another LLM call.
- **Bucket-level vote.** Messages that miss the cache are batched and parsed by
  three statistical parsers; only disagreements escalate.
- **LLM repair.** The LLM receives the message plus the candidate template; the
  returned template is re-rendered against the original line before being
  cached.
- **Benchmark-ready.** Ships with the evaluation code for GA, FGA, PA and FTA on
  the LogHub datasets.

## Installation

```bash
git clone https://github.com/<your-org>/DualParser.git
cd DualParser
pip install -r requirements.txt
```

Python 3.11 or newer is recommended. `torch` and `transformers` are only needed
for the optional T5 second-pass check (`--use_t5`); the default pipeline does not
import them.

## Data

Put each dataset under `datasets/<Dataset>/` as:

```
datasets/HDFS/HDFS_full.log                     # raw log, one message per line
datasets/HDFS/HDFS_full.log_structured.csv      # reference templates (only used by --eval)
```

`scripts/download_data.sh` fetches the LogHub datasets, or point `--data_dir` at
your own copy.

## Configuration

The LLM endpoint is configured through environment variables, so no key is
stored in the repository:

```bash
export LLM_API_KEY="your-key"                     # or DEEPSEEK_API_KEY / OPENAI_API_KEY
export LLM_MODEL="deepseek-chat"                  # optional
export LLM_BASE_URL="https://api.deepseek.com/v1" # optional
```

Optional runtime directories:

```bash
export DUALPARSER_CACHE_DIR=./temp/dualparser     # template cache
export DUALPARSER_RESULT_DIR=./result
```

## Usage

Parse one dataset and report the four metrics:

```bash
python main.py --dataset HDFS --data_dir datasets --out_dir result --data_type full --eval
```

Parse all 14 datasets:

```bash
bash run.sh
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--cold_start` | ignore any existing cache and parse from scratch |
| `--use_t5` | enable the optional T5 second-pass check (off by default) |
| `--shot` | number of prompt examples; the release uses `0` |
| `--no_tpl_canon` | disable template canonicalisation |
| `--eval` | compute GA / PA / FGA / FTA after parsing |

## Output

```
result/
  <Dataset>_<type>.log_structured.csv   # LineId, EventId, Content, EventTemplate
  <Dataset>_<type>.log_templates.csv    # template inventory
  _evaluation_<Dataset>.json            # per-dataset counters
  _llm_calls_<Dataset>.txt              # LLM call and token accounting
```

## Evaluation

Metrics follow the LogHub canonicalisation (`correct_single_template` applied to
both the prediction and the reference before comparison):

- **GA** — grouping accuracy
- **PA** — parsing accuracy (per-message template equality)
- **FGA** — grouping F1
- **FTA** — template-level F1

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
