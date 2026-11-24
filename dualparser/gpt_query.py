import re
import string

from .llm_client import LLM_CALL_STATS, ensure_api_credentials, request_chat_completion
from .parsing_cache import ParsingCache
from .post_process import correct_single_template


def get_openai_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline().strip(), file.readline().strip()


def _ensure_api_key():
    """Compatibility wrapper for the former lazy-loading behavior."""
    ensure_api_credentials()


# ==============================================================================









# ==============================================================================
# LLM statistics are now centrally tracked in llm_client.py for easier reuse.
# We keep the same public name for compatibility with older code paths.




MAX_TOKENS = 16384


def infer_llm(
    instruction,
    exemplars,
    query,
    log_message,
    model='gpt-3.5-turbo-0613',
    temperature=0.0,
    max_tokens=MAX_TOKENS,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert of log parsing, and now you will help to do "
                "log parsing."
            ),
        },
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "Sure, I can help you with log parsing."},
    ]
    if exemplars is not None:
        for exemplar in exemplars:
            messages.append({"role": "user", "content": exemplar['query']})
            messages.append({"role": "assistant", "content": exemplar['answer']})
    messages.append({"role": "user", "content": query})

    retry_times = 0
    print("model: ", model)
    while retry_times < 3:
        try:
            # Load credentials before the client records a request attempt.
            _ensure_api_key()
            response = request_chat_completion(messages, model, temperature, max_tokens)
            return response
        except Exception as e:
            print("Exception :", e)
            retry_times += 1
            LLM_CALL_STATS['retry'] += 1
    print(f"Failed to get response from OpenAI after {retry_times} retries.")
    LLM_CALL_STATS['gave_up'] += 1
    if exemplars is not None and len(exemplars) > 0:
        default_example = {
            'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`',
            'answer': 'Log template: `try to connected to host: {ip_address}, finished.`',
        }
        if (exemplars[0]['query'], exemplars[0]['answer']) != (
            default_example['query'],
            default_example['answer'],
        ):
            examples = [default_example]
            return infer_llm(
                instruction, examples, query, log_message, model, temperature,
                max_tokens,
            )

    LLM_CALL_STATS['fallback_raw'] += 1
    return 'Log message: `{}`'.format(log_message)


def get_response_from_openai_key(
    query, examples=[], model='gpt-3.5-turbo-0613', temperature=0.0
):
    instruction = (
        "I want you to act like an expert of log parsing. I will give you a log "
        "message delimited by backticks. You must identify and abstract all the "
        "dynamic variables in logs with {placeholder} and output a static log "
        "template. Print the input log's template delimited by backticks. "
        "IMPORTANT: Your answer MUST start with the line 'Log template: `' and "
        "end with '`'. Output ONLY the template line, no explanation, no extra text."
    )
    if examples is None or len(examples) == 0:
        examples = [{
            'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`',
            'answer': 'Log template: `try to connected to host: {ip_address}, finished.`',
        }]
    question = 'Log message: `{}`'.format(query)
    return infer_llm(
        instruction, examples, question, query, model, temperature,
        max_tokens=MAX_TOKENS,
    )


def query_template_from_gpt(
    log_message, examples=[], model='gpt-3.5-turbo-0613'
):
    if len(log_message.split()) == 1:
        return log_message, False
    response = get_response_from_openai_key(log_message, examples, model)
    lines = response.split('\n')
    log_template = None
    for line in lines:
        if line.find("Log template:") != -1:
            log_template = line
            break
    if log_template is None:
        for line in lines:
            if line.find("`") != -1:
                log_template = line
                break
    if log_template is not None:
        start_index = log_template.find('`') + 1
        end_index = log_template.rfind('`')

        if start_index == 0 or end_index == -1:
            start_index = log_template.find('"') + 1
            end_index = log_template.rfind('"')

        if start_index != 0 and end_index != -1 and start_index < end_index:
            template = log_template[start_index:end_index]
            return template, True

    print("======================================")
    print("ChatGPT response format error: ")
    print(response)
    print("======================================")
    return log_message, False


def post_process_template(template, regs_common):
    pattern = r'\{(\w+)\}'
    template = re.sub(pattern, "<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(
    log_message,
    regs_common=[],
    examples=[],
    model="gpt-3.5-turbo-0613",
    candidate_template=None,
    spell_template=None,
    t5_template=None,
):
    if candidate_template or t5_template:
        # DualParser: give LLM Drain's candidate to verify/correct
        if spell_template:
            instruction = (
                "I want you to act like an expert of log parsing. I will give you "
                "a log message and two candidate templates from two automated "
                "parsers (Drain and Spell). They may have errors: too many "
                "wildcards or missing variables. Please pick the better one and "
                "correct it. Use <*> to replace ALL dynamic variables. Print the "
                "corrected template delimited by backticks. IMPORTANT: start with "
                "'Log template: `' and end with '`'. Output ONLY the template."
            )
            question = (
                'Log message: `{}`\nDrain candidate: `{}`\nSpell candidate: `{}`'
            ).format(log_message, candidate_template, spell_template)
        elif t5_template:
            # T5+LLM: give both Drain and T5 candidates
            instruction = (
                "I want you to act like an expert of log parsing. I will give you "
                "a log message and two candidate templates from two automated "
                "parsers (Drain and T5-small). They may have errors: too many "
                "wildcards or missing variables. T5-small may over-generalize "
                "(too many <*>). Please pick the better one and correct it. Use "
                "<*> to replace ALL dynamic variables. Print the corrected "
                "template delimited by backticks. IMPORTANT: start with "
                "'Log template: `' and end with '`'. Output ONLY the template."
            )
            question = (
                'Log message: `{}`\nDrain candidate: `{}`\nT5 candidate: `{}`'
            ).format(
                log_message,
                candidate_template if candidate_template else "(none)",
                t5_template,
            )
        else:
            instruction = (
                "I want you to act like an expert of log parsing. I will give you "
                "a log message and a candidate template generated by an automated "
                "parser. The candidate template may have errors: it may have too "
                "many wildcards <*> or miss variables. Please verify and correct "
                "it. Use <*> to replace ALL dynamic variables (numbers, IPs, "
                "paths, IDs, etc.). Print the corrected template delimited by "
                "backticks. IMPORTANT: Your answer MUST start with "
                "'Log template: `' and end with '`'. Output ONLY the template "
                "line, no explanation."
            )
            question = (
                'Log message: `{}`\nCandidate template: `{}`'
            ).format(log_message, candidate_template)
        response = infer_llm(
            instruction,
            examples,
            question,
            log_message,
            model,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        lines = response.split('\n')
        log_template = None
        for line in lines:
            if "Log template:" in line:
                log_template = line
                break
        if log_template:
            start_index = log_template.find('`') + 1
            end_index = log_template.rfind('`')
            if start_index != 0 and end_index != -1 and start_index < end_index:
                template = log_template[start_index:end_index]
                template, flag = post_process_template(template, regs_common)
                if flag:
                    tree = ParsingCache()
                    tree.add_templates(template)
                    if tree.match_event(log_message)[0] != "NoMatch":
                        return template, True
    # Fallback: original from-scratch query
    template, flag = query_template_from_gpt(log_message, examples, model)
    if len(template) == 0 or not flag:
        print(f"ChatGPT error")
    else:
        tree = ParsingCache()
        template, flag = post_process_template(template, regs_common)
        if flag:
            tree.add_templates(template)
            if tree.match_event(log_message)[0] == "NoMatch":
                print("==========================================================")
                print(log_message)
                print(
                    "ChatGPT template wrong: cannot match itself! "
                    "And the wrong template is : "
                )
                print(template)
                print("==========================================================")
            else:
                return template, True
    return post_process_template(log_message, regs_common)
