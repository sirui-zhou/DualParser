import os
import math


def environment_rate(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    rate = float(value)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return rate


def token_cost_usd(input_tokens, output_tokens, input_rate, output_rate):
    if input_rate is None or output_rate is None:
        return None
    return (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000.0


def build_cost_report(api_stats, t5_stats):
    api_input_rate = environment_rate(
        "DUALPARSER_API_INPUT_USD_PER_MILLION_TOKENS"
    )
    api_output_rate = environment_rate(
        "DUALPARSER_API_OUTPUT_USD_PER_MILLION_TOKENS"
    )
    deployment_hourly_rate = environment_rate(
        "DUALPARSER_OPEN_SOURCE_DEPLOYMENT_USD_PER_HOUR"
    )

    if api_stats["attempts"] == 0:
        api_cost = 0.0
    elif api_stats["usage_missing_requests"] > 0:
        api_cost = None
    else:
        api_cost = token_cost_usd(
            api_stats["prompt_tokens"],
            api_stats["completion_tokens"],
            api_input_rate,
            api_output_rate,
        )
    t5_total_tokens = t5_stats["input_tokens"] + t5_stats["output_tokens"]
    deployment_cost = None
    if t5_stats["elapsed_seconds"] == 0:
        deployment_cost = 0.0
    elif deployment_hourly_rate is not None:
        deployment_cost = (
            t5_stats["elapsed_seconds"] * deployment_hourly_rate / 3600.0
        )
    known_costs = [cost for cost in (api_cost, deployment_cost) if cost is not None]
    total_cost = None
    if api_cost is not None and deployment_cost is not None:
        total_cost = api_cost + deployment_cost

    return {
        "currency": "USD",
        "commercial_api": {
            "input_usd_per_million_tokens": api_input_rate,
            "output_usd_per_million_tokens": api_output_rate,
            "estimated_cost_usd": api_cost,
        },
        "open_source_t5": {
            "deployment_usd_per_hour": deployment_hourly_rate,
            "input_tokens": t5_stats["input_tokens"],
            "output_tokens": t5_stats["output_tokens"],
            "inference_seconds": t5_stats["elapsed_seconds"],
            "estimated_deployment_cost_usd": deployment_cost,
            "effective_usd_per_million_tokens": (
                deployment_cost * 1_000_000.0 / t5_total_tokens
                if deployment_cost is not None and t5_total_tokens > 0
                else None
            ),
        },
        "known_cost_subtotal_usd": sum(known_costs),
        "total_cost_usd": total_cost,
        "complete": api_cost is not None and deployment_cost is not None,
    }
