"""REST adapters for structured answer generation."""

import json
from urllib.parse import quote

import httpx
from langchain_core.utils.json_schema import dereference_refs

from .rag import CitedAnswer, RagFlowError

MAX_OUTPUT_TOKENS = 2048
PROVIDER_NAMES = {"gemini": "Gemini", "groq": "Groq", "cloudflare": "Cloudflare Workers AI"}


def _rest_schema(node):
    # Both providers support this subset. The full Pydantic constraints and
    # the current question's allowed quote IDs are checked after generation.
    schema = {key: node[key] for key in ("type", "required", "additionalProperties") if key in node}
    if "properties" in node:
        schema["properties"] = {name: _rest_schema(value) for name, value in node["properties"].items()}
    if "items" in node:
        schema["items"] = _rest_schema(node["items"])
    return schema


REST_SCHEMA = _rest_schema(dereference_refs(CitedAnswer.model_json_schema(), skip_keys=["$defs"]))


def _post(provider, url, api_key, timeout, payload):
    name = PROVIDER_NAMES[provider]
    try:
        response = httpx.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status == 429:
            raise
        raise RagFlowError(
            f"{name} returned HTTP {status}. Check the provider status and your API key and model settings."
        ) from error
    except httpx.TimeoutException as error:
        raise RagFlowError(f"The {name} request timed out. Try again.") from error
    except httpx.RequestError as error:
        raise RagFlowError(f"Could not complete the request to {name}. Check your connection and try again.") from error
    return response


def invoke_groq(api_key, model, timeout, messages):
    response = _post("groq", "https://api.groq.com/openai/v1/chat/completions", api_key, timeout, {
        "model": model,
        "messages": messages,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cited_answer", "strict": True, "schema": REST_SCHEMA},
        },
    })
    try:
        choice = response.json()["choices"][0]
        if choice["finish_reason"] == "stop":
            return json.loads(choice["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    raise RagFlowError("Groq returned incomplete or invalid answer data. No answer was accepted; try again.")


def invoke_cloudflare(api_token, account_id, model, timeout, messages):
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{quote(account_id, safe='')}/ai/run"
    response = _post("cloudflare", f"{endpoint}/{quote(model, safe='@/')}", api_token, timeout, {
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": REST_SCHEMA},
    })
    try:
        body = response.json()
        if body["success"] is True:
            content = body["result"]["response"]
            return json.loads(content) if isinstance(content, str) else content
    except (KeyError, TypeError, ValueError):
        pass
    raise RagFlowError(
        "Cloudflare Workers AI returned incomplete or invalid answer data. No answer was accepted; try again."
    )
