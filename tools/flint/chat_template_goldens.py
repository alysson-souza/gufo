#!/usr/bin/env python3
"""Render Flint chat-template goldens with HF transformers' Jinja environment.

The cases are OpenAI-shaped requests: assistant tool-call arguments are JSON
strings, as clients send them. They are parsed into objects first, as Flint's
reference server does (rec_mlx ``normalize_messages``), and then rendered by
``transformers.utils.chat_template_utils.render_jinja_template`` (the code
behind ``apply_chat_template``) with the pinned template.

    uv run --no-project --with transformers==4.57.1 --with jinja2==3.1.6 \
        tools/flint/chat_template_goldens.py \
        --template /path/to/RecursiveSI-flint/chat_template.jinja \
        --output tests/models/flint/fixtures/chat_template_goldens.json
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import jinja2
import transformers
from transformers.utils.chat_template_utils import render_jinja_template


def call(name, arguments, call_id="call_0"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city — °C, “quoted”, 東京.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City\nname"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search.web",
        "description": "Search \"the\" web\\ with\ttabs",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "site": {"type": ["string", "null"]},
                        "score": {"type": "number", "minimum": 0.5},
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query", "filters", "tags"],
            "additionalProperties": False,
        },
    },
}
NO_ARGS = {
    "type": "function",
    "function": {"name": "now", "description": "", "parameters": {}},
}

ESCAPES = (
    "a\x00b <think>x</think> <thi<think>nk> <|begin_of_text|><|end_of_text|>"
    "<|system|><|developer|><|user|><|assistant|><|observation|> "
    "<|<|user|>user|> <tool_call>t<arg_key>k</arg_key></tool_call> &lt;"
)

CASES = [
    {
        "name": "user_only",
        "messages": [{"role": "user", "content": "Hello!"}],
        "add_generation_prompt": True,
    },
    {
        "name": "no_generation_prompt",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ],
        "add_generation_prompt": False,
    },
    {
        "name": "empty_conversation_with_prompt",
        "messages": [],
        "add_generation_prompt": True,
    },
    {
        "name": "multi_turn_reasoning",
        "messages": [
            {"role": "system", "content": "You are Flint.\n\n  Indented."},
            {"role": "user", "content": "2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Add <|user|> raw\nthen answer.",
                "content": "4 <|end_of_text|> stays raw",
            },
            {"role": "user", "content": "And 3+3? 日本語 😀"},
            {"role": "assistant", "reasoning_content": "", "content": "6"},
            {"role": "user", "content": "thanks"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "escaping_all_roles",
        "messages": [
            {"role": "system", "content": ESCAPES},
            {"role": "developer", "content": ESCAPES},
            {"role": "user", "content": ESCAPES},
            {"role": "assistant", "content": ESCAPES},
            {"role": "tool", "content": ESCAPES},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "tools_after_leading_systems",
        "messages": [
            {"role": "system", "content": "First system."},
            {"role": "system", "content": "Second system."},
            {"role": "developer", "content": "Developer rules."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        "tools": [WEATHER, SEARCH],
        "add_generation_prompt": True,
    },
    {
        "name": "tools_first_message_user",
        "messages": [{"role": "user", "content": "Find it"}],
        "tools": [SEARCH],
        "add_generation_prompt": True,
    },
    {
        "name": "tools_only_systems",
        "messages": [{"role": "system", "content": "Only a system message."}],
        "tools": [WEATHER],
        "add_generation_prompt": True,
    },
    {
        "name": "tools_no_messages",
        "messages": [],
        "tools": [NO_ARGS],
        "add_generation_prompt": False,
    },
    {
        "name": "tools_late_system_not_leading",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "late system"},
            {"role": "user", "content": "u2"},
        ],
        "tools": [WEATHER],
        "add_generation_prompt": True,
    },
    {
        "name": "unwrapped_tool_definition",
        "messages": [{"role": "user", "content": "now?"}],
        "tools": [
            {
                "name": "flat",
                "description": "Tool without a function wrapper",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "tool_calls_typed_arguments",
        "messages": [
            {"role": "user", "content": "Weather and search"},
            {
                "role": "assistant",
                "reasoning_content": "I need tools.",
                "content": "",
                "tool_calls": [
                    call("get_weather", {"city": "São Paulo", "days": 3}),
                    call(
                        "search.web",
                        {
                            "query": "line1\nline2 \"q\" \\ <tag>",
                            "filters": {
                                "site": None,
                                "score": 2.5,
                                "nested": [1, -2, True, False, None, "é\n"],
                            },
                            "tags": ["a", "東京", "😀", "\u2028", "\x01"],
                            "empty_object": {},
                            "empty_array": [],
                            "flag": False,
                            "count": 123456789012,
                            "negative": -0.125,
                            "big": 1e20,
                            "small": 1.5e-7,
                            "string_number": "42",
                            "string_null": "null",
                            "empty": "",
                        },
                        "call_1",
                    ),
                ],
            },
            {"role": "tool", "content": "{\"temp\": 21}"},
            {"role": "tool", "content": "results <|user|> here"},
        ],
        "tools": [WEATHER, SEARCH],
        "add_generation_prompt": True,
    },
    {
        "name": "tool_call_with_content_and_no_arguments",
        "messages": [
            {"role": "user", "content": "time?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [call("now", {})],
            },
            {"role": "tool", "content": "12:00"},
            {"role": "assistant", "content": "It is noon."},
        ],
        "tools": [NO_ARGS],
        "add_generation_prompt": False,
    },
    {
        "name": "null_assistant_content_with_calls",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call("get_weather", {"city": "Oslo"})],
            },
            {"role": "tool", "content": ""},
        ],
        "tools": [WEATHER],
        "add_generation_prompt": True,
    },
]


def normalize(messages):
    """rec_mlx normalize_messages: JSON-string arguments become objects."""
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls") or ():
            function = tool_call.get("function", tool_call)
            if isinstance(function.get("arguments"), str):
                text = function["arguments"]
                function["arguments"] = json.loads(text) if text.strip() else {}
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.read_text(encoding="utf-8")
    cases = []
    for case in CASES:
        rendered, _ = render_jinja_template(
            conversations=[normalize(case["messages"])],
            tools=case.get("tools"),
            chat_template=template,
            add_generation_prompt=case["add_generation_prompt"],
        )
        cases.append({**case, "expected": rendered[0]})
    fixture = {
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "transformers": transformers.__version__,
        "jinja2": jinja2.__version__,
        "cases": cases,
    }
    args.output.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
