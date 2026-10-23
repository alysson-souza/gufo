"""Incremental native tool arguments, stopped calls, and successful retries."""

import json
import time

import openai

from metrics import validate_tool_events


TEXT = (
    'First line: preserve "quotes", backslashes \\, and café.\n'
    "Literal <tool_call>, </parameter> is data, <|im_end|>, and </think>.\n"
    + "alpha beta gamma delta; " * 12
    + "STOP_HERE\nThe final line."
)
EXPECTED = {"text": TEXT, "count": 42, "values": [True, None, {"key": "value"}]}
FUNCTION = {
    "name": "record",
    "description": "Record these values exactly once.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "const": TEXT},
            "count": {"type": "integer", "const": 42},
            "values": {"type": "array", "const": EXPECTED["values"]},
        },
        "required": list(EXPECTED),
        "additionalProperties": False,
    },
}


def request(model, api, thinking=False, sampling_preset=None):
    prompt = (
        "Call record exactly once, with text, count, and values matching the "
        "constants in its schema. Keep all literal characters. No visible prose."
    )
    if thinking:
        # Asked to match the constants, greedy reasoning transcribes the long
        # text (DeepSeek loops on its repeats); the call is what this streams.
        prompt = (
            "Call record exactly once. Its arguments are fixed by the constants "
            "in its schema; copy them exactly in the call. No visible prose."
        )
    common = dict(model=model, temperature=0)
    # Flint's grammar enforces argument constants only for strict tools; the
    # stream must carry exact arguments, so Chat and Responses ask for them.
    # Strict arrays need an items schema; the const still fixes the value.
    function = FUNCTION
    if sampling_preset == "flint":
        function = json.loads(json.dumps(FUNCTION))
        function["strict"] = True
        function["parameters"]["properties"]["values"]["items"] = {"anyOf": [
            {"type": "boolean"}, {"type": "null"},
            {"type": "object", "properties": {"key": {"type": "string"}},
             "required": ["key"], "additionalProperties": False}]}
    # Flint's reasoning transcribes the long constant before its call.
    limit = 4096 if sampling_preset == "flint" and thinking else 1024
    if api == "chat":
        return {**common, "messages": [{"role": "user", "content": prompt}],
                "tools": [{"type": "function", "function": function}],
                "tool_choice": "required", "parallel_tool_calls": False,
                "reasoning_effort": "low" if thinking else "none",
                "max_completion_tokens": limit, "seed": 347}
    if api == "responses":
        return {**common, "input": prompt,
                "tools": [{"type": "function", **function}],
                "tool_choice": "required", "parallel_tool_calls": False,
                "reasoning": {"effort": "low" if thinking else "none"},
                "max_output_tokens": limit, "store": False,
                "extra_body": {"seed": 347}}
    # Messages has no zero reasoning budget (thinking_off.py), and Flint ignores
    # disabled thinking; its shortest equivalent is a one-token budget.
    off = ({"type": "enabled", "budget_tokens": 1} if sampling_preset == "flint"
           else {"type": "disabled"})
    return {**common, "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": FUNCTION["name"],
                       "description": FUNCTION["description"],
                       "input_schema": FUNCTION["parameters"]}],
            "tool_choice": {"type": "tool", "name": "record",
                            "disable_parallel_tool_use": True},
            "thinking": {"type": "enabled", "budget_tokens": 256} if thinking else off,
            "output_config": {"effort": "low"},
            "max_tokens": 1024, "seed": 347}


def collect(client, api, body):
    started = time.monotonic()
    if api == "chat":
        stream = client.chat.completions.create(
            **body, stream=True, stream_options={"include_usage": True})
    elif api == "responses":
        stream = client.responses.create(**body, stream=True)
    else:
        stream = client.post("/messages", body={**body, "stream": True},
                             cast_to=object, stream=True,
                             stream_cls=openai.Stream[object])
    events, arrivals = [], []
    with stream:
        for item in stream:
            event = item if isinstance(item, dict) else item.to_dict()
            events.append(event)
            arrivals.append((time.monotonic() - started) * 1000)
    return analyze(api, events, arrivals)


def analyze(api, events, arrivals):
    """Check headers, identity, order, and fragments independently of the SDK."""
    arguments, names, identifiers, fragments = {}, {}, {}, []
    content, reasoning, finish, usage = "", "", None, None
    terminal = None
    for number, event in enumerate(events):
        assert not event.get("error") and event.get("type") != "response.failed", event

        def header(index, name, identifier):
            assert index not in identifiers and identifier and name == "record", event
            identifiers[index], names[index], arguments[index] = identifier, name, ""

        def append(index, text):
            assert index in identifiers and finish is None, event
            arguments[index] += text
            if text:
                fragments.append({"event": number, "ms": arrivals[number],
                                  "bytes": len(text.encode("utf-8"))})

        if api == "chat":
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice["delta"]
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for call in delta.get("tool_calls") or []:
                    function = call.get("function", {})
                    if call.get("id"):
                        header(call["index"], function.get("name"), call["id"])
                    else:
                        assert not function.get("name"), event
                    append(call["index"], function.get("arguments") or "")
                finish = choice.get("finish_reason") or finish
        elif api == "responses":
            kind = event["type"]
            if kind == "response.output_item.added" and event["item"]["type"] == "function_call":
                header(event["output_index"], event["item"]["name"], event["item"]["call_id"])
            elif kind == "response.function_call_arguments.delta":
                append(event["output_index"], event["delta"])
            elif kind == "response.output_text.delta":
                content += event["delta"]
            elif kind == "response.reasoning_summary_text.delta":
                reasoning += event["delta"]
            elif kind in ("response.completed", "response.incomplete"):
                terminal = event["response"]
                usage, finish = terminal["usage"], terminal["status"]
        else:
            kind = event["type"]
            if kind == "content_block_start" and event["content_block"]["type"] == "tool_use":
                block = event["content_block"]
                header(event["index"], block["name"], block["id"])
            elif kind == "content_block_delta":
                delta = event["delta"]
                if delta["type"] == "input_json_delta":
                    append(event["index"], delta["partial_json"])
                elif delta["type"] == "text_delta":
                    content += delta["text"]
                elif delta["type"] == "thinking_delta":
                    reasoning += delta["thinking"]
            elif kind == "message_delta":
                usage, finish = event["usage"], event["delta"]["stop_reason"]
    assert finish and usage, events[-3:]
    if api == "responses":
        validate_tool_events(events, terminal)
    return dict(arguments=list(arguments.values()), identifiers=list(identifiers.values()),
                content=content, reasoning=reasoning, finish=finish, usage=usage,
                fragments=fragments, terminal_ms=arrivals[-1])


def check_tool_streaming(client, model, checks, chat_result, sampling_preset=None):
    for api in ("chat", "responses", "messages"):
        # Messages tools have no strict flag, so Flint's Messages calls keep
        # only the call format: their constants are the model's to copy.
        exact = api != "messages" or sampling_preset != "flint"
        for thinking in (False, True):
            body = request(model, api, thinking, sampling_preset)
            result = collect(client, api, body)
            checks[f"tool_streaming_{api}_{'thinking' if thinking else 'plain'}"] = result
            assert result["finish"] == {
                "chat": "tool_calls", "responses": "completed", "messages": "tool_use"}[api], result
            assert len(result["arguments"]) == 1, result
            arguments = json.loads(result["arguments"][0])
            assert arguments == EXPECTED if exact else arguments.keys() == EXPECTED.keys(), result
            assert not result["content"].strip(), result
            # Flint's one-token Messages budget may or may not show reasoning.
            if thinking or api != "messages" or sampling_preset != "flint":
                assert bool(result["reasoning"].strip()) == thinking, result
            assert len(result["fragments"]) > 3, ("tool arguments were buffered", result)
            assert result["fragments"][0]["ms"] < result["fragments"][-1]["ms"] < result["terminal_ms"], result

        if not exact:
            # The stop and resume checks need STOP_HERE inside an enforced value.
            continue
        body = request(model, api, sampling_preset=sampling_preset)
        phase = "limited" if api == "responses" else "stopped"
        if api == "responses":
            # Responses has no stop-sequence request field.
            stopped = {**body, "max_output_tokens": 32}
        else:
            stopped = {**body, "stop" if api == "chat" else "stop_sequences": ["STOP_HERE"]}
        result = collect(client, api, stopped)
        checks[f"tool_streaming_{api}_{phase}"] = result
        assert result["finish"] == {
            "chat": "stop", "responses": "incomplete", "messages": "stop_sequence"}[api], result
        assert not result["content"].strip() and len(result["arguments"]) == 1, result
        assert "STOP_HERE" not in result["arguments"][0], result
        try:
            json.loads(result["arguments"][0])
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError(("interrupted tool was made executable", result))
        resumed = collect(client, api, body)
        checks[f"tool_streaming_{api}_resume"] = resumed
        assert len(resumed["arguments"]) == 1 and json.loads(resumed["arguments"][0]) == EXPECTED, resumed
        assert not resumed["content"].strip(), resumed
        assert set(resumed["identifiers"]).isdisjoint(result["identifiers"]), resumed

    # Byte limits interrupt argument generation; the next request starts cleanly.
    body = request(model, "chat", sampling_preset=sampling_preset)
    limited = chat_result(client, {**body, "max_completion_tokens": 32}, True)
    checks["tool_streaming_length"] = limited
    assert limited["finish"] == "length" and limited.get("partial_tools"), limited
    restored = chat_result(client, body, True)
    checks["tool_streaming_after_length"] = restored
    assert restored["finish"] == "tool_calls" and len(restored["tools"]) == 1, restored
    assert json.loads(restored["tools"][0]["function"]["arguments"]) == EXPECTED, restored

    for thinking in (False, True):
        body = request(model, "chat", thinking, sampling_preset)
        received, reasoning = "", ""
        with client.chat.completions.create(**body, stream=True) as stream:
            for chunk in stream:
                for choice in chunk.choices:
                    assert choice.finish_reason is None, "call finished before cancellation"
                    reasoning += getattr(choice.delta, "reasoning_content", "") or ""
                    for delta in choice.delta.tool_calls or []:
                        if delta.function:
                            received += delta.function.arguments or ""
                if len(received) >= 40:
                    break
            else:
                raise AssertionError("no partial argument reached the cancelling client")
        checks[f"tool_streaming_cancel_{thinking}"] = {
            "partial_arguments": received, "reasoning": reasoning}
        assert bool(reasoning.strip()) == thinking, reasoning
        resumed = chat_result(client, body, True)
        checks[f"tool_streaming_after_cancel_{thinking}"] = resumed
        assert resumed["finish"] == "tool_calls" and len(resumed["tools"]) == 1, resumed
        assert json.loads(resumed["tools"][0]["function"]["arguments"]) == EXPECTED, resumed
        assert not resumed["text"].strip(), resumed
