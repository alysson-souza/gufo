"""How the harness asks Flint for a reasoning-free answer.

Flint has no thinking switch (docs/SERVER.md): it opens `<think>` on every
turn and ignores `reasoning_effort` and `enable_thinking`. Its only cap is a
reasoning budget, and a budget of 0 closes `</think>` at once, which is how a
client gets a short answer without reasoning. Under `--sampling-preset flint`
the harness therefore adds `reasoning_budget_tokens: 0` to every Chat or
Responses request that turns thinking off and sets no budget of its own, so
checks that rely on a thinking-off answer keep their exact expectations.

A check that needs Flint's own reasoning on such a request sets the budget
itself; -1 keeps the server default (`keep_reasoning`). After a forced
`</think>` Flint often continues its reasoning as content (the llama.cpp
oracle does too), so checks that assert an exact short answer or a tool call
keep Flint's reasoning and give it room instead of relying on budget 0.

Flint's template renders no `<think></think>` for a turn without reasoning,
so a budget-0 turn replayed by a client never matches its generated tokens.
Checks that require the next turn to reuse the generated frontier keep
Flint's reasoning and echo it back as `reasoning_content` instead.
"""

import json

BUDGET_FIELDS = ("reasoning_budget_tokens", "thinking_budget_tokens")


def thinking_off(body):
    """Whether a Chat or Responses body asks for no reasoning."""
    reasoning = body.get("reasoning")
    kwargs = body.get("chat_template_kwargs")
    thinking = body.get("thinking")
    return (body.get("reasoning_effort") in ("none", "off")
            or (isinstance(reasoning, dict) and reasoning.get("effort") == "none")
            or (isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False)
            or (isinstance(thinking, dict) and thinking.get("type") == "disabled"))


def budgeted(body):
    """`body` with Flint's thinking-off budget, or `body` itself if unchanged."""
    if not thinking_off(body) or any(field in body for field in BUDGET_FIELDS):
        return body
    return {**body, "reasoning_budget_tokens": 0}


def budget_request(request):
    """Rewrite a Chat or Responses request in place before it is sent or recorded.

    httpx event hooks cannot replace the request, so the rewritten request is
    built through the public constructor and its state adopted wholesale."""
    if request.method != "POST" or not request.url.path.endswith(
            ("/chat/completions", "/responses")):
        return
    body = json.loads(request.read() or b"{}")
    patched = budgeted(body)
    if patched is body:
        return
    import httpx  # Raw-HTTP scripts (continuation.py) use only budgeted().
    headers = [(key, value) for key, value in request.headers.multi_items()
               if key.lower() != "content-length"]
    rewritten = httpx.Request(request.method, request.url, headers=headers,
                              content=json.dumps(patched).encode(),
                              extensions=request.extensions)
    rewritten.read()
    vars(request).update(vars(rewritten))


async def async_budget_request(request):
    budget_request(request)


def keep_reasoning(sampling_preset):
    """Extra body fields that let Flint reason on a thinking-off request."""
    return {"reasoning_budget_tokens": -1} if sampling_preset == "flint" else {}
