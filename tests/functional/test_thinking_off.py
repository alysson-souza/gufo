"""Flint's thinking-off budget reaches the wire and the recorded request."""

import json
import unittest

import httpx
from openai import OpenAI

from thinking_off import budget_request, budgeted, keep_reasoning


class ThinkingOffTest(unittest.TestCase):
    def test_only_thinking_off_requests_without_a_budget_change(self):
        for body in ({"reasoning_effort": "none"}, {"reasoning_effort": "off"},
                     {"reasoning": {"effort": "none"}},
                     {"chat_template_kwargs": {"enable_thinking": False}},
                     {"thinking": {"type": "disabled"}}):
            self.assertEqual(budgeted(body), {**body, "reasoning_budget_tokens": 0})
        for body in ({}, {"reasoning_effort": "low"},
                     {"chat_template_kwargs": {"enable_thinking": True}},
                     {"reasoning_effort": "none", **keep_reasoning("flint")},
                     {"reasoning_effort": "none", "thinking_budget_tokens": 4}):
            self.assertIs(budgeted(body), body)
        self.assertEqual(keep_reasoning("qwen38"), {})

    def test_client_sends_and_records_the_budget(self):
        sent, recorded = [], []

        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}]})

        client = OpenAI(api_key="test", base_url="http://127.0.0.1:1/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler), event_hooks={
                            "request": [budget_request,
                                        lambda request: recorded.append(json.loads(request.read()))]}))
        for effort in ("none", "low"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}],
                                           reasoning_effort=effort)
        self.assertEqual(sent, recorded)
        self.assertEqual([body.get("reasoning_budget_tokens") for body in sent], [0, None])
        client.post("/messages", body={"thinking": {"type": "disabled"}}, cast_to=object)
        self.assertNotIn("reasoning_budget_tokens", sent[-1])


if __name__ == "__main__":
    unittest.main()
