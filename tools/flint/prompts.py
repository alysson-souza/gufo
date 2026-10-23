#!/usr/bin/env python3
# /// script
# dependencies = ["transformers", "jinja2", "tokenizers"]
# ///
"""Write the token ids of the Flint numerical-parity prompts.

Tokenization is independent of Gufo: the official tokenizer.json and
chat_template.jinja from the source checkpoint, through Hugging Face
`tokenizers` and `transformers` apply_chat_template.

  uv run tools/flint/prompts.py --source /path/to/RecursiveSI-flint --out DIR

Writes DIR/{chat,code,long}.ids (comma-separated ids). `long` crosses the
513-token sliding window more than twice so windowed and full (NoPE) layers
both see long histories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

CHAT = [
    {"role": "system", "content": "You are a concise assistant. Answer in "
                                  "plain English and avoid jargon."},
    {"role": "user", "content": "What is the capital of Australia?"},
    {"role": "assistant", "content": "The capital of Australia is Canberra."},
    {"role": "user",
     "content": "Explain in two sentences why the sky looks blue during the "
                "day but red at sunset."},
]

CODE = '''import heapq
from collections import defaultdict


def dijkstra(graph: dict[str, list[tuple[str, int]]], source: str) -> dict[str, int]:
    """Return the shortest distance from source to every reachable node."""
    dist = defaultdict(lambda: float("inf"))
    dist[source] = 0
    queue = [(0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist[node]:
            continue
        for neighbour, weight in graph.get(node, []):
            candidate = d + weight
            if candidate < dist[neighbour]:
                dist[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return dict(dist)


def test_dijkstra():
    graph = {"a": [("b", 1), ("c", 4)], "b": [("c", 2), ("d", 5)], "c": [("d", 1)]}
    assert dijkstra(graph, "a") == '''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="checkpoint dir with tokenizer.json and chat_template.jinja")
    parser.add_argument("--long-text", type=Path,
                        help="document for the long prompt (default: SOURCE/LICENSE)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.source)
    template = (args.source / "chat_template.jinja").read_text()

    def chat(messages):
        text = tok.apply_chat_template(messages, chat_template=template,
                                       tokenize=False, add_generation_prompt=True)
        return tok.encode(text, add_special_tokens=False)

    document = (args.long_text or args.source / "LICENSE").read_text()
    prompts = {
        "chat": chat(CHAT),
        "code": tok.encode(CODE, add_special_tokens=False),
        "long": chat([{"role": "user",
                       "content": document + "\n\nSummarize the document above "
                                  "in five bullet points."}]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in prompts.items():
        (args.out / f"{name}.ids").write_text(",".join(map(str, ids)) + "\n")
        print(f"{name}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
