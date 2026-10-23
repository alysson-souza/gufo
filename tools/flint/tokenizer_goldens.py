#!/usr/bin/env python3
"""Generate Flint tokenizer goldens with HF `tokenizers`.

Each case records the text and the ids HF produces in both of gufo's modes:

* ``special``: ``Tokenizer.encode(text, add_special_tokens=False)``, which
  splits every added token (control and user-defined) out of the text first;
* ``plain``: ``pre_tokenizer.pre_tokenize_str`` followed by
  ``model.tokenize`` on each piece, i.e. pure text with no added-token
  extraction.

``decodes`` records whether HF's ``decode`` of the ``special`` ids reproduces
the text, which holds for every case in this corpus.

    uv run --no-project --with tokenizers==0.23.3 \
        tools/flint/tokenizer_goldens.py \
        --tokenizer /path/to/RecursiveSI-flint/tokenizer.json \
        --output tests/models/flint/fixtures/tokenizer_goldens.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

CPP = """\
#include <vector>

template <typename T>
struct Box {
  std::vector<T> items;
  int Count() const { return static_cast<int>(items.size()); }
};

int main(int argc, char** argv) {
  Box<int> box{{1, 2, 3}};
  if (argc > 1 && argv[1][0] == '-') {
    return box.Count() << 2;  // shift
  }
  return 0;
}
"""

PYTHON = """\
def fib(n: int) -> list[int]:
    \"\"\"Return the first n Fibonacci numbers.\"\"\"
    out = [0, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


if __name__ == "__main__":
\tprint(fib(10), {"k": 'v'}, 3.14e-10, 0x1F)
"""

JSON_TEXT = json.dumps(
    {
        "name": "get_weather",
        "arguments": {"city": "東京", "unit": "°C", "days": 14,
                      "flags": [True, False, None], "ratio": -0.125},
    },
    ensure_ascii=False,
    indent=2,
)

PROSE = (
    "It was the best of times, it was the worst of times; it was the age of "
    "wisdom, it was the age of foolishness... \"Really?\" she asked — and "
    "he didn't answer. (Nobody knew why.) Mr. O'Neil's car cost $12,345.67!"
)

CONTRACTIONS = [
    "I'm sure they'll say we've done what you'd expect, isn't it?",
    "IT'S THEY'LL WE'VE YOU'D I'M ISN'T THEY'RE",
    "It'S tHeY'Ll We'Ve yOu'D i'M iSn'T tHeY'rE",
    "John'S Mary'Ll Bob'RE Ann'VE",
    "it'ſ wiſe'ſ ſo'ſ",  # long s folds to s
    "the K'ſ Kelvin'K'k",
    "'s 't 're 've 'm 'll 'd 'x",
    "dogs' 'quoted' rock'n'roll o'clock y'all",
    "CAN'T can't Can't cAN'T",
    "don’t (curly apostrophe) won't",
    "ÉCOLE'S naïve'll ΣΊΣΥΦΟΣ'S straße'd",
]

CJK = [
    "\u4dff\u4e00\u4e01\u9fa4\u9fa5\u9fa6\u9fa7",
    "a\u4dffb\u4e00c\u9fa5d\u9fa6e",
    "\u303f\u3040\u3041\u309f\u30a0\u30a1\u30ff\u3100\u3105",
    "東京は日本の首都です。",
    "私はカタカナとひらがなと漢字を使います、ラーメンが好き！",
    "ｶﾀｶﾅ半角とＦＵＬＬＷＩＤＴＨ１２３，全角！",
    "𠀀𠀁𪜀 CJK ext B 𠮷野家",
    "한국어 텍스트입니다. 안녕하세요!",
    "中文English混合123数字456中文",
    "我在2024年3月15日去了北京",
    "这是一个测试，包含标点符号：“引号”和《书名号》。",
    "ー〜々〆〇",
]

DIGITS = [
    "1", "12", "123", "1234", "12345", "123456", "1234567",
    "a1b22c333d4444e55555",
    "x1234567y",
    "中1234567文",
    "3.14159 2,718 1e10 0x7FFF -42",
    "version 1.2.3-rc4",
    "²³ ½ Ⅻ ⅻ ٣٤٥ ۱۲۳ ४२ ①②③④ 𝟙𝟚",
    "12²³45",
    "١٢٣٤٥٦٧",
    " 123  456\n789",
    "000000000000000000000000000000",
]

WHITESPACE = [
    "a\r\nb\r\n\r\nc",
    "line1\nline2\n\n\nline3\n",
    "\tindented\n\t\tdouble\n    four spaces",
    "trailing spaces   \nnext",
    "trailing spaces at end   ",
    "   leading spaces",
    "a\u00a0b\u00a0\u00a0c \u00a0d",
    "全角\u3000スペース\u3000\u3000です",
    "lone\rcarriage\rreturns\r",
    "mixed \t \n \r\n \u2028 \u2029 \u0085 end",
    "\u2003em\u2002en\u2009thin\u200bzero-width\u202fnarrow\u205fmath",
    "\n",
    " ",
    "  ",
    "\r\n",
    " \n ",
    "word  \n  word",
    "a \t\tb",
    "\x0b\x0cvt ff",
    "\x00\x01\x02 controls \x7f\x1f",
    "x\x00y\x1fz",
]

EMOJI = [
    "Hello 👋 world 🌍!",
    "👨‍👩‍👧‍👦 family",
    "👍🏽 👍🏿 thumbs",
    "🇮🇪🇧🇷🇯🇵 flags",
    "❤️ ☺️ ✈️",
    "🏳️‍🌈 🧑🏾‍💻",
    "😀😃😄😁😆😅😂🤣",
    "#️⃣ 1️⃣ *️⃣",
]

MARKS = [
    "e\u0301 cafe\u0301 Cafe\u0301",
    "\u0301\u0302 standalone marks",
    " \u0301x",
    "a\u0300\u0301\u0302\u0303b",
    "न्दी हिन्दी भाषा",
    "ภาษาไทย",
    "עִבְרִית",
    "العربية مرحبا",
    "Ǆǅǆ ǈ ǋ titlecase",
    "ʰʲ modifier ʼnot",
    "ABCdef ABC DEFghi aBC",
    "ΑΒΓαβγ Привет МИР",
    ".Hello ,World !x ?Y",
    "@user #tag $var %p &a",
]

SYMBOLS = [
    "a+b=c; x<y>z; p->q; r=>s; t::u",
    "!!!??? ... ---- ====",
    "$100 €200 £300 ¥400 ₹500",
    "→ ← ↑ ↓ ∀∃∈∉ ∑∏ √∞",
    " !\n!!\r\n?\n\n",
    "(( )) [[ ]] {{ }}",
    "a.b.c,d,e;f",
    "©®™ °±×÷",
    "\"quoted\" 'single' `back`",
    "~/path/to/file.txt C:\\Windows\\System32",
    "https://example.com/a?b=1&c=2#frag",
    "<div class=\"x\">text</div>",
    "<thin <think <think/> </ think> <|user| |> <||>",
]

# Oniguruma's Unicode 16.0 tables versus newer ICU: U+0295 became Lo in 17.0,
# and U+11DE0.., U+16EA0.., U+1E6A3 were assigned in 17.0.
UNICODE_VERSION = [
    "Aʕb ʕa aʕ ʕ's",
    "x\U00011de0\U00011de1\U00011de2\U00011de3y",
    "\U00016ea0\U00016ebb abc \U00016ea1's",
    "s\U0001e6a3a \U0001e6a3 \U0001e6a3x",
]

SPECIAL_EDGES = [
    "a<think>b</think>c",
    "<think></think>",
    "<think><think></think></think>",
    "x<|user|>y<|assistant|>z",
    "<|begin_of_text|><|system|>sys<|end_of_text|>",
    "<tool_call>get<arg_key>city</arg_key><arg_value>東京</arg_value></tool_call>",
    "<tools>[]</tools><tool_response>{\"ok\":true}</tool_response>",
    "<|developer|>dev<|observation|>obs",
    "<|reserved_150019|><|reserved_150143|><|reserved_150100|>",
    "<|reserved_150144|> <|reserved_15001|>",
    " <think> spaced </think> ",
    "\n<think>\n\n</think>\n\n",
    "<<think>>",
    "<thin",
    "</tool_call",
    "<tool_call><tool_response></tools>",
    "word<arg_key>key</arg_key>word",
    "123<think>456",
]

LONG = [
    PROSE * 12,
    ("for (int i = 0; i < n; ++i) {\n    sum += a[i] * b[i];\n}\n") * 30,
    "東京タワーは高い。" * 60,
    "a" * 300,
    " " * 200 + "x",
    "\n" * 50,
    "9" * 100,
    "ab" * 200,
    "🙂" * 50,
]


def cases():
    out = []

    def add(category, texts):
        for index, text in enumerate(texts):
            out.append((f"{category}_{index}", text))

    add("cpp", [CPP])
    add("python", [PYTHON])
    add("json", [JSON_TEXT])
    add("prose", [PROSE])
    add("contraction", CONTRACTIONS)
    add("cjk", CJK)
    add("digits", DIGITS)
    add("whitespace", WHITESPACE)
    add("emoji", EMOJI)
    add("marks", MARKS)
    add("symbols", SYMBOLS)
    add("unicode_version", UNICODE_VERSION)
    add("special_edge", SPECIAL_EDGES)
    add("long", LONG)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.tokenizer.read_bytes()
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    added = tok.get_added_tokens_decoder()
    # Every added token, special and user-defined, inline.
    add_all = "".join(added[i].content for i in sorted(added))
    user = "".join(added[i].content for i in sorted(added) if not added[i].special)
    corpus = cases()
    corpus.append(("special_all", add_all))
    corpus.append(("special_all_spaced", " ".join(
        added[i].content for i in sorted(added))))
    corpus.append(("special_user_defined", user))

    records = []
    for name, text in corpus:
        special = tok.encode(text, add_special_tokens=False).ids
        plain = []
        for piece, _ in tok.pre_tokenizer.pre_tokenize_str(text):
            plain.extend(token.id for token in tok.model.tokenize(piece))
        decoded = tok.decode(special, skip_special_tokens=False)
        records.append({
            "name": name,
            "text": text,
            "special": special,
            "plain": plain,
            "decodes": decoded == text,
        })

    def dump(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    # One case per line keeps fixture diffs readable.
    lines = [
        "{",
        f'"tokenizer_json_sha256":{dump(hashlib.sha256(raw).hexdigest())},',
        f'"tokenizers_version":{dump(tokenizers.__version__)},',
        '"cases":[',
        ",\n".join(dump(record) for record in records),
        "]}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(records)} cases -> {args.output}")


if __name__ == "__main__":
    main()
