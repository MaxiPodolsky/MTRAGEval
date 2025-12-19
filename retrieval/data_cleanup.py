import json
import re
import sys
from typing import List

# ============ CONFIG ============
INPUT_PATH = "../corpora/passage_level/govt.jsonl/govt.jsonl"
OUTPUT_PATH = "../corpora/passage_level/govt.jsonl/govt_clean.jsonl"
TEXT_FIELD = "text"

# Common boilerplate patterns to remove whole lines of site chrome
BOILERPLATE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.I)
    for p in [
        r"skip to main content",
        r"all rights reserved",
        r"cookie",
        r"privacy policy",
        r"subscribe",
        r"sign up for email",
        r"follow us on",
        r"terms of use",
        r"javascript enabled",
    ]
]


def clean_text(text: str) -> str:
    """Clean obvious noise but keep the document content."""

    # 1) Remove control chars / null bytes (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)

    # 2) Remove <script> and <style> blocks
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)

    # 3) Remove HTML comments
    text = re.sub(r"(?s)<!--.*?-->", " ", text)

    # 4) Remove all other HTML tags
    text = re.sub(r"(?s)<[^>]+>", " ", text)

    # 5) Remove very long digit-only sequences (garbage like 2020202020..., 1234567891011...)
    text = re.sub(r"\d{20,}", " ", text)

    # 6) Remove mega-words (1000+ non-space chars = likely corrupted)
    text = re.sub(r"\S{1000,}", " ", text)

    # 7) Remove boilerplate lines (optional, but usually pure noise)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            # keep blank lines for paragraph structure
            lines.append("")
            continue
        if any(p.search(s) for p in BOILERPLATE_PATTERNS):
            # drop that line entirely
            continue
        lines.append(s)

    text = "\n".join(lines)

    # 8) Whitespace normalization
    # collapse runs of spaces/tabs
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    # collapse long runs of newlines: \n\n\n\n -> \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    in_path = INPUT_PATH if len(sys.argv) < 2 else sys.argv[1]
    out_path = OUTPUT_PATH if len(sys.argv) < 3 else sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            # Preserve completely empty lines as-is
            if line.strip() == "":
                fout.write(line)
                continue

            # Try JSON parse; if it fails, write line unchanged
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            # If TEXT_FIELD is missing or not a string, just write unchanged
            text = obj.get(TEXT_FIELD, None)
            if not isinstance(text, str):
                fout.write(line)
                continue

            cleaned = clean_text(text)
            obj[TEXT_FIELD] = cleaned

            # Re-dump JSON; structure and fields preserved (whitespace may differ)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
