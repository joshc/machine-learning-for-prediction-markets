"""Print the actual ordered manuscript inventory for authors and reviewers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict


BOOK = Path(__file__).resolve().parents[1]


class ChapterRecord(TypedDict):
    chapter: int
    title: str
    file: str
    prose_words: int
    citation_keys: list[str]
    examples: list[str]


class ManuscriptInventory(TypedDict):
    title: str
    chapter_count: int
    total_prose_words: int
    chapters: list[ChapterRecord]
    notice: str


def inventory() -> ManuscriptInventory:
    manifest = json.loads((BOOK / "contents.json").read_text(encoding="utf-8"))
    records: list[ChapterRecord] = []
    for chapter in manifest["chapters"]:
        path = BOOK / "chapters" / chapter["file"]
        text = path.read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```|\$\$.*?\$\$", "", text, flags=re.DOTALL)
        words = re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", prose)
        sources = sorted(set(re.findall(r"\]\(\.\./references\.md#([^)]+)\)", text)))
        records.append({
            "chapter": chapter["number"],
            "title": chapter["title"],
            "file": str(path.relative_to(BOOK.parent)),
            "prose_words": len(words),
            "citation_keys": sources,
            "examples": chapter["examples"],
        })
    return {
        "title": manifest["title"],
        "chapter_count": len(records),
        "total_prose_words": sum(record["prose_words"] for record in records),
        "chapters": records,
        "notice": "An inventory proves neither full reading nor correctness of any claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable inventory.")
    args = parser.parse_args()
    result = inventory()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    for chapter in result["chapters"]:
        print(
            f"{chapter['chapter']:02d}  {chapter['prose_words']:5d} words  "
            f"{chapter['file']}"
        )
    print(f"\nTotal: {result['total_prose_words']} prose words in {result['chapter_count']} chapters.")
    print(result["notice"])


if __name__ == "__main__":
    main()
