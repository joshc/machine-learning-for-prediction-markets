"""Assemble the source audit and chapter claim fragments into one claim map."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


BOOK = Path(__file__).resolve().parents[1]
FIELDS = (
    "claim_id", "chapter", "claim", "source_ids", "locator",
    "evidence_type", "limitations",
)


def assembled_claims(root: Path = BOOK) -> str:
    sources = json.loads((root / "research" / "sources.json").read_text(encoding="utf-8"))
    known = {source["id"] for source in sources}
    paths = sorted((root / "research" / "claim-fragments").glob("*.csv"))
    if not paths:
        raise ValueError("No audited claim fragments were found.")
    rows: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path.name}: unexpected claim schema.")
            for line, raw in enumerate(reader, start=2):
                if None in raw or any(raw.get(field) is None for field in FIELDS):
                    raise ValueError(f"{path.name}:{line}: malformed CSV record.")
                row = {field: raw[field] for field in FIELDS}
                identifier = row["claim_id"]
                if not identifier or identifier in identifiers:
                    raise ValueError(f"{path.name}:{line}: empty or duplicate claim ID.")
                if not row["chapter"].isdigit() or not 1 <= int(row["chapter"]) <= 30:
                    raise ValueError(f"{identifier}: chapter must be between 1 and 30.")
                for field in ("claim", "locator", "evidence_type", "limitations"):
                    if not row[field].strip():
                        raise ValueError(f"{identifier}: missing {field}.")
                for source in row["source_ids"].split(";"):
                    if source.strip() and source.strip() not in known:
                        raise ValueError(f"{identifier}: unknown source {source.strip()}.")
                identifiers.add(identifier)
                rows.append(row)
    rows.sort(key=lambda row: (int(row["chapter"]), row["claim_id"]))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without modifying the map.")
    args = parser.parse_args()
    content = assembled_claims()
    destination = BOOK / "research" / "claims.csv"
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            raise ValueError("Claim map is stale; run this script without --check after review.")
        print("Claim map matches all audited fragments.")
    else:
        destination.write_bytes(content.encode("utf-8"))
        print(f"Assembled {len(list(csv.DictReader(io.StringIO(content))))} claim records.")


if __name__ == "__main__":
    main()
