"""Structural checks for the manuscript, not a substitute for editorial review."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


BOOK = Path(__file__).resolve().parents[1]
ROOT = BOOK.parent
COMPANION = BOOK / "companion"
CLAIM_FIELDS = {
    "claim_id",
    "chapter",
    "claim",
    "source_ids",
    "locator",
    "evidence_type",
    "limitations",
}


def load_book_tool(name: str) -> ModuleType:
    """Load the standalone book utility without installing a root-level package."""
    spec = importlib.util.spec_from_file_location(f"book_{name}", BOOK / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the book utility {name}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown_files(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for parent, children, files in os.walk(directory):
        children[:] = [
            name for name in children
            if not name.startswith(".") and not name.endswith(".egg-info")
            and name not in {"__pycache__", "node_modules", "build", "dist"}
        ]
        paths.extend(Path(parent) / name for name in files if name.endswith(".md"))
    return sorted(paths)


def prose_words(text: str) -> list[str]:
    without_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    without_display_math = re.sub(r"\$\$.*?\$\$", "", without_code, flags=re.DOTALL)
    return re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", without_display_math)


def heading_anchors(text: str) -> set[str]:
    counts: dict[str, int] = {}
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE):
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", plain.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    anchors.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    return anchors


class ClaimAssemblerTests(unittest.TestCase):
    def test_fragments_sort_and_reject_invalid_records(self) -> None:
        tool = load_book_tool("assemble_claims")
        fields, assemble = tool.FIELDS, tool.assembled_claims

        with tempfile.TemporaryDirectory(prefix="prediction-book-claims-") as temporary:
            root = Path(temporary)
            fragments = root / "research" / "claim-fragments"
            fragments.mkdir(parents=True)
            (root / "research" / "sources.json").write_text(
                json.dumps([{"id": "SOURCE"}]), encoding="utf-8",
            )
            path = fragments / "fixture.csv"

            def write_rows(rows: list[dict[str, str]]) -> None:
                content = io.StringIO(newline="")
                writer = csv.DictWriter(content, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                path.write_text(content.getvalue(), encoding="utf-8")

            row = {
                "claim_id": "C2", "chapter": "2", "claim": "A scoped claim.",
                "source_ids": "SOURCE", "locator": "Abstract",
                "evidence_type": "abstract", "limitations": "Not an empirical result.",
            }
            earlier = {**row, "claim_id": "C1", "chapter": "1"}
            write_rows([row, earlier])
            result = list(csv.DictReader(io.StringIO(assemble(root))))
            self.assertEqual([item["claim_id"] for item in result], ["C1", "C2"])
            for invalid in (
                [row, row],
                [{**row, "source_ids": "UNKNOWN"}],
                [{**row, "chapter": "31"}],
                [{**row, "claim_id": ""}],
                [{**row, "limitations": ""}],
                [{**row, "locator": ""}],
            ):
                with self.subTest(invalid=invalid):
                    write_rows(invalid)
                    with self.assertRaises(ValueError):
                        assemble(root)
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                assemble(root)


class BookLayoutTests(unittest.TestCase):
    def test_companion_has_its_own_project_and_assets(self) -> None:
        for relative in (
            Path("pyproject.toml"),
            Path("requirements.lock"),
            Path("src") / "prediction_market_lab" / "__init__.py",
            Path("examples") / "payoffs.py",
            Path("data") / "fixtures" / "recurring_events.csv",
            Path("experiments") / "noaa_study.py",
            Path("tests") / "test_noaa_study.py",
        ):
            with self.subTest(path=relative):
                self.assertTrue((COMPANION / relative).is_file())
        tool = load_book_tool("manuscript_inventory")
        with tempfile.TemporaryDirectory(prefix="prediction-book-inventory-") as temporary:
            book = Path(temporary) / "book"
            (book / "chapters").mkdir(parents=True)
            (book / "contents.json").write_text(json.dumps({
                "title": "Fixture",
                "chapters": [{"number": 1, "title": "Sample", "file": "sample.md", "examples": []}],
            }), encoding="utf-8")
            (book / "chapters" / "sample.md").write_text("# A chapter\n\nReadable prose.\n", encoding="utf-8")
            with patch.object(tool, "BOOK", book):
                result = tool.inventory()
            self.assertEqual(result["chapter_count"], 1)
            self.assertEqual(result["total_prose_words"], 4)
            self.assertEqual(result["chapters"][0]["file"], str(Path("book") / "chapters" / "sample.md"))

    def test_markdown_inventory_excludes_installed_and_generated_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prediction-book-layout-") as temporary:
            directory = Path(temporary)
            included = directory / "companion" / "README.md"
            excluded = [
                directory / "companion" / ".venv" / "README.md",
                directory / "companion" / "src" / "lab.egg-info" / "README.md",
                directory / "companion" / "__pycache__" / "README.md",
                directory / "companion" / "node_modules" / "README.md",
            ]
            for path in [included, *excluded]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Fixture\n", encoding="utf-8")
            self.assertEqual(markdown_files(directory), [included])


class BookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((BOOK / "contents.json").read_text(encoding="utf-8"))
        cls.chapters = cls.manifest["chapters"]

    def test_reading_order_and_companion_files(self) -> None:
        self.assertEqual([chapter["number"] for chapter in self.chapters], list(range(1, 31)))
        filenames = [chapter["file"] for chapter in self.chapters]
        self.assertEqual(len(set(filenames)), 30)
        for chapter in self.chapters:
            with self.subTest(chapter=chapter["number"]):
                self.assertTrue((BOOK / "chapters" / chapter["file"]).is_file())
                for example in chapter["examples"]:
                    self.assertTrue((COMPANION / "examples" / example).is_file(), example)
        self.assertEqual(self.chapters[24]["part"], 5)
        self.assertEqual([chapter["part"] for chapter in self.chapters[-5:]], [6] * 5)

    def test_full_manuscript_not_chapter_stubs(self) -> None:
        total = 0
        for chapter in self.chapters:
            text = (BOOK / "chapters" / chapter["file"]).read_text(encoding="utf-8")
            count = len(prose_words(text))
            total += count
            with self.subTest(chapter=chapter["number"]):
                self.assertGreaterEqual(count, 1600, "A full chapter needs developed exposition.")
                self.assertRegex(text.lower(), r"prerequisite")
                self.assertRegex(text.lower(), r"learning objectives")
                self.assertRegex(text.lower(), r"exercise")
                self.assertRegex(text.lower(), r"solution")
                self.assertNotRegex(text, r"(?mi)^(?:TODO|TBD|PLACEHOLDER)\s*[:\-]")
        self.assertGreaterEqual(total, 60000, "The approved scope is a comprehensive book.")

    def test_bibliography_source_records(self) -> None:
        sources = json.loads((BOOK / "research" / "sources.json").read_text(encoding="utf-8"))
        self.assertIsInstance(sources, list)
        ids = [source["id"] for source in sources]
        self.assertEqual(len(ids), len(set(ids)))
        anchors = heading_anchors((BOOK / "references.md").read_text(encoding="utf-8"))
        required = {
            "id", "title", "authors", "year", "kind", "status", "url",
            "doi", "accessed", "verification", "supports", "limitations",
        }
        for source in sources:
            with self.subTest(source=source["id"]):
                self.assertTrue(required <= source.keys())
                self.assertTrue(source["title"])
                self.assertTrue(source["authors"])
                self.assertTrue(source["verification"])
                self.assertTrue(source["supports"])
                self.assertTrue(source["limitations"])
                self.assertIn(source["id"].lower(), anchors)
                self.assertIn(urlsplit(source["url"]).scheme, {"http", "https"})

    def test_claim_records_reference_known_sources(self) -> None:
        sources = json.loads((BOOK / "research" / "sources.json").read_text(encoding="utf-8"))
        known = {source["id"] for source in sources}
        aggregate = BOOK / "research" / "claims.csv"
        self.assertTrue(aggregate.is_file())
        fragments = sorted((BOOK / "research" / "claim-fragments").glob("*.csv"))
        for path in [aggregate, *fragments]:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertTrue(CLAIM_FIELDS <= set(reader.fieldnames or []), str(path))
                seen: set[str] = set()
                for row in reader:
                    with self.subTest(file=path.name, claim=row["claim_id"]):
                        self.assertTrue(row["claim_id"])
                        self.assertNotIn(row["claim_id"], seen)
                        seen.add(row["claim_id"])
                        self.assertTrue(row["claim"])
                        self.assertTrue(row["evidence_type"])
                        self.assertTrue(row["limitations"])
                        for source in row["source_ids"].split(";"):
                            if source.strip():
                                self.assertIn(source.strip(), known)

    def test_aggregate_claim_map_is_current(self) -> None:
        tool = load_book_tool("assemble_claims")

        self.assertEqual(
            (BOOK / "research" / "claims.csv").read_text(encoding="utf-8"),
            tool.assembled_claims(BOOK),
        )

    def test_local_markdown_links(self) -> None:
        for path in markdown_files(BOOK):
            text = path.read_text(encoding="utf-8")
            without_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            without_code = re.sub(r"`+[^`\n]*`+", "", without_code)
            for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", without_code):
                target = match.group(1).strip().split(' "', 1)[0].strip("<>")
                parts = urlsplit(target)
                if parts.scheme or parts.netloc:
                    continue
                relative = unquote(parts.path)
                destination = (path.parent / relative).resolve() if relative else path
                with self.subTest(file=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(destination.is_relative_to(ROOT), "Link escapes repository.")
                    self.assertTrue(
                        destination.is_relative_to(BOOK) or destination == ROOT / "README.md",
                        "Book assets must stay inside book; only the repository README backlink is external.",
                    )
                    self.assertTrue(destination.exists(), "Missing local link target.")
                    if parts.fragment and destination.suffix == ".md" and destination.is_file():
                        anchors = heading_anchors(destination.read_text(encoding="utf-8"))
                        self.assertIn(unquote(parts.fragment), anchors, "Missing heading anchor.")


if __name__ == "__main__":
    unittest.main()
