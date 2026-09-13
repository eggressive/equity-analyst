"""Document invariants: one changelog, resolvable links, no em dashes.

Run directly, like the other suites:

    python tests/test_docs.py

No pytest dependency and no third-party imports, so it runs before anything is
installed. It guards the mechanical rules in AGENTS.md that a reviewer would
otherwise have to check by hand:

  - no em dash (U+2014) in any document or module,
  - every relative link resolves, anchors included,
  - CHANGELOG.md keeps an Unreleased section and dated version headings,
  - a document that names a `runs/` file names one that exists,
  - a recorded test count matches the suite it describes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS = ["README.md", "AGENTS.md", "TODO.md", "CHANGELOG.md"]
MODULES = [
    "analyze.py",
    "src/agents.py",
    "src/data.py",
    "src/metrics.py",
    "src/rubric.py",
    "src/verify.py",
    "tests/test_docs.py",
    "tests/test_metrics.py",
    "tests/test_rubric.py",
]
EM_DASH = "\u2014"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.S)
CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
UNRELEASED_RE = re.compile(r"^## \[Unreleased\]", re.M)
VERSION_RE = re.compile(r"^## \[([^\]]+)\]", re.M)
RUN_REF_RE = re.compile(r"`(runs/[A-Za-z0-9_./-]+)`")
COUNT_RE = re.compile(r"`tests/(test_\w+\.py)`[^.\n]*?(\d+)/\2")
DATE_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+ - \d{4}-\d{2}-\d{2}")


def _prose(path: Path) -> str:
    """Only the text a renderer would linkify: fenced blocks and code spans removed.

    A syntax example such as `[Part 5](path/to/file.md#anchor)` documents the form, it
    does not link anything, so it must not be checked as a link.
    """
    text = FENCE_RE.sub(" ", path.read_text())
    return CODE_SPAN_RE.sub(" ", text)


def _slug(title: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    title = title.strip().lower().replace("`", "")
    return re.sub(r"[^a-z0-9 \-_]", "", title).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Every anchor a heading in this file produces, duplicates numbered as GitHub does."""
    out: set[str] = set()
    seen: dict[str, int] = {}
    for title in HEADING_RE.findall(path.read_text()):
        slug = _slug(title)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.add(slug if not count else f"{slug}-{count}")
    return out


def test_no_em_dash_in_documents_or_modules():
    for name in DOCUMENTS + MODULES:
        text = (ROOT / name).read_text()
        assert EM_DASH not in text, f"{name} contains U+2014"


def test_relative_links_resolve_including_anchors():
    for name in DOCUMENTS:
        path = ROOT / name
        for target in LINK_RE.findall(_prose(path)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            target_path = (path.parent / file_part).resolve() if file_part else path
            assert target_path.exists(), f"{name} links {target}, which does not exist"
            if anchor:
                assert anchor in _anchors(target_path), (
                    f"{name} links {target}, which has no such heading")


def test_changelog_keeps_an_unreleased_section():
    assert UNRELEASED_RE.search((ROOT / "CHANGELOG.md").read_text()), (
        "CHANGELOG.md has no '## [Unreleased]' section")


def test_changelog_version_headings_carry_a_date():
    for heading in VERSION_RE.findall((ROOT / "CHANGELOG.md").read_text()):
        if heading == "Unreleased":
            continue
        assert DATE_RE.fullmatch(heading), (
            f"version heading '{heading}' is not '<major>.<minor>.<patch> - YYYY-MM-DD'")


def test_named_run_artifacts_exist():
    for name in ("CHANGELOG.md", "TODO.md"):
        text = (ROOT / name).read_text()
        for ref in RUN_REF_RE.findall(text):
            assert (ROOT / ref).exists(), f"{name} names {ref}, which does not exist"


def test_recorded_test_counts_match_the_suites():
    for name in ("CHANGELOG.md", "TODO.md"):
        text = (ROOT / name).read_text()
        for suite, recorded in COUNT_RE.findall(text):
            defined = len(re.findall(r"^def test_", (ROOT / "tests" / suite).read_text(), re.M))
            assert int(recorded) == defined, (
                f"{name} records {suite} at {recorded}/{recorded}, the suite defines {defined}")


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    raise SystemExit(1 if failed else 0)
