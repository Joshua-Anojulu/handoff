"""Text handling shared by the extractor and the readability scorer.

Both have to agree on what counts as part of the note. The notes in `data/`
carry a `# note: <id>` header so fixtures and gold labels can be matched up,
and an id like `post-op-chole` is a rare word that would otherwise be counted
as unexplained clinical jargon, making every note look three points harder than
it is.
"""
from __future__ import annotations

import re

_METADATA = re.compile(r"^\s*#.*$", re.M)


def strip_metadata(text: str) -> str:
    return _METADATA.sub("", text)


def note_id(text: str) -> str:
    match = re.search(r"^\s*#\s*note:\s*(\S+)", text, re.M)
    return match.group(1) if match else ""
