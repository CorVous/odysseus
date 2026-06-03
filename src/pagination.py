"""
pagination.py

Shared line-based pagination for read-style tools (read_file, manage_documents
read). Centralising it keeps the two tools identical: same 1-based `offset`
(start line) and `limit` (max lines) semantics, same char safety net, so a model
that learns to page one of them pages the other the same way.

Pure stdlib / no project imports — safe to import from anywhere.
"""

from typing import Dict

# A single read call returns at most this many lines (when the model doesn't
# ask for a specific `limit`) ...
DEFAULT_READ_LINES = 500
# ... and never more than this many characters, regardless of line count, so one
# pathologically long line can't blow up the context. The cut lands on a line
# boundary (except a single over-long line, which is hard-cut to make progress).
MAX_READ_CHARS = 20_000


def paginate_lines(
    text: str,
    offset: int = 0,
    limit: int = 0,
    *,
    max_chars: int = MAX_READ_CHARS,
    default_lines: int = DEFAULT_READ_LINES,
) -> Dict:
    """Return a line-based page of *text*.

    offset: 1-based start line (<=0 means line 1).
    limit:  max lines to return (<=0 means `default_lines`).

    Returns a dict the caller turns into a tool result:
      data:     the page text (no footer — the caller appends its own cursor)
      start:    1-based first line returned
      last:     1-based last line returned (== start-1 when nothing is returned)
      total:    total line count of the document
      has_more: True if lines remain after `last`
      past_eof: True if `offset` is beyond the document
    """
    lines = text.splitlines(keepends=True)
    total = len(lines)
    start = max(1, offset)

    if total > 0 and start > total:
        return {"data": "", "start": start, "last": start - 1, "total": total,
                "has_more": False, "past_eof": True}

    page = limit if limit > 0 else default_lines
    window = lines[start - 1:start - 1 + page]

    out, chars = [], 0
    for ln in window:
        if out and chars + len(ln) > max_chars:
            break
        if not out and len(ln) > max_chars:
            ln = ln[:max_chars]  # single over-long line: hard-cut so we progress
        out.append(ln)
        chars += len(ln)

    last = start - 1 + len(out)
    return {"data": "".join(out), "start": start, "last": last, "total": total,
            "has_more": last < total, "past_eof": False}
