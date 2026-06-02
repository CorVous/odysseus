"""manage_documents (read/list) must travel with the document WRITE tools.

RAG retrieval + the top-k tool budget routinely surface create/edit/update/
suggest_document but evict manage_documents, so the model could edit a doc but
had no tool that returns its content — "show me / what's in the document" then
failed silently. `get_tools_for_query` now force-includes the reader whenever any
document writer is in the selected set, and a few generic "view the contents"
phrasings force-include it via keyword hint.

`retrieve` (which needs a chroma collection) is stubbed so these tests exercise
only the selection logic, not embeddings.
"""
from src.tool_index import ToolIndex, _DOC_WRITE_TOOLS


def _index(retrieved=None):
    ti = ToolIndex.__new__(ToolIndex)
    ti.retrieve = lambda query, k=8: list(retrieved or [])
    return ti


def test_reader_paired_when_writer_retrieved():
    # Reproduces the bug: RAG surfaced the writers but not the reader.
    for writer in _DOC_WRITE_TOOLS:
        ti = _index(retrieved=[writer])
        tools = ti.get_tools_for_query("unrelated phrasing with no doc keywords")
        assert "manage_documents" in tools, writer


def test_no_doc_tools_no_forced_reader():
    # A query with neither doc writers retrieved nor doc keywords must NOT
    # pull in manage_documents (don't bloat every turn).
    ti = _index(retrieved=["web_search"])
    tools = ti.get_tools_for_query("what is the capital of france")
    assert "manage_documents" not in tools


def test_generic_view_phrasings_force_reader():
    ti = _index(retrieved=[])  # isolate the keyword loop
    for q in (
        "view the project outline",
        "what's in the document",
        "show me the contents of the report",
        "what does it say",
    ):
        assert "manage_documents" in ti.get_tools_for_query(q), q


def test_unrelated_words_do_not_force_reader():
    # Word-boundary safety: these contain no genuine doc-read intent.
    ti = _index(retrieved=[])
    for q in ("review the documentation site", "the outliner app crashed"):
        # "documentation"/"outliner" must not trip "the document"/"the outline".
        assert "manage_documents" not in ti.get_tools_for_query(q), q
