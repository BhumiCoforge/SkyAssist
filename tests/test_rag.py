import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

import rag
from tests.fakes import ScriptedLLM

DOC = "# Baggage Policy\n\n## Cabin baggage\nOne cabin bag up to 7 kg.\n\n## Checked baggage\n15 kg is included.\n"


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "baggage.md").write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(rag, "DOCS_DIR", d)
    monkeypatch.setattr(rag, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(rag, "_store", None)
    monkeypatch.setattr(rag, "_emb", DeterministicFakeEmbedding(size=32))   # no model download
    monkeypatch.setattr(rag.settings, "rag_min_score", -100.0)              # fake vectors are not semantic
    return d


def test_markdown_is_split_by_heading(docs_dir):
    chunks = rag.split_documents(rag.load_documents())
    sections = {c.metadata.get("section") for c in chunks}
    assert {"Cabin baggage", "Checked baggage"} <= sections
    assert all(c.metadata["source"].endswith("baggage.md") for c in chunks)


def test_retrieval_returns_numbered_context_with_sources(docs_dir):
    assert rag.build_index() >= 2
    context, sources = rag.retrieve_policy("One cabin bag up to 7 kg.")
    assert context.startswith("[1]") and sources[0]["file"] == "baggage.md"
    assert sources[0]["section"] in {"Cabin baggage", "Checked baggage"}


def test_irrelevant_chunks_are_dropped_by_threshold(docs_dir, monkeypatch):
    rag.build_index()
    monkeypatch.setattr(rag.settings, "rag_min_score", 10.0)   # nothing can score this high
    context, sources = rag.retrieve_policy("what is the capital of France")
    assert context == rag.NO_DOCS and sources == []


def test_index_rebuilds_automatically_when_docs_change(docs_dir, monkeypatch):
    rag.build_index()
    calls = []
    real = rag.build_index
    monkeypatch.setattr(rag, "build_index", lambda: calls.append(1) or real())

    monkeypatch.setattr(rag, "_store", None)
    rag.get_store()
    assert calls == []                                           # unchanged docs: loaded from disk

    (docs_dir / "baggage.md").write_text(DOC + "\n## Pets\nPets travel in the cabin.\n", encoding="utf-8")
    monkeypatch.setattr(rag, "_store", None)
    store = rag.get_store()
    assert calls == [1]                                          # changed docs: rebuilt
    assert any("Pets" in d.page_content for d in store.docstore._dict.values())


def test_stale_index_with_other_embedding_model_is_rebuilt(docs_dir, monkeypatch):
    rag.build_index()
    meta = rag.INDEX_DIR / rag.META_FILE
    meta.write_text(json.dumps({"fingerprint": "old"}))
    assert rag._index_is_current() is False


def test_lcel_chain_returns_docs_and_answer(docs_dir):
    rag.build_index()
    llm = ScriptedLLM(script=[AIMessage("Cabin bags can weigh up to 7 kg [1].")])
    out = rag.build_policy_chain(llm=llm).invoke("What is the cabin bag limit?")
    assert out["answer"].startswith("Cabin bags") and out["docs"]
    prompt_sent = llm.seen[0][-1].content
    assert "Question: What is the cabin bag limit?" in prompt_sent and "[1]" in prompt_sent


def test_answer_policy_skips_llm_when_nothing_relevant(docs_dir, monkeypatch):
    rag.build_index()
    monkeypatch.setattr(rag.settings, "rag_min_score", 10.0)
    monkeypatch.setattr(rag, "_llm", ScriptedLLM(script=[]))     # would crash if it were called
    out = rag.answer_policy("who won the cricket world cup?")
    assert "don't have information" in out["answer"] and out["sources"] == []


# ---------------------------------------------------------------- Step 5.2: citations
def test_clean_snippet_removes_headings_and_cuts_at_word_boundary():
    text = "# Baggage Policy\n## Cabin baggage\nOne cabin bag up to 7 kg with dimensions not exceeding 55 x 35 x 25 cm."
    out = rag.clean_snippet(text, limit=40)
    assert "#" not in out and out.endswith("...") and " " in out
    assert rag.clean_snippet("Short text.") == "Short text."


def test_mark_cited_flags_only_sources_used_in_the_answer():
    sources = [{"n": 1}, {"n": 2}, {"n": 3}]
    out = rag.mark_cited("Cabin bags: 7 kg [2]. Checked: 15 kg [1].", sources)
    assert [s["cited"] for s in out] == [True, True, False]


def test_retrieve_docs_attaches_score_without_changing_the_index(docs_dir):
    rag.build_index()
    docs = rag.retrieve_docs("One cabin bag up to 7 kg.")
    assert docs and all("relevance_score" in d.metadata for d in docs)
    stored = rag.get_store().docstore._dict.values()
    assert all("relevance_score" not in d.metadata for d in stored)