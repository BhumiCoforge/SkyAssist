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