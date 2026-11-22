"""Retrieval (RAG): load documents -> split by heading -> embed -> FAISS -> retriever -> LCEL chain."""
import hashlib
import json
import logging
import re
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from config import settings

log = logging.getLogger("skyassist.rag")

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"
META_FILE = "meta.json"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100

_llm = None
_emb = None
_store = None


# ---------------------------------------------------------------- models
def get_llm():
    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model

        kwargs = dict(
            temperature=0,                         # deterministic: best for support answers
            max_tokens=settings.llm_max_tokens,
            max_retries=2,
            timeout=60,
        )
        if settings.llm_base_url:                  # company gateway (OpenAI-style API)
            kwargs["base_url"] = settings.llm_base_url
            kwargs["api_key"] = settings.llm_api_key
            kwargs["default_headers"] = {settings.llm_api_key_header: settings.llm_api_key}
        _llm = init_chat_model(
            settings.llm_model,
            model_provider=settings.llm_provider,
            **kwargs,
        )
    return _llm


def get_embeddings():
    global _emb
    if _emb is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _emb = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},   # needed for the 0-1 relevance scores
        )
    return _emb


def text_of(content) -> str:
    """LLM content can be a str or a list of blocks; normalise to str."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


# ---------------------------------------------------------------- loading + chunking
def load_documents() -> list[Document]:
    """Load every PDF / .md / .txt in docs/. Markdown is split by heading so one chunk = one policy
    section, and the heading is stored in the metadata (used for citations)."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False)
    docs: list[Document] = []
    for p in sorted(DOCS_DIR.iterdir()):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        elif suffix == ".md":
            text = p.read_text(encoding="utf-8")
            for d in header_splitter.split_text(text):
                d.metadata["source"] = str(p)
                docs.append(d)
        elif suffix == ".txt":
            docs += TextLoader(str(p), encoding="utf-8").load()
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Only sections longer than CHUNK_SIZE get cut further; short sections stay whole.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)


def _fingerprint() -> str:
    """Changes whenever a document, the embedding model or the chunk settings change."""
    parts = [settings.embedding_model, str(CHUNK_SIZE), str(CHUNK_OVERLAP)]
    for p in sorted(DOCS_DIR.iterdir()):
        if p.is_file():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------- vector store
def build_index() -> int:
    global _store
    from langchain_community.vectorstores import FAISS

    docs = load_documents()
    if not docs:
        raise RuntimeError("No documents found in docs/")
    chunks = split_documents(docs)
    _store = FAISS.from_documents(chunks, get_embeddings())
    INDEX_DIR.mkdir(exist_ok=True)
    _store.save_local(str(INDEX_DIR))
    (INDEX_DIR / META_FILE).write_text(json.dumps({"fingerprint": _fingerprint()}))
    log.info("indexed %s chunks from %s documents", len(chunks), len(docs))
    return len(chunks)


def _index_is_current() -> bool:
    meta = INDEX_DIR / META_FILE
    try:
        return meta.exists() and json.loads(meta.read_text()).get("fingerprint") == _fingerprint()
    except (OSError, ValueError):
        return False


def get_store():
    """Load the saved index, rebuilding it automatically if docs/ changed since it was built."""
    global _store
    if _store is None:
        if INDEX_DIR.exists() and _index_is_current():
            from langchain_community.vectorstores import FAISS

            # Safe here: we only load an index file this app wrote itself.
            _store = FAISS.load_local(str(INDEX_DIR), get_embeddings(),
                                      allow_dangerous_deserialization=True)
        else:
            build_index()
    return _store


def get_retriever(k: int | None = None, min_score: float | None = None):
    """LangChain retriever that drops chunks below the relevance threshold, so an unrelated
    question returns nothing (and the assistant says 'I don't know') instead of 4 random chunks."""
    return get_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k or settings.rag_top_k,
                       "score_threshold": settings.rag_min_score if min_score is None else min_score},
    )


def retrieve_docs(question: str, k: int | None = None, min_score: float | None = None) -> list[Document]:
    """Search the index and keep only chunks scoring at least the threshold. Every returned chunk
    carries its score in metadata['relevance_score'] (a copy, so the stored index is not changed)."""
    threshold = settings.rag_min_score if min_score is None else min_score
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
        pairs = get_store().similarity_search_with_relevance_scores(question, k=k or settings.rag_top_k)
    return [Document(page_content=d.page_content, metadata={**d.metadata, "relevance_score": float(s)})
            for d, s in pairs if s >= threshold]


# ---------------------------------------------------------------- retrieval helpers
CITATION = re.compile(r"\[(\d+)\]")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({Path(d.metadata.get('source', '')).name} / {d.metadata.get('section', 'general')}) "
        f"{d.page_content}" for i, d in enumerate(docs))


def clean_snippet(text: str, limit: int = 220) -> str:
    """Readable preview for the Sources box: no markdown headings, one line, cut at a word boundary."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    flat = " ".join(" ".join(lines).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "..."


def sources_of(docs: list[Document]) -> list[dict]:
    """One entry per retrieved chunk, numbered like the [n] markers the model writes in its answer."""
    out = []
    for i, d in enumerate(docs):
        score = d.metadata.get("relevance_score")
        out.append({"n": i + 1,
                    "file": Path(d.metadata.get("source", "")).name,
                    "title": d.metadata.get("title"),
                    "section": d.metadata.get("section"),
                    "page": d.metadata.get("page"),
                    "score": round(float(score), 2) if score is not None else None,
                    "snippet": clean_snippet(d.page_content)})
    return out


def mark_cited(answer: str, sources: list[dict]) -> list[dict]:
    """Add cited=True/False to every source, depending on whether the answer contains its [n]."""
    used = {int(n) for n in CITATION.findall(answer)}
    return [{**s, "cited": s["n"] in used} for s in sources]


NO_DOCS = "No relevant policy text was found for this question."


def retrieve_policy(question: str, k: int | None = None):
    """Return (numbered context text, sources list) for a policy question."""
    docs = retrieve_docs(question, k)
    if not docs:
        return NO_DOCS, []
    return format_docs(docs), sources_of(docs)


# ---------------------------------------------------------------- LCEL chain
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer airline policy questions. Use ONLY the numbered context.\n"
               "- Put the source number in square brackets right after each fact it supports, "
               "like: Cabin bags can weigh up to 7 kg [1].\n"
               "- Cite only the sources you actually used.\n"
               "- Give just the answer. Do not add closing lines such as 'feel free to ask'.\n"
               "- If the context does not contain the answer, say you don't know and do not cite."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def build_answer_chain(llm=None):
    """{docs, question} -> prompt -> LLM -> text."""
    return (RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
            | RAG_PROMPT | (llm or get_llm()) | StrOutputParser())


def build_policy_chain(llm=None, retriever=None):
    """Full LCEL pipeline: retriever -> prompt -> LLM -> text. Returns {docs, question, answer}."""
    return (RunnableParallel(docs=retriever or get_retriever(), question=RunnablePassthrough())
            | RunnablePassthrough.assign(answer=build_answer_chain(llm)))


def answer_policy(question: str) -> dict:
    """Standalone cited answer (used by POST /rag/ask)."""
    docs = retrieve_docs(question)
    if not docs:                                            # nothing relevant: skip the LLM call
        return {"answer": "I don't have information about that in the airline policy documents.",
                "sources": []}
    answer = build_answer_chain().invoke({"docs": docs, "question": question})
    return {"answer": answer, "sources": mark_cited(answer, sources_of(docs))}