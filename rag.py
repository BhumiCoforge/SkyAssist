"""Phase 5: PDF/text loading -> chunking -> embeddings -> FAISS -> cited answers."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"

_llm = None
_emb = None
_store = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model

        _llm = init_chat_model(
            os.getenv("LLM_MODEL", "claude-haiku-5-5"),
            model_provider=os.getenv("LLM_PROVIDER", "anthropic"),
            temperature=0,
        )
    return _llm


def get_embeddings():
    global _emb
    if _emb is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _emb


def text_of(content) -> str:
    """LLM content can be a str or a list of blocks; normalise to str."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def build_index() -> int:
    global _store
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    for p in sorted(DOCS_DIR.iterdir()):
        if p.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        elif p.suffix.lower() in (".md", ".txt"):
            docs += TextLoader(str(p), encoding="utf-8").load()
    if not docs:
        raise RuntimeError("No documents found in docs/")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    _store = FAISS.from_documents(chunks, get_embeddings())
    _store.save_local(str(INDEX_DIR))
    return len(chunks)


def get_store():
    global _store
    if _store is None:
        if INDEX_DIR.exists():
            from langchain_community.vectorstores import FAISS

            _store = FAISS.load_local(
                str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
            )
        else:
            build_index()
    return _store


def retrieve_policy(question: str, k: int = 4):
    """Return (numbered context text, sources list) for a policy question."""
    hits = get_store().as_retriever(search_kwargs={"k": k}).invoke(question)
    context = "\n\n".join(f"[{i + 1}] ({Path(h.metadata.get('source', '')).name}) {h.page_content}"
                          for i, h in enumerate(hits))
    sources = [
        {"n": i + 1, "file": Path(h.metadata.get("source", "")).name,
         "page": h.metadata.get("page"), "snippet": h.page_content[:150]}
        for i, h in enumerate(hits)
    ]
    return context, sources


def answer_policy(question: str) -> dict:
    """Standalone RAG answer (used outside the agent)."""
    context, sources = retrieve_policy(question)
    prompt = ("Answer ONLY from the context below. Cite sources like [1]. "
              "If the answer is not in the context, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    return {"answer": text_of(get_llm().invoke(prompt).content), "sources": sources}
