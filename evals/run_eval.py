"""Retrieval evaluation for the RAG pipeline (Phase 5).

Run from the project root:   python evals/run_eval.py

No LLM is called, so it is free and fast. It measures:
  1. Ranking quality: is the RIGHT policy section found, and at which position?
  2. The relevance-score threshold: which RAG_MIN_SCORE keeps real questions
     and rejects unrelated ones?
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

import rag  # noqa: E402
from config import settings  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_set.json"
K_RANK = 8                                                # how deep we look when ranking
THRESHOLDS = [round(i * 0.05, 2) for i in range(0, 11)]   # 0.00, 0.05 ... 0.50


def search(question: str) -> list[tuple[str, str, float]]:
    """Best-first list of (file, section, score). No threshold is applied here."""
    pairs = rag.get_store().similarity_search_with_relevance_scores(question, k=K_RANK)
    return [(Path(d.metadata.get("source", "")).name, d.metadata.get("section"), float(s))
            for d, s in pairs]


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def main() -> None:
    items = json.loads(GOLDEN.read_text(encoding="utf-8"))
    top_k = settings.rag_top_k
    print(f"Embedding model : {settings.embedding_model}")
    print(f"Questions       : {len(items)}   (top_k={top_k}, current RAG_MIN_SCORE={settings.rag_min_score})\n")

    # ---------- answerable questions: where does the correct section rank? ----------
    answerable = []
    for it in (i for i in items if i["type"] == "answerable"):
        res = search(it["question"])
        rank = next((n for n, (f, s, _) in enumerate(res, 1)
                     if f == it["file"] and s == it["section"]), None)
        answerable.append((it, res, rank, res[rank - 1][2] if rank else None))

    total = len(answerable)
    hit1 = sum(1 for *_, r, _ in answerable if r == 1)
    hitk = sum(1 for *_, r, _ in answerable if r is not None and r <= top_k)
    mrr = sum(1 / r for *_, r, _ in answerable if r) / total
    print("1) RANKING QUALITY (ignores the threshold)")
    print(f"   Correct section is #1       : {hit1}/{total}  {pct(hit1 / total)}")
    print(f"   Correct section in top {top_k}     : {hitk}/{total}  {pct(hitk / total)}")
    print(f"   MRR (1.0 is perfect)        : {mrr:.3f}\n")

    wrong = [(it, res, rank) for it, res, rank, _ in answerable if rank != 1]
    if wrong:
        print("   Questions where the correct section was NOT first:")
        for it, res, rank in wrong:
            top = f"{res[0][0]} / {res[0][1]} ({res[0][2]:.2f})" if res else "nothing"
            where = f"rank {rank}" if rank else "not found"
            print(f"   - {it['id']} [{where}] {it['question']}")
            print(f"       expected {it['file']} / {it['section']}   got first: {top}")
        print()

    # ---------- off-topic and not-in-docs: what is their best score? ----------
    def best_scores(kind):
        out = []
        for it in (i for i in items if i["type"] == kind):
            res = search(it["question"])
            out.append((it, res[0] if res else None))
        return out

    off = best_scores("off_topic")
    nid = best_scores("not_in_docs")

    def top_score(row):
        return row[1][2] if row[1] else -1.0

    print("2) BEST SCORE OF QUESTIONS THAT SHOULD GET NO ANSWER")
    print("   Off-topic (a good threshold must be ABOVE all of these):")
    for it, best in sorted(off, key=top_score, reverse=True):
        print(f"   {top_score((it, best)):6.2f}  {it['question']}")
    print("   Airline-related but NOT in the documents (retrieval alone cannot reject these):")
    for it, best in sorted(nid, key=top_score, reverse=True):
        label = f"{best[0]} / {best[1]}" if best else "nothing"
        print(f"   {top_score((it, best)):6.2f}  {it['question']}  -> {label}")
    print()

    # ---------- threshold sweep ----------
    print("3) THRESHOLD SWEEP")
    print("   threshold | real questions answered | off-topic rejected | not-in-docs rejected | total")
    table = []
    for t in THRESHOLDS:
        found = sum(1 for *_, r, sc in answerable if r is not None and r <= top_k and sc >= t)
        rej_off = sum(1 for row in off if top_score(row) < t)
        rej_nid = sum(1 for row in nid if top_score(row) < t)
        recall, rej = found / total, rej_off / len(off)
        table.append((t, recall, rej, recall + rej))
        print(f"     {t:4.2f}    |      {pct(recall)}           |      {pct(rej)}        |"
              f"      {pct(rej_nid / len(nid))}       | {recall + rej:.2f}")
    best = max(row[3] for row in table)
    tied = [row[0] for row in table if row[3] == best]
    suggested = tied[len(tied) // 2]
    print(f"\n   Best total = {best:.2f} at thresholds {tied}")
    print(f"   SUGGESTED RAG_MIN_SCORE = {suggested}   (middle of the best range, so it has a safety margin)")
    print("   Note: 'not-in-docs' questions need the LLM to say \"I don't know\"; the threshold alone cannot do it.")


if __name__ == "__main__":
    main()