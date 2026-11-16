"""BM25 keyword index over chunks — the sparse half of hybrid retrieval.

BM25 (Best Match 25): ranks documents by how often query terms appear,
discounted by how common each term is across the corpus (a word in every
abstract carries no signal). Hand-rolled in ~30 lines — the formula is
standard and a dependency would add nothing.

Pure functions: chunks in → serializable index dict out; index + query →
ranked pmids out.
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import cast

K1 = 1.2  # term-frequency saturation
B = 0.75  # length normalization

_token_re = re.compile(r"[a-z0-9]+")

# Question words are corpus-rare without being entity signals — "how",
# "does", "work" appear in fewer chunks than a niche substance name, so rarity
# alone can't tell them from real entities. English + Spanish (medical
# questions arrive in both).
_STOP_WORDS = (
    "a an the and or but if of in on at to by for with about into than then "
    "there here is are was were be been being do does did done have has had "
    "can could would should will shall may might must not no yes so up down "
    "out off over under after before between through during while what which "
    "who whom how when where why i you he she it we they me him her us them "
    "my your his its our their this that these those some any each other "
    "another others such only just very much many more most all both few "
    "still even really quite rather pretty almost nearly yet already again "
    "once twice else instead though although either neither whether maybe "
    "perhaps probably actually basically literally seriously honestly "
    "frankly clearly obviously certainly definitely sure okay well hey hi "
    "hello thanks thank please ever never always often sometimes usually "
    "every everyone everything everywhere somebody someone something "
    "somewhere anybody anyone anything anywhere nobody nothing nowhere "
    "everybody use used using work worked working works build built "
    "building make made show shows think know want say said tell told ask "
    "asked get got like look see new now way big small long old first last "
    "best worst part back thing things people year years time times day days "
    "kind kinds type types go goes went gone come came take took taken give "
    "gave given put keep kept try tried trying seem seems seemed become "
    "became feel felt mean meant happen happens happened need needs needed "
    "help helps call called find found heard hear read write wrote play "
    "run ran move moved start started stop stopped end ended open close "
    "change changed "
    "que como cuanto cuanta cuantos cuantas donde cuando quien quienes "
    "cual cuales para con por los las una uno unos unas del este esta "
    "estos estas ese esa esos esas son es fue fueron han ha tiene tienen "
    "tienes tengo sobre entre desde hasta mas muy ya asi tambien tampoco "
    "pero porque si no al lo le les se su sus mi mis tu tus el la de en y "
    "o un me te nos hace hizo hicieron hacen ano anos vez veces cosa cosas "
    "parte forma manera tipo tipos ser estar estan hay puede pueden "
    "podria deberia siempre nunca jamas aun todavia cada todo "
    "toda todos todas algo alguien nadie nada ningun ninguna mismo misma "
    "mismos otro otra otros otras mucho mucha muchos muchas poco "
    "poca pocos pocas antes despues mientras durante segun quiza quizas "
    "tal acaso ahora entonces luego ademas incluso bien mal mejor peor "
    "bueno buena buenos buenas gran grande grandes nuevo nueva nuevos "
    "nuevas propio propia general ejemplo casi apenas solo solamente "
    "unicamente exactamente realmente actualmente normalmente "
    "generalmente posiblemente probablemente seguramente ciertamente "
    "definitivamente obviamente claramente basicamente literalmente "
    "seriamente honestamente francamente querer quiere quiero queria "
    "gustar gusta gustaria parecer parece ir va voy van venir viene dar "
    "da dio poner pone decir dice dijo ver ve vio saber sabe conocer "
    "conoce llamar llama llamado buscar busca encontrar encuentra usar "
    "usa usado"
)
STOP = frozenset(_STOP_WORDS.split())


def tokenize(text: str) -> list[str]:
    """NFKD + accent-strip + lowercase alphanumeric tokens ('ácido' →
    'acido'). No stopword removal — BM25's IDF already downweights
    ubiquitous words."""
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return _token_re.findall(t)


def dense_query(question: str) -> str:
    """Embedding input: interrogative scaffolding drags the query vector
    away from document space (chunks are declarative text), so the dense
    leg embeds a keyword-only form — 'what does magnesium do for sleep' →
    'information about magnesium sleep'. All-STOP queries fall back to the
    full token list (an empty keyword form still embeds to *something* — a
    phantom vector worse than the noisy one)."""
    kw = " ".join(t for t in tokenize(question) if t not in STOP)
    return f"information about {kw or ' '.join(tokenize(question))}"


def entity_tokens(index: dict, query: str) -> list[str]:
    """Query tokens naming a candidate substance/entity — corpus-rare
    (df <= ~0.5%) or entirely absent. Same premise as rare_hits, widened
    to df=0: a token nowhere in the corpus is still the entity the
    question asks about — and the one no evidence can cover."""
    rare_max = max(2, len(index["docs"]) // 200)
    return [
        t
        for t in tokenize(query)
        if len(t) >= 3 and t not in STOP and index["df"].get(t, 0) <= rare_max
    ]


def rare_hits(index: dict, query: str, max_hits: int = 2) -> list[str]:
    """Entity-signal coverage: a query token that appears in only a few
    docs names a substance/entity ('laxogenin', 'berberine') that semantic
    similarity blurs into phrasing — chunks containing it must reach the
    pool regardless of rank. Rare = df ≤ ~0.5% of the corpus (assumption:
    a named entity in a 17.7k-doc corpus shows in tens, not thousands, of
    abstracts; tunable)."""
    rare_max = max(2, len(index["docs"]) // 200)
    rare = [
        t
        for t in tokenize(query)
        if len(t) >= 3 and t not in STOP and 1 <= index["df"].get(t, 0) <= rare_max
    ]
    if not rare:
        return []
    hits = [cid for cid, d in index["docs"].items() if any(t in d["tf"] for t in rare)]
    rank = {cid: s for cid, s in score(index, query, top_k=len(hits) or 1)}
    return sorted(hits, key=lambda c: -rank.get(c, 0.0))[:max_hits]


def build_index(chunks: list[dict]) -> dict:
    """chunks → {docs, df, avgdl} index dict (JSON-serializable)."""
    docs = {}
    df = Counter()
    for c in chunks:
        toks = tokenize(f"{c['title']} {c['text']}")
        counts = Counter(toks)
        docs[c["chunk_id"]] = {"tf": dict(counts), "len": len(toks)}
        df.update(counts.keys())
    avgdl = sum(cast(int, d["len"]) for d in docs.values()) / max(len(docs), 1)
    return {"docs": docs, "df": dict(df), "avgdl": avgdl, "k1": K1, "b": B}


def score(index: dict, query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """query → [(chunk_id, score)] ranked. Score is standard BM25."""
    docs, df, avgdl = index["docs"], index["df"], index["avgdl"]
    n_docs = len(docs)
    out = []
    for cid, d in docs.items():
        s = 0.0
        for t in dict.fromkeys(tokenize(query)):
            f = d["tf"].get(t, 0)
            if f == 0:
                continue
            idf = math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * d["len"] / avgdl))
        if s > 0:
            out.append((cid, s))
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


def run(chunks_path: Path, out_path: Path) -> dict:
    chunks = [json.loads(l) for l in chunks_path.open() if l.strip()]
    index = build_index(chunks)
    out_path.write_text(json.dumps(index))
    return {
        "docs": len(index["docs"]),
        "vocab": len(index["df"]),
        "mb": round(out_path.stat().st_size / 1e6, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the BM25 keyword index.")
    ap.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/bm25.json"))
    a = ap.parse_args()
    print(json.dumps(run(a.chunks, a.out), indent=1))
