import json
import os
import subprocess
import sys

def setup_java_env():
    if not os.environ.get("JAVA_HOME"):
        java_home = None
        for version in ("21", "17", "11"):
            try:
                java_home = subprocess.check_output(
                    ["/usr/libexec/java_home", "-v", version],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                if java_home:
                    break
            except Exception:
                continue
        if not java_home:
            for path in (
                "/opt/homebrew/opt/openjdk@21", "/usr/local/opt/openjdk@21",
                "/opt/homebrew/opt/openjdk", "/usr/local/opt/openjdk",
            ):
                if os.path.isdir(path):
                    java_home = path
                    break
        if not java_home:
            raise RuntimeError(
                "JAVA_HOME não encontrado. Rode: brew install openjdk@21 "
                "e o symlink indicado no topo deste arquivo."
            )
        os.environ["JAVA_HOME"] = java_home

    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-key-to-bypass-error")
    print(f"[java] JAVA_HOME = {os.environ['JAVA_HOME']}")


setup_java_env()

import pandas as pd
import pytrec_eval

# Constantes
CORPUS_DIR = "corpus_dir"
CORPUS_PATH = os.path.join(CORPUS_DIR, "trec2021_corpus.jsonl")
INDEX_DIR = "indexes/trec2021_bm25"
QUERIES_CSV = "trec2021_queries.csv"
QRELS_CSV = "trec2021_qrels.csv"
TOP_K = 100
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
METRICS = {"ndcg_cut_10", "P_10", "recall_100"}


# Avaliação
def load_qrels(qrels_csv=QRELS_CSV):
    df = pd.read_csv(qrels_csv)
    qrels = {}
    for _, row in df.iterrows():
        qid, docid = str(row["query_id"]), str(row["doc_id"])
        qrels.setdefault(qid, {})[docid] = int(row["relevance"])
    return qrels


def evaluate(run_results, qrels_csv=QRELS_CSV):
    evaluator = pytrec_eval.RelevanceEvaluator(load_qrels(qrels_csv), METRICS)
    per_query = evaluator.evaluate(run_results)
    avg = {m: 0.0 for m in ("ndcg_cut_10", "P_10", "recall_100")}
    for qid in per_query:
        for m in avg:
            avg[m] += per_query[qid][m]
    for m in avg:
        avg[m] /= len(per_query)
    return avg


def print_results(title, avg):
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")
    print(f"NDCG@10:      {avg['ndcg_cut_10']:.4f}")
    print(f"Precision@10: {avg['P_10']:.4f}")
    print(f"Recall@100:   {avg['recall_100']:.4f}")
    print(line)


def load_queries():
    df = pd.read_csv(QUERIES_CSV)
    return dict(zip(df["query_id"].astype(str), df["text"]))


# FASE 1 — Download e geração dos ficheiros estáticos
def phase1():
    if all(os.path.exists(p) for p in (QUERIES_CSV, QRELS_CSV, CORPUS_PATH)):
        print("[fase 1] Ficheiros já existem, a pular download.")
        return

    import ir_datasets

    os.makedirs(CORPUS_DIR, exist_ok=True)
    print("[fase 1] A descarregar o TREC Clinical Trials 2021 (lento na 1ª vez)...")
    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

    queries = [{"query_id": q.query_id, "text": q.text} for q in dataset.queries_iter()]
    pd.DataFrame(queries).to_csv(QUERIES_CSV, index=False)
    print(f"[fase 1] {len(queries)} queries -> {QUERIES_CSV}")

    qrels = [
        {"query_id": qr.query_id, "doc_id": qr.doc_id,
         "relevance": qr.relevance, "iteration": qr.iteration}
        for qr in dataset.qrels_iter()
    ]
    pd.DataFrame(qrels).to_csv(QRELS_CSV, index=False)
    print(f"[fase 1] {len(qrels)} qrels -> {QRELS_CSV}")

    print("[fase 1] A gerar o corpus JSONL...")
    doc_count = 0
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for doc in dataset.docs_iter():
            content = "\n".join([
                doc.title or "", doc.summary or "",
                doc.detailed_description or "", doc.eligibility or "",
            ])
            f.write(json.dumps({"id": doc.doc_id, "contents": content}) + "\n")
            doc_count += 1
            if doc_count % 50000 == 0:
                print(f"   {doc_count} documentos...")
    print(f"[fase 1] {doc_count} documentos -> {CORPUS_PATH}")


# FASE 2 — Indexação e busca BM25
def build_index_if_needed():
    if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("[fase 2] Índice já existe, a pular a indexação.")
        return
    print("[fase 2] A indexar com Pyserini (2 a 5 min)...")
    subprocess.run(
        [sys.executable, "-m", "pyserini.index.lucene",
         "--collection", "JsonCollection", "--input", CORPUS_DIR,
         "--index", INDEX_DIR, "--generator", "DefaultLuceneDocumentGenerator",
         "--threads", "2", "--storePositions", "--storeDocvectors", "--storeRaw"],
        check=True, env=os.environ,
    )


def _run_search(searcher, queries, out_csv):
    run_results, records = {}, []
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=TOP_K)
        run_results[qid] = {}
        for rank, hit in enumerate(hits):
            run_results[qid][hit.docid] = hit.score
            records.append({"query_id": qid, "doc_id": hit.docid,
                            "rank": rank + 1, "score": hit.score})
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"   candidatos -> {out_csv}")
    return run_results


def phase2():
    from pyserini.search.lucene import LuceneSearcher

    build_index_if_needed()
    print("[fase 2] Busca BM25...")
    searcher = LuceneSearcher(INDEX_DIR)
    run_results = _run_search(searcher, load_queries(), "fase2_bm25_top100_candidatos.csv")
    print_results("BASELINE LÉXICO (BM25)", evaluate(run_results))


# FASE 3 — Expansão RM3
def phase3():
    from pyserini.search.lucene import LuceneSearcher

    print("[fase 3] Busca BM25 + RM3...")
    searcher = LuceneSearcher(INDEX_DIR)
    searcher.set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5)
    run_results = _run_search(searcher, load_queries(), "fase3_rm3_top100_candidatos.csv")
    print_results("FASE 3: EXPANSÃO DE CONSULTAS (BM25 + RM3)", evaluate(run_results))


# FASE 4 — Re-ranqueamento neural
def phase4():
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm

    candidates_csv = "fase3_rm3_top100_candidatos.csv"
    print("[fase 4] A carregar o Cross-Encoder...")
    model = CrossEncoder(CROSS_ENCODER, max_length=512)

    df_candidates = pd.read_csv(candidates_csv)
    queries = load_queries()

    needed_ids = set(df_candidates["doc_id"].astype(str))
    print(f"[fase 4] A carregar {len(needed_ids)} documentos do corpus...")
    corpus = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] in needed_ids:
                corpus[data["id"]] = data["contents"]

    print("[fase 4] A re-ranquear...")
    final_records = []
    for qid, group in tqdm(df_candidates.groupby("query_id")):
        qtext = queries[str(qid)]
        pairs = [(qtext, corpus[str(d)]) for d in group["doc_id"]]
        group = group.copy()
        group["neural_score"] = model.predict(pairs)
        group = group.sort_values(by="neural_score", ascending=False)
        for rank, (_, row) in enumerate(group.iterrows()):
            final_records.append({"query_id": qid, "doc_id": row["doc_id"],
                                  "rank": rank + 1, "score": row["neural_score"]})

    df_final = pd.DataFrame(final_records)
    df_final.to_csv("fase4_final_reranked.csv", index=False)
    print("   resultados -> fase4_final_reranked.csv")

    run_results = {}
    for qid in df_final["query_id"].unique():
        subset = df_final[df_final["query_id"] == qid]
        run_results[str(qid)] = dict(zip(subset["doc_id"].astype(str), subset["score"]))
    print_results("FINAL: PIPELINE NEURAL (BM25 + RM3 + Cross-Encoder)", evaluate(run_results))


# FASE 5 — Estudo de caso: consultas específicas de Alzheimer
#   Roda os 3 pipelines (BM25 / BM25+RM3 / +Cross-Encoder) nas consultas de
#   Alzheimer e monta um pool de julgamento (top-5 de cada pipeline, sem
#   duplicados) para avaliação manual de relevância. As métricas quantitativas
#   só ficam disponíveis depois que o pool for julgado (ver ALZHEIMER_QRELS_CSV
#   e phase6())
ALZHEIMER_POOL_CSV = "alzheimer_pool_para_julgar.csv"
ALZHEIMER_QRELS_CSV = "alzheimer_qrels.csv"
POOL_DEPTH = 5


def phase5():
    from pyserini.search.lucene import LuceneSearcher
    from sentence_transformers import CrossEncoder
    from alzheimer_queries import ALZHEIMER_QUERIES

    print(f"[fase 5] {len(ALZHEIMER_QUERIES)} consultas de Alzheimer...")

    print("[fase 5] Busca BM25 (Pipeline 1)...")
    searcher_bm25 = LuceneSearcher(INDEX_DIR)
    run_bm25 = _run_search(
        searcher_bm25, ALZHEIMER_QUERIES, "fase5_alzheimer_bm25_top100.csv"
    )

    print("[fase 5] Busca BM25 + RM3 (Pipeline 2)...")
    searcher_rm3 = LuceneSearcher(INDEX_DIR)
    searcher_rm3.set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5)
    run_rm3 = _run_search(
        searcher_rm3, ALZHEIMER_QUERIES, "fase5_alzheimer_bm25rm3_top100.csv"
    )

    print("[fase 5] A carregar o Cross-Encoder e a re-ranquear (Pipeline 3)...")
    model = CrossEncoder(CROSS_ENCODER, max_length=512)
    needed_ids = {docid for hits in run_bm25.values() for docid in hits}
    needed_ids |= {docid for hits in run_rm3.values() for docid in hits}
    corpus = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] in needed_ids:
                corpus[data["id"]] = data["contents"]

    final_records = []
    for qid, qtext in ALZHEIMER_QUERIES.items():
        doc_ids = list(run_rm3[qid].keys())
        pairs = [(qtext, corpus[d]) for d in doc_ids]
        scores = model.predict(pairs)
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked):
            final_records.append({"query_id": qid, "doc_id": doc_id,
                                  "rank": rank + 1, "score": score})

    df_final = pd.DataFrame(final_records)
    df_final.to_csv("fase5_alzheimer_reranked.csv", index=False)
    print("   resultados -> fase5_alzheimer_reranked.csv")

    pool_rows = []
    for qid, qtext in ALZHEIMER_QUERIES.items():
        pooled_ids, seen = [], set()
        top_bm25 = list(run_bm25[qid].keys())[:POOL_DEPTH]
        top_rm3 = list(run_rm3[qid].keys())[:POOL_DEPTH]
        top_rerank = list(
            df_final[df_final["query_id"] == qid].sort_values("rank")["doc_id"]
        )[:POOL_DEPTH]
        for doc_id in top_bm25 + top_rm3 + [str(d) for d in top_rerank]:
            if doc_id not in seen:
                seen.add(doc_id)
                pooled_ids.append(doc_id)
        for doc_id in pooled_ids:
            content = corpus[doc_id]
            title = content.split("\n", 1)[0][:150]
            snippet = content[:400].replace("\n", " ")
            pool_rows.append({
                "query_id": qid, "query_text": qtext, "doc_id": doc_id,
                "title": title, "snippet": snippet, "relevance": "",
            })

    df_pool = pd.DataFrame(pool_rows)
    df_pool.to_csv(ALZHEIMER_POOL_CSV, index=False)
    print(f"   pool de julgamento ({len(df_pool)} pares query-doc, "
          f"top-{POOL_DEPTH} por pipeline, deduplicados) -> {ALZHEIMER_POOL_CSV}")
    print(f"\n[fase 5] Preencha a coluna 'relevance' (0 ou 1) em {ALZHEIMER_POOL_CSV} "
          f"e depois gere {ALZHEIMER_QRELS_CSV} a partir dele para rodar a fase 6 "
          "(métricas quantitativas).")


# FASE 6 — Métricas quantitativas dos 3 pipelines no subconjunto Alzheimer
def phase6():
    from alzheimer_queries import ALZHEIMER_QUERIES

    if not os.path.exists(ALZHEIMER_QRELS_CSV):
        raise FileNotFoundError(
            f"{ALZHEIMER_QRELS_CSV} não encontrado. Rode a fase 5 primeiro, "
            f"julgue {ALZHEIMER_POOL_CSV} e gere o qrels."
        )

    def load_run(csv_path):
        df = pd.read_csv(csv_path)
        run = {}
        for qid, group in df.groupby("query_id"):
            run[str(qid)] = dict(zip(group["doc_id"].astype(str), group["score"]))
        return run

    pipelines = [
        ("Pipeline 1: BM25", "fase5_alzheimer_bm25_top100.csv"),
        ("Pipeline 2: BM25 + RM3", "fase5_alzheimer_bm25rm3_top100.csv"),
        ("Pipeline 3: BM25 + RM3 + Cross-Encoder", "fase5_alzheimer_reranked.csv"),
    ]
    print(f"\n[fase 6] Avaliando {len(ALZHEIMER_QUERIES)} consultas de Alzheimer "
          f"contra {ALZHEIMER_QRELS_CSV}...")
    per_query_ndcg = {}
    runs = {}
    for title, csv_path in pipelines:
        run = load_run(csv_path)
        runs[title] = run
        evaluator = pytrec_eval.RelevanceEvaluator(
            load_qrels(ALZHEIMER_QRELS_CSV), METRICS
        )
        per_query = evaluator.evaluate(run)
        per_query_ndcg[title] = {qid: v["ndcg_cut_10"] for qid, v in per_query.items()}
        avg = {m: 0.0 for m in ("ndcg_cut_10", "P_10", "recall_100")}
        for qid in per_query:
            for m in avg:
                avg[m] += per_query[qid][m]
        for m in avg:
            avg[m] /= len(per_query)
        print_results(f"ALZHEIMER — {title}", avg)

    # Análise de erro (Fase 5.2): consultas onde o Cross-Encoder mais
    # ajudou e mais atrapalhou em relação ao BM25+RM3, para inspeção manual.
    title_rm3 = "Pipeline 2: BM25 + RM3"
    title_ce = "Pipeline 3: BM25 + RM3 + Cross-Encoder"
    deltas = sorted(
        ((qid, per_query_ndcg[title_ce][qid] - per_query_ndcg[title_rm3][qid])
         for qid in per_query_ndcg[title_ce]),
        key=lambda x: x[1],
    )
    from alzheimer_queries import ALZHEIMER_QUERIES

    needed_ids = set()
    for _, run in runs.items():
        for qid in (deltas[0][0], deltas[-1][0]):
            needed_ids |= set(list(run[qid].keys())[:5])
    corpus = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] in needed_ids:
                corpus[data["id"]] = data["contents"]

    print("\n" + "=" * 60)
    print("FASE 6: ANÁLISE DE ERRO — maior ganho e maior perda do Cross-Encoder")
    print("=" * 60)
    for label, (qid, delta) in (("MAIOR GANHO", deltas[-1]), ("MAIOR PERDA", deltas[0])):
        print(f"\n[{label}] {qid}  (NDCG@10 RM3={per_query_ndcg[title_rm3][qid]:.3f} "
              f"-> Cross-Encoder={per_query_ndcg[title_ce][qid]:.3f}, "
              f"delta={delta:+.3f})")
        print(f"   {ALZHEIMER_QUERIES[qid][:160]}...")
        for pl_title in (title_rm3, title_ce):
            print(f"   -- top-5 {pl_title} --")
            top5 = list(runs[pl_title][qid].keys())[:5]
            for doc_id in top5:
                rel = load_qrels(ALZHEIMER_QRELS_CSV).get(qid, {}).get(doc_id, "?")
                title_doc = corpus.get(doc_id, "").split("\n", 1)[0][:90]
                print(f"      {doc_id}  rel={rel}  {title_doc}")


PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6}

def main():
    args = sys.argv[1:]
    to_run = [int(a) for a in args] if args else [1, 2, 3, 4]
    for n in to_run:
        PHASES[n]()
    print("\nConcluído.")


if __name__ == "__main__":
    main()