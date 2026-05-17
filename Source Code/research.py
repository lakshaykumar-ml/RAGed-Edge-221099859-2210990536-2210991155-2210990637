# ================= RESEARCH-GRADE FAST RAG SCRIPT =================
# Optimized for A100 (fast + strong experiments)

import time
import json
import faiss
import numpy as np
import re
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
import torch
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIG =================
EMBED_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Fast model for bulk
# OLLAMA_MODEL = "llama3.2:3b"
# For final validation (manually switch)
OLLAMA_MODEL = "gemma3:12b"

DATASET_SIZE = 120

CHUNK_SIZES = [128, 256]
OVERLAPS = [0,32]
TOP_K_VALUES = [3, 5, 10]
FINAL_K = 5

PROMPT_TYPES = ["baseline", "strict", "citation"]
RERANKING_TYPES = ["off", "cross"]
TEMPERATURES = [0.0]

NUM_THREADS = 8
OUTPUT_FILE = "rag_results_research.json"

# ================= CACHE =================
CACHE = {}

# ================= LLM =================
def generate_answer(prompt, temperature=0.0):
    key = hash(prompt)
    if key in CACHE:
        return CACHE[key]

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        out = res.json()["response"].strip()
        CACHE[key] = out
        return out
    except:
        return None

# ================= NORMALIZATION =================
def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

# ================= METRICS =================
def compute_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

# ================= RETRIEVAL METRICS =================

def recall_at_k(chunks, gold_answer):
    gold = normalize(gold_answer)
    for c in chunks:
        if gold in normalize(c):
            return 1
    return 0


def reciprocal_rank(chunks, gold_answer):
    gold = normalize(gold_answer)
    for i, c in enumerate(chunks):
        if gold in normalize(c):
            return 1 / (i + 1)
    return 0.0

# ================= FAITHFULNESS =================
def faithfulness_score(answer, contexts):
    ctx = "\n".join(contexts)

    prompt = f"""
Check if answer is fully supported by context.

Answer: {answer}
Context: {ctx}

Reply YES or NO.
"""

    resp = generate_answer(prompt, 0.0)
    return 1 if resp and "yes" in resp.lower() else 0

# ================= DATA =================
def get_text(example):
    q = example["question"]
    a = example["answers"]["text"][0] if example["answers"]["text"] else ""
    c = example["context"]
    return q, a, c

# ================= CHUNKING =================
def chunk_text(text, size=256, overlap=32):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return chunks

# ================= INDEX =================
def build_index(chunks, model):
    emb = model.encode(chunks, batch_size=128, show_progress_bar=False)
    emb = np.array(emb).astype("float32")

    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    return index, emb

# ================= RETRIEVE =================
def retrieve(query, model, index, chunks, emb, k):
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]], emb[I[0]]

# ================= RERANK =================
def cross_rerank(query, chunks, reranker):
    pairs = [(query, c[:512]) for c in chunks]
    scores = reranker.predict(pairs)
    idx = np.argsort(scores)[::-1]
    return [chunks[i] for i in idx]

# ================= PROMPTS =================
def build_prompt(query, contexts, ptype):
    ctx = "\n\n".join([f"[{i+1}] {c}" for i,c in enumerate(contexts)])

    if ptype == "baseline":
        return f"Answer using context only.\nQ: {query}\n{ctx}\nA:"

    if ptype == "strict":
        return f"Only answer if fully supported else say Not found.\nQ: {query}\n{ctx}\nA:"

    if ptype == "citation":
        return f"Answer with citations [1],[2]. No unsupported facts.\nQ: {query}\n{ctx}\nA:"

# ================= NO RAG =================
def no_rag_answer(query):
    prompt = f"Answer the question:\n{query}"
    return generate_answer(prompt, 0.0)

# ================= QUERY EXEC =================
def run_query(ex, embed_model, index, corpus, emb, rerank, ptype, k, cross_encoder):
    q, gold, _ = get_text(ex)

    chunks, chunk_embs = retrieve(q, embed_model, index, corpus, emb, k)

    recall_k = recall_at_k(chunks, gold)
    mrr = reciprocal_rank(chunks, gold)

    if rerank == "cross":
        chunks = cross_rerank(q, chunks, cross_encoder)

    final = chunks[:FINAL_K]

    prompt = build_prompt(q, final, ptype)
    pred = generate_answer(prompt)

    if not pred:
        return None

    return {
        "f1": compute_f1(pred, gold),
        "em": exact_match(pred, gold),
        "faithfulness": faithfulness_score(pred, final),
        "recall@k": recall_k,
        "mrr": mrr
    }

# ================= MAIN =================
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    dataset = load_dataset("squad", split=f"validation[:{DATASET_SIZE}]")

    results = []

    # ===== NO-RAG BASELINE =====
    print("Running No-RAG baseline...")
    baseline_scores = []
    for ex in tqdm(dataset):
        q, gold, _ = get_text(ex)
        pred = no_rag_answer(q)
        if pred:
            baseline_scores.append(compute_f1(pred, gold))

    results.append({
        "type": "no_rag",
        "f1_mean": mean(baseline_scores)
    })

    # ===== RAG =====
    for chunk_size in CHUNK_SIZES:
        corpus = []
        for ex in dataset:
            _,_,ctx = get_text(ex)
            corpus.extend(chunk_text(ctx, chunk_size, OVERLAPS[0]))

        index, emb = build_index(corpus, embed_model)

        for k in TOP_K_VALUES:
            for rerank in RERANKING_TYPES:
                for ptype in PROMPT_TYPES:

                    print(f"Running: chunk={chunk_size}, k={k}, rerank={rerank}, prompt={ptype}")

                    config = []

                    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                        futures = [
                            executor.submit(run_query, ex, embed_model, index, corpus, emb, rerank, ptype, k, cross_encoder)
                            for ex in dataset
                        ]

                        for f in tqdm(as_completed(futures), total=len(futures)):
                            r = f.result()
                            if r:
                                config.append(r)

                    if config:
                        f1s = [x["f1"] for x in config]
                        recalls = [x["recall@k"] for x in config]
                        mrrs = [x["mrr"] for x in config]

                        results.append({
                            "chunk_size": chunk_size,
                            "top_k": k,
                            "reranking": rerank,
                            "prompt": ptype,
                            "f1_mean": mean(f1s),
                            "f1_std": stdev(f1s) if len(f1s)>1 else 0,
                            "recall@k_mean": mean(recalls),
                            "mrr_mean": mean(mrrs)
                        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved research-grade results")


if __name__ == "__main__":
    run()