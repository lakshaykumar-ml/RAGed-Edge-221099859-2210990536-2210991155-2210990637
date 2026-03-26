# ================= EXTENSIVE RESEARCH-GRADE RAG SCRIPT =================

import time
import json
import faiss
import numpy as np
import re
import requests
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
import torch
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIG =================
EMBED_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "gemma3:12b"

DATASET_SIZE = 120

CHUNK_SIZES = [128, 256]
OVERLAPS = [0, 32]
TOP_K_VALUES = [3, 5, 10]
FINAL_K = 5

PROMPT_TYPES = ["baseline", "strict", "citation"]
RERANKING_TYPES = ["off", "cross"]

# 🔥 NEW EXPERIMENT DIMENSIONS
NOISE_LEVELS = [0.0, 0.3, 0.6]
MAX_CONTEXT_CHUNKS = [2, 3, 5]

NUM_THREADS = 8
OUTPUT_FILE = "rag_results_extensive.json"

CACHE = {}

# ================= LLM =================
def generate_answer(prompt):
    key = hash(prompt)
    if key in CACHE:
        return CACHE[key]

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        out = res.json()["response"].strip()
        CACHE[key] = out
        return out
    except:
        return None

# ================= NORMALIZE =================
def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

# ================= METRICS =================
def compute_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common)/len(pred_tokens)
    recall = len(common)/len(gold_tokens)
    return 2*precision*recall/(precision+recall)

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

# ================= RETRIEVAL METRICS =================
def recall_at_k(chunks, gold):
    g = normalize(gold)
    return int(any(g in normalize(c) for c in chunks))

def reciprocal_rank(chunks, gold):
    g = normalize(gold)
    for i, c in enumerate(chunks):
        if g in normalize(c):
            return 1/(i+1)
    return 0.0

# ================= EXTRA METRICS =================
def answer_length(ans):
    return len(ans.split())

def hallucination_flag(pred, contexts):
    ctx = " ".join(contexts)
    return int(normalize(pred) not in normalize(ctx))

def faithfulness_score(answer, contexts):
    ctx = "\n".join(contexts)
    prompt = f"Answer: {answer}\nContext: {ctx}\nSupported? YES or NO"
    resp = generate_answer(prompt)
    return 1 if resp and "yes" in resp.lower() else 0

# ================= DATA =================
def get_text(ex):
    return ex["question"], ex["answers"]["text"][0], ex["context"]

# ================= CHUNK =================
def chunk_text(text, size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += max(1, size-overlap)
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
def retrieve(q, model, index, chunks, emb, k):
    q_emb = model.encode([q])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# ================= RERANK =================
def cross_rerank(query, chunks, cross_encoder):
    if not chunks:
        return chunks

    pairs = [(query, chunk) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked]

# ================= NOISE =================
def add_noise(chunks, corpus, ratio):
    if ratio == 0:
        return chunks
    noise = random.sample(corpus, min(len(corpus), int(len(chunks)*ratio)))
    mixed = chunks + noise
    random.shuffle(mixed)
    return mixed

# ================= PROMPT =================
def build_prompt(q, ctxs, p):
    ctx = "\n\n".join([f"[{i+1}] {c}" for i,c in enumerate(ctxs)])
    if p=="baseline":
        return f"Answer using context.\nQ:{q}\n{ctx}\nA:"
    if p=="strict":
        return f"Only answer if supported else say Not found.\nQ:{q}\n{ctx}\nA:"
    if p=="citation":
        return f"Answer with citations.\nQ:{q}\n{ctx}\nA:"

# ================= QUERY =================
def run_query(ex, model, index, corpus, emb, rerank, ptype, k, cross, noise, max_ctx):
    q, gold, _ = get_text(ex)

    chunks = retrieve(q, model, index, corpus, emb, k)
    chunks = add_noise(chunks, corpus, noise)

    recall_k = recall_at_k(chunks, gold)
    mrr = reciprocal_rank(chunks, gold)

    if rerank=="cross":
        chunks = cross_rerank(q, chunks, cross)

    final = chunks[:max_ctx]

    prompt = build_prompt(q, final, ptype)
    pred = generate_answer(prompt)

    if not pred:
        return None

    return {
        "f1": compute_f1(pred, gold),
        "em": exact_match(pred, gold),
        "faithfulness": faithfulness_score(pred, final),
        "recall@k": recall_k,
        "mrr": mrr,
        "answer_len": answer_length(pred),
        "hallucination": hallucination_flag(pred, final)
    }

# ================= MAIN =================
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    dataset = load_dataset("squad", split=f"validation[:{DATASET_SIZE}]")
    results = []

    for chunk_size in CHUNK_SIZES:
        corpus = []
        for ex in dataset:
            _,_,ctx = get_text(ex)
            corpus.extend(chunk_text(ctx, chunk_size, OVERLAPS[0]))

        index, emb = build_index(corpus, embed_model)

        for k in TOP_K_VALUES:
            for rerank in RERANKING_TYPES:
                for ptype in PROMPT_TYPES:
                    for noise in NOISE_LEVELS:
                        for max_ctx in MAX_CONTEXT_CHUNKS:

                            print(f"Run: cs={chunk_size}, k={k}, noise={noise}, ctx={max_ctx}")

                            config = []

                            with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex_pool:
                                futures = [
                                    ex_pool.submit(run_query, ex, embed_model, index, corpus, emb,
                                                   rerank, ptype, k, cross_encoder, noise, max_ctx)
                                    for ex in dataset
                                ]

                                for f in tqdm(as_completed(futures), total=len(futures)):
                                    try:
                                        r = f.result()
                                    except Exception as e:
                                        print(f"Worker failed: {e}")
                                        continue
                                    if r:
                                        config.append(r)

                            if config:
                                results.append({
                                    "chunk_size": chunk_size,
                                    "top_k": k,
                                    "reranking": rerank,
                                    "prompt": ptype,
                                    "noise": noise,
                                    "max_context": max_ctx,

                                    "f1_mean": mean(x["f1"] for x in config),
                                    "recall@k_mean": mean(x["recall@k"] for x in config),
                                    "mrr_mean": mean(x["mrr"] for x in config),
                                    "hallucination_rate": mean(x["hallucination"] for x in config),
                                    "avg_answer_len": mean(x["answer_len"] for x in config)
                                })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Extensive experiments complete")

if __name__ == "__main__":
    run()
