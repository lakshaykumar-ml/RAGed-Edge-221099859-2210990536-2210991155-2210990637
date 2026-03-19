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

# ================= CONFIG =================
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "llama3.2:3b"

CHUNK_SIZES = [128, 256]
OVERLAPS = [0, 32]

TOP_K_VALUES = [5, 10]
FINAL_K = 5

PROMPT_TYPES = ["baseline", "strict", "citation"]
TEMPERATURES = [0.0]

RERANKING_TYPES = ["off", "embedding", "cross"]

DATASET_SIZE = 30
MAX_PROMPT_LENGTH = 6000

OUTPUT_FILE = "rag_results.json"

# ================= OLLAMA =================
def generate_answer(prompt, temperature=0.0):
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
        return res.json()["response"].strip()
    except Exception as e:
        print("❌ LLM ERROR:", e)
        return None

# ================= DATA =================
def get_text(example):
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    context = example["context"]
    return question, answer, context

# ================= CHUNKING =================
def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

# ================= BUILD INDEX =================
def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

# ================= RETRIEVE =================
def retrieve(query, model, index, chunks, embeddings, top_k):
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]], embeddings[I[0]]

# ================= RERANKING =================
def embedding_rerank(query_emb, chunk_embs, chunks):
    scores = np.dot(chunk_embs, query_emb.T).reshape(-1)
    sorted_idx = np.argsort(scores)[::-1]
    return [chunks[i] for i in sorted_idx]

def cross_rerank(query, chunks, reranker):
    try:
        pairs = [(query, c[:512]) for c in chunks]
        scores = reranker.predict(pairs)
        sorted_idx = np.argsort(scores)[::-1]
        return [chunks[i] for i in sorted_idx]
    except Exception as e:
        print("❌ RERANK ERROR:", e)
        return chunks

# ================= PROMPT =================
def build_prompt(query, contexts, prompt_type):
    context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])

    if prompt_type == "baseline":
        prompt = f"""
Answer using ONLY the context.

Question:
{query}

Context:
{context_text}

Answer:
"""

    elif prompt_type == "strict":
        prompt = f"""
STRICT:
- Only answer if fully supported
- Otherwise say "Not found"

Question:
{query}

Context:
{context_text}

Answer:
"""

    elif prompt_type == "citation":
        prompt = f"""
Answer using ONLY the context.

RULES:
- Every statement MUST include citation like [1], [2]
- No unsupported claims
- If not found say "Not found"

Question:
{query}

Context:
{context_text}

Answer:
"""

    if len(prompt) > MAX_PROMPT_LENGTH:
        return None

    return prompt

# ================= METRICS =================
def compute_f1(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def not_found(pred):
    return int("not found" in pred.lower())

# ================= ADVANCED METRICS =================
def faithfulness_score(answer, contexts):
    ctx = "\n".join(contexts)

    prompt = f"""
Check if the answer is fully supported by the context.

Answer:
{answer}

Context:
{ctx}

Reply ONLY YES or NO
"""
    resp = generate_answer(prompt, 0.0)
    return 1 if resp and "yes" in resp.lower() else 0

def citation_accuracy(answer, contexts):
    citations = re.findall(r"\[(\d+)\]", answer)

    if not citations:
        return 0

    valid = 0
    for c in citations:
        idx = int(c) - 1
        if 0 <= idx < len(contexts):
            valid += 1

    return valid / len(citations)

def context_recall(contexts, answer):
    return int(any(answer.lower() in c.lower() for c in contexts))

def context_precision(contexts, answer):
    relevant = sum(1 for c in contexts if answer.lower() in c.lower())
    return relevant / len(contexts) if contexts else 0

# ================= MAIN =================
def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    cross_encoder = None
    if "cross" in RERANKING_TYPES:
        cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu")

    dataset = load_dataset("squad", split=f"validation[:{DATASET_SIZE}]")

    results = []

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:

            print(f"\n=== chunk={chunk_size}, overlap={overlap} ===")

            corpus = []
            for ex in dataset:
                _, _, context = get_text(ex)
                corpus.extend(chunk_text(context, chunk_size, overlap))

            index, embeddings = build_index(corpus, embed_model)

            for top_k in TOP_K_VALUES:
                for rerank_type in RERANKING_TYPES:
                    for prompt_type in PROMPT_TYPES:
                        for temp in TEMPERATURES:

                            print(f"\nk={top_k}, rerank={rerank_type}, prompt={prompt_type}")

                            for ex in tqdm(dataset):

                                query, answer, _ = get_text(ex)
                                start = time.time()

                                retrieved_chunks, retrieved_embs = retrieve(
                                    query, embed_model, index, corpus, embeddings, top_k
                                )

                                # RERANK
                                if rerank_type == "embedding":
                                    q_emb = embed_model.encode([query])[0]
                                    reranked = embedding_rerank(q_emb, retrieved_embs, retrieved_chunks)

                                elif rerank_type == "cross":
                                    reranked = cross_rerank(query, retrieved_chunks, cross_encoder)

                                else:
                                    reranked = retrieved_chunks

                                final_chunks = reranked[:FINAL_K]

                                prompt = build_prompt(query, final_chunks, prompt_type)
                                if prompt is None:
                                    continue

                                pred = generate_answer(prompt, temp)
                                if pred is None:
                                    continue

                                latency = time.time() - start

                                results.append({
                                    "query": query,
                                    "ground_truth": answer,
                                    "prediction": pred,
                                    "chunk_size": chunk_size,
                                    "overlap": overlap,
                                    "top_k": top_k,
                                    "reranking": rerank_type,
                                    "prompt_type": prompt_type,
                                    "f1": compute_f1(pred, answer),
                                    "em": exact_match(pred, answer),
                                    "faithfulness": faithfulness_score(pred, final_chunks),
                                    "citation_acc": citation_accuracy(pred, final_chunks),
                                    "context_recall": context_recall(final_chunks, answer),
                                    "context_precision": context_precision(final_chunks, answer),
                                    "latency": latency,
                                    "not_found": not_found(pred)
                                })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    run_experiment()