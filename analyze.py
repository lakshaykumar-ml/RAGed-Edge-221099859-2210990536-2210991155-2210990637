import json
import pandas as pd

INPUT_FILE = "rag_results.json"

# ================= LOAD =================
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("\nLoaded rows:", len(df))


# ================= TABLE A1 =================
print("\n=== TABLE A1: Performance ===")

group_cols = [
    "chunk_size",
    "overlap",
    "top_k",
    "reranking",
    "prompt_type"
]

metrics = [
    "f1",
    "em",
    "faithfulness",
    "citation_acc",
    "context_recall",
    "context_precision"
]

table_a1 = df.groupby(group_cols)[metrics].mean().reset_index()

print(table_a1)

table_a1.to_csv("table_A1_performance.csv", index=False)


# ================= TABLE A2 =================
print("\n=== TABLE A2: Efficiency ===")

table_a2 = df.groupby(group_cols)[["latency"]].mean().reset_index()

print(table_a2)

table_a2.to_csv("table_A2_latency.csv", index=False)


# ================= TABLE A3 =================
print("\n=== TABLE A3: Interaction (chunk_size × top_k) ===")

table_a3 = df.groupby(["chunk_size", "top_k"])[metrics].mean().reset_index()

print(table_a3)

table_a3.to_csv("table_A3_chunk_topk.csv", index=False)


# ================= EXTRA: BEST CONFIG =================
print("\n=== BEST CONFIG (by F1) ===")

best = table_a1.sort_values("f1", ascending=False).head(5)

print(best)


# ================= EXTRA: FAILURE ANALYSIS =================
print("\n=== FAILURE ANALYSIS ===")

failures = df[df["f1"] < 0.2]

print("Total failures:", len(failures))

failures.to_csv("failures.csv", index=False)


# ================= EXTRA: NOT FOUND RATE =================
print("\n=== NOT FOUND RATE ===")

if "not_found" in df.columns:
    nf = df.groupby(group_cols)["not_found"].mean().reset_index()
    print(nf)
    nf.to_csv("not_found_rate.csv", index=False)


print("\nSaved all tables.")