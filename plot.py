import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data
with open("rag_results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ================= HEATMAP =================
pivot = df.groupby(["chunk_size", "top_k"])["f1"].mean().unstack()

plt.figure()
sns.heatmap(pivot, annot=True)
plt.title("Chunk Size vs Top-k (F1)")
plt.xlabel("Top-k")
plt.ylabel("Chunk Size")
plt.savefig("heatmap.png")
plt.close()


# ================= PROMPT BAR =================
prompt_df = df.groupby("prompt_type")[["f1", "not_found"]].mean()

prompt_df.plot(kind="bar")
plt.title("Prompt Trade-off")
plt.ylabel("Score")
plt.savefig("prompt_bar.png")
plt.close()


# ================= LATENCY SCATTER =================
plt.figure()
plt.scatter(df["latency"], df["f1"])
plt.xlabel("Latency")
plt.ylabel("F1 Score")
plt.title("Latency vs Quality")
plt.savefig("latency_scatter.png")
plt.close()