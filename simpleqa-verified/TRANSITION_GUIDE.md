You can treat **SimpleQA-Verified** as a cleaned, labeled subset of the SimpleQA you’re already using, so migrating is actually pretty painless.

---

## 1. What changes when you move to SimpleQA-Verified?

**Original SimpleQA (OpenAI)**

* ~4,326 questions.([Ivan Zhou][1])
* Fields (HF `basicv8vc/SimpleQA`):
  `metadata` (JSON with `topic`, `answer_type`, `urls`), `problem` (question), `answer`.([Hugging Face][2])

**SimpleQA-Verified (Google DeepMind)**

* 1,000 prompts, curated from SimpleQA to: de-duplicate, fix wrong labels, rebalance topics, reconcile conflicting sources, and align URLs with publisher crawling preferences.([arXiv][3])
* Fields (HF `google/simpleqa-verified` and Kaggle):
  `original_index`, `problem`, `answer`, `topic`, `answer_type`, `multi_step`, `requires_reasoning`, `urls`.([Kaggle][4])

The key thing for you:

* **`original_index` tells you exactly which row in the original SimpleQA each verified example came from.** ([Kaggle][4])
* Schema is almost the same as what you’ve already been consuming, just with extra columns and a smaller, higher-quality subset.

---

## 2. “Quick transition” path for your eval pipeline

### A. Swap the dataset in your loader

If you’re currently loading SimpleQA from Hugging Face like:

```python
from datasets import load_dataset

simpleqa = load_dataset("basicv8vc/SimpleQA", split="test")
# Fields: metadata (dict), problem, answer
```

You can add SimpleQA-Verified alongside it:

```python
from datasets import load_dataset

sq = load_dataset("basicv8vc/SimpleQA", split="test")          # 4.33k rows
sqv = load_dataset("google/simpleqa-verified", split="eval")   # 1k rows

def iter_simpleqa_verified():
    for ex in sqv:
        yield {
            "id": ex["original_index"],      # or your own ID
            "question": ex["problem"],
            "answer": ex["answer"],
            "topic": ex["topic"],
            "answer_type": ex["answer_type"],
            "multi_step": ex["multi_step"],
            "requires_reasoning": ex["requires_reasoning"],
            "urls": ex["urls"].split(","),
        }
```

Then just point your RAG benchmarking harness at `iter_simpleqa_verified()` instead of your old `iter_simpleqa()`.

### B. Keep comparability with your past SimpleQA runs

Because you have `original_index`, you can **recompute scores on the Verified subset using your existing logs**:

1. Load `simpleqa-verified` and grab the set of `original_index` values.
2. Filter your existing SimpleQA run logs to only those indices (or match by exact `problem` text).
3. Recompute metrics (accuracy/F1, abstain behavior, etc.) on that subset.

That gives you “retroactive” SimpleQA-Verified numbers with no need to rerun models, as long as you kept per-question outputs.

### C. If you rely on the official SimpleQA evaluator

* OpenAI’s `simple-evals` repo has the reference SimpleQA evaluator.([GitHub][5])
* SimpleQA-Verified provides **starter code and an updated autorater prompt** via Kaggle (“SimpleQA Verified Benchmark Starter Code”).([Kaggle][4])

Fastest path:

* Keep your existing eval flow but:

  * Swap the dataset to `google/simpleqa-verified`.
  * Port over the updated grading / autorater logic from the Kaggle starter notebook into your evaluation harness.

For RAG benchmarking, you might *not* actually need their autorater; you can just do exact/normalized string matching where possible and fall back to LLM-as-judge for fuzzy numeric/date answers, using their grading prompt as inspiration.

---

## 3. Getting the full dataset (QAs) + underlying webpages/files

### A. Full SimpleQA-Verified QA table

You can get the 1,000-row verified dataset from either:

* **Hugging Face**: `google/simpleqa-verified`([Hugging Face][6])
* **Kaggle dataset**: “SimpleQA Verified” from DeepMind.([Kaggle][4])

Example with Hugging Face:

```python
from datasets import load_dataset

sqv = load_dataset("google/simpleqa-verified", split="eval")
print(sqv[0])
# {
#   'original_index': 5,
#   'problem': '...',
#   'answer': '...',
#   'topic': 'Politics',
#   'answer_type': 'Number',
#   'multi_step': True/False,
#   'requires_reasoning': True/False,
#   'urls': 'url1,url2,url3'
# }
```

For the original SimpleQA (if you still want it):

```python
sq = load_dataset("basicv8vc/SimpleQA", split="test")  # 4.33k rows
```

Both are MIT-licensed.([Hugging Face][2])

### B. Underlying webpages / files for RAG

There is **no official, ready-made zip of all underlying web pages** (for either SimpleQA or SimpleQA-Verified) as far as the public releases go — only the **URLs** are provided.([Hugging Face][2])

So to build a RAG corpus you need to:

1. Read the `urls` field for each row (comma-separated in SimpleQA-Verified; list inside `metadata` in the original SimpleQA).([Hugging Face][2])
2. Crawl those URLs yourself (respecting `robots.txt` and publisher ToS — the Verified authors explicitly filtered URLs to align with crawling preferences, but you still need to behave nicely).([arXiv][3])
3. Store the responses as HTML/PDF/etc., and then run your usual text extraction → chunking → indexing for RAG.

Here’s a small “one-evening” script to fetch a local corpus for **SimpleQA-Verified**:

```python
from datasets import load_dataset
import httpx, os, time
from urllib.parse import urlparse

ds = load_dataset("google/simpleqa-verified", split="eval")

OUT_DIR = "simpleqa_verified_docs"
os.makedirs(OUT_DIR, exist_ok=True)

client = httpx.Client(follow_redirects=True, timeout=15.0, headers={
    "User-Agent": "SimpleQA-Verified-RAG-bench/1.0 (contact: your-email@example.com)"
})

for row in ds:
    qid = row["original_index"]
    # urls is a comma-separated string
    urls = [u.strip() for u in row["urls"].split(",") if u.strip()]
    for j, url in enumerate(urls):
        try:
            parsed = urlparse(url)
            # Heuristic: most are HTML; keep extension generic
            fname = f"{qid:04d}_{j}_{parsed.netloc.replace('.', '_')}.html"
            path = os.path.join(OUT_DIR, fname)

            if os.path.exists(path):
                continue  # already downloaded

            r = client.get(url)
            if r.status_code == 200 and r.text.strip():
                with open(path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(r.text)
            # Be a polite crawler
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed {qid} {url}: {e}")
```

Then you can:

* Run your HTML → text / PDF → text pipeline.
* Index everything into your vector store (or your existing CustomGPT ingestion).
* At eval time, your RAG system can be restricted to this corpus.

For the **original SimpleQA** corpus, the logic is almost identical; you just read `example["metadata"]["urls"]` instead of a comma-separated string.

---

## 4. How I’d wire this into your RAG benchmark specifically

Given what you’re doing at CustomGPT.ai, I’d do:

1. **Adopt SimpleQA-Verified as your default factuality eval**

   * Replace SimpleQA with SimpleQA-Verified for new experiments.
   * Tag each question with `multi_step` / `requires_reasoning` so you can slice metrics by complexity.([Hugging Face][6])

2. **Build a “SimpleQA-Verified RAG corpus” once**

   * Use the script above (maybe parallelized) to fetch all URLs.
   * Normalize to text and index with your standard pipeline.
   * Store a manifest: `original_index -> list of doc IDs` so you can optionally restrict each query to only its own source docs.

3. **Run two modes in your benchmark:**

   * **Parametric**: Just the LLM, no retrieval (SimpleQA-Verified as intended).
   * **RAG**: Same questions, but answered by your RAG stack using the corpus you just built.

That gives you a very clean story:

* “We match the industry standard SimpleQA-Verified parametric scores”
* “…and here is how much our RAG stack improves factuality on the same benchmark.”

If you’d like, next step I can help you design the exact JSON schema + eval script so this drops cleanly into your existing internal “eval runner” without much glue code.

[1]: https://www.ivanzhou.me/reading-notes/2024/11/5/simpleqa-build-high-quality-benchmark-dataset-with-llm-human?utm_source=chatgpt.com "SimpleQA - Build High Quality Benchmark Dataset with LLM + Human — Ivan ..."
[2]: https://huggingface.co/datasets/basicv8vc/SimpleQA "basicv8vc/SimpleQA · Datasets at Hugging Face"
[3]: https://arxiv.org/pdf/2509.07968?utm_source=chatgpt.com "SimpleQA Verified: A Reliable Factuality Benchmark to Measure ..."
[4]: https://www.kaggle.com/datasets/deepmind/simpleqa-verified?utm_source=chatgpt.com "SimpleQA Verified - Kaggle"
[5]: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py?utm_source=chatgpt.com "simple-evals/simpleqa_eval.py at main · openai/simple-evals"
[6]: https://huggingface.co/datasets/google/simpleqa-verified?utm_source=chatgpt.com "google/simpleqa-verified · Datasets at Hugging Face"

