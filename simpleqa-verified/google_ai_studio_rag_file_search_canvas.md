# Google AI Studio / Gemini File Search RAG Canvas

A working blueprint for building a **high-accuracy, fully Google-managed RAG** stack using the **Gemini API + File Search tool**, suitable for a multi-tenant SaaS like CustomGPT and for running the SimpleQA Verified benchmark with Google's RAG.

---
## 0. Architecture at a Glance

**Goal:** Minimal plumbing, maximum accuracy.

- **Model:** `gemini-3-pro` (or `gemini-2.5-pro` if you prefer GA / more stable pricing).
- **RAG engine:** Gemini **File Search tool** (managed chunking, embeddings, vector search).
- **Tenancy:** Either
  - (A) **Store-per-tenant**, or
  - (B) Single store with `tenant_id` metadata filter.
- **Accuracy controls:**
  - Conservative generation params (`temperature` ≈ 0.2–0.3).
  - Strict grounding prompt ("only answer from docs").
  - Citations surfaced from `grounding_metadata`.

Use this canvas as a checklist + code skeleton for your implementation.

---
## 1. Project & Environment Setup

### 1.1. Create API Key

- Go to **Google AI Studio → Get API key**.
- Restrict it to the project you’ll use in production.

**ENV vars**

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
export GCP_PROJECT_ID="your-gcp-project-id"  # optional if needed elsewhere
```

### 1.2. Install SDKs

**Python**

```bash
pip install -U google-genai
```

**Node.js**

```bash
npm install @google/genai
```

---
## 2. Store Design (Multi-Tenant Friendly)

Decide your tenancy strategy *before* indexing.

### 2.1. Recommended Metadata Schema

Each document in File Search gets `custom_metadata`. For a CustomGPT-style platform, use:

- `tenant_id` – your workspace / customer key (e.g., Stripe customer ID, internal account ID).
- `doc_type` – e.g., `kb`, `faq`, `policy`, `legal`, `api_docs`.
- `lang` – e.g., `en`, `pt`, `fr`.
- (Optional) `product`, `region`, `version`, etc.

You’ll use these in `metadata_filter` for precise retrieval per tenant.

### 2.2. Store Layout Options

**Option A: One File Search store per tenant**

- ✅ Simple isolation.
- ✅ Easy to delete everything for a tenant.
- ❌ More stores to manage; may hit quotas if you have many tenants.

**Option B: One big store with `tenant_id` metadata filter**

- ✅ Fewer stores to manage, better for thousands of tenants.
- ❌ Need to be very disciplined with `metadata_filter` on every query.

**Recommendation:**

- For **< 100 tenants** or **high-value enterprise**: Option A.
- For mass market, small tenants: Option B (+ strong tenant_id filtering in your backend).

Define constants in your backend:

```text
FILE_SEARCH_STORE_NAME_PREFIX = "fileSearchStores/customgpt-"
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80
```

---
## 3. Indexing Pipeline (Ingestion → Chunking → Store)

You’ll likely have a background worker that ingests/updates tenant data.

### 3.1. Create / Ensure a File Search Store

**Python skeleton**

```python
from google import genai

client = genai.Client()

def ensure_file_search_store(display_name: str) -> str:
    # Try to find an existing store by display name (pseudo-code: you may need to list & filter)
    # Here we just create and return the name; in production you’d cache/store it per tenant.
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    return store.name  # e.g. "fileSearchStores/support-kb-prod-123"
```

Usage (Option A – store-per-tenant):

```python
store_name = ensure_file_search_store(f"customgpt-tenant-{tenant_id}")
```

Or (Option B – shared store):

```python
STORE_NAME = "fileSearchStores/customgpt-shared-prod"
```

### 3.2. Chunking Configuration (Accuracy-Focused)

Recommended settings for general KB/FAQ/API docs:

- `max_tokens_per_chunk`: **400**
- `max_overlap_tokens`: **80** (≈ 20% overlap)

This balances:

- Enough context inside each chunk.
- Limited duplication so the index doesn’t explode.

### 3.3. Index a File with Metadata

**Python ingestion skeleton**

```python
import time
from google.genai import types
from google import genai

client = genai.Client()

CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80


def index_file(path: str, store_name: str, *, tenant_id: str,
               doc_type: str, lang: str = "en") -> None:
    """Upload a file, then import it into File Search with metadata + chunking.
    """
    # 1) Upload raw file (this returns a File resource used by File Search)
    uploaded = client.files.upload(
        file=path,
        config={"name": path},  # used in citations
    )

    # 2) Import into File Search store
    op = client.file_search_stores.import_file(
        file_search_store_name=store_name,
        file_name=uploaded.name,
        custom_metadata=[
            {"key": "tenant_id", "string_value": tenant_id},
            {"key": "doc_type",  "string_value": doc_type},
            {"key": "lang",      "string_value": lang},
        ],
        config={
            "chunking_config": {
                "white_space_config": {
                    "max_tokens_per_chunk": CHUNK_TOKENS,
                    "max_overlap_tokens":  CHUNK_OVERLAP,
                }
            }
        },
    )

    # 3) Poll operation until done (or integrate with your job framework)
    while not op.done:
        time.sleep(5)
        op = client.operations.get(op)
```

### 3.4. Indexing TODOs

- [ ] Implement a worker that:
  - [ ] Watches for new/updated content (website, docs, KB, etc.).
  - [ ] Downloads / normalizes it into files.
  - [ ] Calls `index_file` with correct metadata.
- [ ] Decide on an **update strategy**: re-import entire file vs partial updates.
- [ ] Implement **delete / reindex** for tenants leaving the platform or major doc revamps.

---
## 4. Query Pipeline (RAG Call via API)

You want a clean `/answer` API in your backend that:

1. Maps the caller to a `tenant_id`.
2. Calls Gemini with File Search tool configured.
3. Returns:
   - Final answer text.
   - Citations & metadata for your UI.

### 4.1. System Prompt Template (Strict Grounding)

Use something along these lines as a system instruction:

```text
You are a support assistant for our product.
You must answer ONLY using the context retrieved by the File Search tool.
If the answer is not clearly supported by the documentation, say:
"I don’t know based on the available documentation."
Always be concise and include short natural-language references to the source documents.
```

You can prepend this as a system message, then send the user question as a user message.

### 4.2. Recommended Generation Parameters

For high factual accuracy over creativity:

- `temperature`: `0.0` (use 0 for SimpleQA Verified / maximum determinism) (low randomness)
- `top_p`: `0.8`
- `top_k`: `40`
- `max_output_tokens`: `1024`
- `candidate_count`: `1`

You can tune these later with an evaluation set.

### 4.3. Python: Single `answer_question` Helper

```python
from google import genai
from google.genai import types

client = genai.Client()

MODEL_NAME = "gemini-2.5-pro"  # for SimpleQA Verified; you can also try "gemini-3-pro"

SYSTEM_PROMPT = """You are a support assistant for our product.
You must answer ONLY using the context retrieved by the File Search tool.
If the answer is not clearly supported by the documentation, say:
"I don’t know based on the available documentation."
Always be concise and include short natural-language references to the source documents.
""".strip()


def answer_question(question: str, *, store_name: str, tenant_id: str,
                    lang: str = "en"):
    # Filter so we only see this tenant’s docs
    metadata_filter = f"tenant_id={tenant_id} AND lang={lang}"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    f"{SYSTEM_PROMPT}\n\nUser question: {question}"
                )],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.8,
            top_k=40,
            max_output_tokens=1024,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name],
                        metadata_filter=metadata_filter,
                    )
                )
            ],
        ),
    )

    # response.text is the answer
    answer_text = response.text

    # Optional: citations / grounding metadata, if available
    candidate = response.candidates[0]
    grounding = getattr(candidate, "grounding_metadata", None)

    return answer_text, grounding
```

### 4.4. Node.js: Equivalent Helper

```js
import { GoogleGenAI, types } from "@google/genai";

const ai = new GoogleGenAI();
const MODEL_NAME = "gemini-3-pro"; // or "gemini-2.5-pro"

const SYSTEM_PROMPT = `You are a support assistant for our product.
You must answer ONLY using the context retrieved by the File Search tool.
If the answer is not clearly supported by the documentation, say:
"I don’t know based on the available documentation."
Always be concise and include short natural-language references to the source documents.`;

export async function answerQuestion(question, { storeName, tenantId, lang = "en" }) {
  const metadataFilter = `tenant_id=${tenantId} AND lang=${lang}`;

  const response = await ai.models.generateContent({
    model: MODEL_NAME,
    contents: [
      {
        role: "user",
        parts: [
          { text: `${SYSTEM_PROMPT}\n\nUser question: ${question}` },
        ],
      },
    ],
    config: {
      temperature: 0.0,
      topP: 0.8,
      topK: 40,
      maxOutputTokens: 1024,
      tools: [
        {
          fileSearch: {
            fileSearchStoreNames: [storeName],
            metadataFilter,
          },
        },
      ],
    },
  });

  const answerText = response.text;
  const candidate = response.candidates?.[0];
  const grounding = candidate?.groundingMetadata;

  return { answerText, grounding };
}
```

### 4.5. HTTP Endpoint Sketch

**FastAPI-style**

```python
from fastapi import FastAPI, Depends

app = FastAPI()

@app.post("/v1/answer")
async def answer(payload: dict, user = Depends(auth_user)):
    tenant_id = user.tenant_id
    question = payload["question"]

    store_name = get_store_name_for_tenant(tenant_id)  # your logic

    answer, grounding = answer_question(
        question,
        store_name=store_name,
        tenant_id=tenant_id,
    )

    return {
        "answer": answer,
        "citations": serialize_grounding(grounding),
    }
```

You can adapt this to Express.js / Nest / etc.

---
## 5. UX: Surfacing Citations & Chunks

Even though Gemini + File Search manage retrieval, your **UI** should:

- Show **document titles** and maybe a short snippet for each citation.
- Let the user click to open the source doc (if allowed).
- Optionally highlight the cited span in the doc (using offsets from grounding metadata, if exposed).

Backend shape for a single citation (conceptually):

```json
{
  "doc_name": "kb_billing_refunds.pdf",
  "display_name": "Billing – Refund Policy",
  "url": "https://.../kb/billing-refunds",
  "snippet": "Refunds are processed within 5–7 business days..."
}
```

---
## 6. Accuracy & Evaluation Checklist

Create a small **golden set** of real questions and use it to iterate on settings.

### 6.1. Golden Set

- [ ] At least **30–50** real or realistic questions per tenant / domain.
- [ ] For each, store:
  - [ ] Question.
  - [ ] Expected answer or source doc.
  - [ ] Severity (critical vs nice-to-have).

### 6.2. Knobs to Try

- [ ] `max_tokens_per_chunk`: try 300 / 400 / 600.
- [ ] `max_overlap_tokens`: try 40 / 80 / 120.
- [ ] `temperature`: try 0.0 vs 0.2.
- [ ] Strict vs relaxed system prompts (e.g., allow small interpolations or not).

For each config version, run your golden set and manually inspect:

- **Correctness** – answer matches the docs.
- **Groundedness** – answer is supported by cited passages.
- **Recall** – do we miss important parts because chunks are too small or too few?

---
## 7. Optional Extensions

### 7.1. Add World Knowledge (Google Search Grounding)

You can later extend your design by adding a second grounding source (Google Search), but keep the ***contract***:

> Internal docs rule; web grounding is only used to fill gaps and must be labeled as such.

Implementation pattern:

- Add an additional tool configuration for Google Search grounding.
- Update the system prompt to:
  - Prefer File Search docs.
  - Only use web data if docs are silent.
  - Clearly label web-sourced statements.

### 7.2. Different Profiles per Use Case

You might define different RAG profiles:

- **Support / Helpdesk** – strict, low temperature, smaller chunks.
- **Sales / Enablement** – slightly higher temperature, more summarization.
- **Research / Ops** – allow the model to say "I’m not sure" more often.

Each profile can:

- Use the same File Search store.
- Change:
  - System prompt.
  - Generation parameters.
  - Doc filters (e.g., `doc_type IN ("kb", "policy")`).

---
## 8. Implementation TODO Summary

Use this as your action list:

1. **Setup**
   - [ ] Create Gemini API key.
   - [ ] Install `google-genai` / `@google/genai`.

2. **Store Strategy**
   - [ ] Decide per-tenant vs shared store.
   - [ ] Define metadata schema: `tenant_id`, `doc_type`, `lang`, etc.

3. **Indexing Worker**
   - [ ] Implement `ensure_file_search_store`.
   - [ ] Implement `index_file` with chunking config.
   - [ ] Wire your content sources (website, KB, docs) into this worker.

4. **Answer API**
   - [ ] Implement `answer_question` helper (Python / Node).
   - [ ] Implement `/v1/answer` HTTP endpoint.
   - [ ] Add auth → resolve `tenant_id`.

5. **Frontend Integration**
   - [ ] Call `/v1/answer` from your chat UI.
   - [ ] Render answer + citations.
   - [ ] Add loading states, error handling.

6. **Evaluation Loop**
   - [ ] Build golden question set.
   - [ ] Periodically run it on new configs.
   - [ ] Track regression / improvements.

Once you’ve wired this up, you’ll have a **Google-native, File Search–powered RAG backend** that’s easy to extend with more tenants, more docs, and extra grounding sources over time.

---

## 9. SimpleQA Verified Benchmark Harness (Google RAG)

This section specializes the above setup for running the **SimpleQA Verified** benchmark with Google’s best RAG stack.

> Note: The official SimpleQA Verified leaderboard measures parametric knowledge only (no tools / web). Here we intentionally evaluate a different regime: **Gemini + Google Search grounding** (and optionally URL / File Search). Your scores won’t be directly comparable to the paper, but you’ll get a best‑case “Google RAG” number on the same questions.

### 9.1. Dataset hookup

- Download from Hugging Face or Kaggle:
  - Hugging Face: `google/simpleqa-verified` (split `eval`, 1k rows).
  - Kaggle: `deepmind/simpleqa-verified`.
- Columns you care about:
  - `problem` – the question.
  - `answer` – the reference answer text (sometimes includes ranges like `150 (acceptable range: 148–152)`).
  - `urls` – comma‑separated verifying URLs the question is based on.

Example loader (Python):

```python
from datasets import load_dataset

ds = load_dataset("google/simpleqa-verified")["eval"]
```

### 9.2. Recommended model + RAG mode

For this benchmark:

- **Model:** `gemini-2.5-pro` by default (matches the model used in the SimpleQA Verified paper); you can optionally also try `gemini-3-pro` for a "latest model" run.
- **RAG:** Grounding with Google Search via the `google_search` tool.
- **Generation config:**
  - `temperature = 0.0`  ← as requested, for determinism.
  - `top_p = 1.0`
  - `max_output_tokens = 256`
  - `candidate_count = 1`

You still keep File Search for your own docs; for SimpleQA you usually don’t need to index anything, since all answers live on the open web.

### 9.3. SimpleQA‑style answering helper (Python)

```python
from google import genai
from google.genai import types

client = genai.Client()

SIMPLEQA_MODEL = "gemini-2.5-pro"  # try "gemini-3-pro" as a second run

simpleqa_tool = types.Tool(
    google_search=types.GoogleSearch()
)

simpleqa_config = types.GenerateContentConfig(
    temperature=0.0,
    top_p=1.0,
    max_output_tokens=256,
    tools=[simpleqa_tool],
)

SIMPLEQA_SYSTEM_PROMPT = """You are being evaluated on the SimpleQA Verified benchmark.

You MUST use the Google Search tool to look up each answer before responding.
For every question:
- Call the Google Search tool.
- Read the results.
- Return a SHORT answer phrase that directly answers the question.
- Return ONLY the answer text. No explanation, no extra words.

If, after searching, you truly cannot find the answer, reply exactly with: unknown
"""

def answer_simpleqa_question(question: str) -> str:
    response = client.models.generate_content(
        model=SIMPLEQA_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    f"{SIMPLEQA_SYSTEM_PROMPT}

Question: {question}"
                )],
            )
        ],
        config=simpleqa_config,
    )

    return (response.text or "").strip()
```

### 9.4. Running the full benchmark loop

Very bare‑bones loop:

```python
import pandas as pd
from datasets import load_dataset
import time


ds = load_dataset("google/simpleqa-verified")["eval"]

rows = []
for ex in ds:
    q = ex["problem"]
    gold = ex["answer"]
    pred = answer_simpleqa_question(q)
    rows.append({"original_index": ex["original_index"],
                 "question": q,
                 "gold": gold,
                 "pred": pred})
    # Optional: throttle calls a bit
    time.sleep(0.2)


df = pd.DataFrame(rows)
df.to_csv("simpleqa_gemini_rag_predictions.csv", index=False)
```

You can then either:

- Plug this CSV into the official SimpleQA Verified evaluation notebook on Kaggle (they use a GPT‑4.1‑style autorater prompt tuned to numeric ranges and alternative phrasings).
- Or roll your own quick metric (case‑insensitive exact match, simple numeric tolerance, etc.) for quick smoke tests.

### 9.5. Optional “oracle RAG” using dataset URLs

If you want an **upper‑bound** RAG score that ignores retrieval difficulty and only tests reading/comprehension:

- For each example, take its `urls` field.
- Pass those URLs via the URL context tool (or pre‑index them in File Search) so Gemini reads exactly the pages the question was built from.

Conceptual pattern:

```python
url_tool = types.Tool(
    url_context=types.UrlContext(
        allowed_urls=ex["urls"].split(",")  # clean trailing characters if needed
    )
)

# Then use tools=[simpleqa_tool, url_tool] in GenerateContentConfig
```

This will give you a near‑ceiling SimpleQA score, but remember: it’s no longer comparable to the official parametric‑only leaderboard; you’re now evaluating “Google + perfect documents” instead of pure model memory.


