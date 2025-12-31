# Appendix D: Reproducibility

**Run ID:** `20251214_152848_133`
**Framework Version:** 3.0.0

---

## D.1 Environment Requirements

### Software Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Docker | 26.1.3+ | Container runtime |
| Docker Compose | 2.27.0+ | Multi-container orchestration |
| Python | 3.11+ | (inside container) |
| OpenAI SDK | 1.0+ | API client |
| Google GenAI SDK | Latest | Gemini API client |

### API Keys Required

```bash
# Required for all evaluations
OPENAI_API_KEY=sk-...              # GPT-5.1 and judge

# Required for OpenAI RAG provider
OPENAI_VECTOR_STORE_ID=vs_...      # Pre-built vector store

# Required for CustomGPT RAG provider
CUSTOMGPT_API_KEY=...              # CustomGPT API
CUSTOMGPT_PROJECT=88141            # Project ID

# Required for Gemini RAG provider
GEMINI_API_KEY=...                 # Google AI API
GOOGLE_FILE_SEARCH_STORE_NAME=...  # File Search store
```

---

## D.2 Quick Start Reproduction

### Step 1: Clone Repository

```bash
git clone https://github.com/customgpt/simple-evals.git
cd simple-evals
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 3: Run Benchmark

```bash
# Debug run (5 questions, fast)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Standard run (100 questions)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100

# Full benchmark (1000 questions)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 1000
```

### Step 4: View Results

Results are saved to `results/run_YYYYMMDD_HHMMSS_mmm/`:

```bash
# Open main dashboard
open results/run_*/index.html

# View JSON results
cat results/run_*/quality_benchmark_results.json | jq .
```

---

## D.3 Exact Reproduction of Run 20251214_152848_133

### Random Seeds Used

| Component | Seed | Purpose |
|-----------|------|---------|
| Question sampling | 42 | Reproducible question selection |
| Question shuffling | 42 | Reproducible ordering |
| Judge model | 42 | Deterministic grading |

### Command Used

```bash
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py \
  --examples 100 \
  --max-workers 8
```

### Expected Output

| Metric | Expected Value |
|--------|----------------|
| Total evaluations | 400 |
| CustomGPT correct | 87 |
| CustomGPT incorrect | 2 |
| CustomGPT abstained | 11 |
| OpenAI RAG correct | 89 |
| OpenAI RAG incorrect | 11 |
| Gemini RAG correct | 90 |
| Gemini RAG incorrect | 5 |
| Vanilla correct | 22 |
| Vanilla incorrect | 36 |

**Note:** API versions change over time. Results may vary slightly with different model versions.

---

## D.4 Configuration Files

### docker-compose.yml

```yaml
services:
  simple-evals:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
      - ./results:/app/results
    env_file:
      - .env
    working_dir: /app
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest"]
```

### requirements.txt (key dependencies)

```
openai>=1.0.0
google-generativeai>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
requests>=2.28.0
python-dotenv>=1.0.0
```

---

## D.5 Dataset Verification

### SimpleQA-Verified Dataset

**Location:** `simpleqa-verified/simpleqa_verified.csv`

**Schema:**
```csv
original_index,problem,answer,topic,answer_type,urls
0,"Which district did Mary Ann Arty serve...","165th District",politics,entity,"url1,url2"
```

**Verification Command:**
```bash
docker compose run --rm simple-evals python -c "
import pandas as pd
df = pd.read_csv('simpleqa-verified/simpleqa_verified.csv')
print(f'Total questions: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Sample:\n{df.head(2)}')"
```

**Expected Output:**
```
Total questions: 1000
Columns: ['original_index', 'problem', 'answer', 'topic', 'answer_type', 'urls']
```

---

## D.6 Knowledge Base Setup

### OpenAI Vector Store

1. Download knowledge base files:
```bash
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py
```

2. Upload to OpenAI:
```bash
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py \
  knowledge_base_full \
  --store-name "SimpleQA-Verified-KB"
```

3. Note the vector store ID and add to `.env`:
```bash
OPENAI_VECTOR_STORE_ID=vs_xxxxxxxxxxxxxxxxxxxx
```

### CustomGPT Project

1. Create project at app.customgpt.ai
2. Upload knowledge base files
3. Note project ID and add to `.env`

### Google Gemini File Search

1. Create File Search store via Google AI Studio
2. Upload knowledge base files
3. Note store name and add to `.env`

---

## D.7 Validation Checks

### Check 1: Environment Verification

```bash
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --dry-run
```

Expected: "✓ All providers configured correctly"

### Check 2: API Connectivity

```bash
docker compose run --rm simple-evals python -c "
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-5.1',
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
print('OpenAI:', response.choices[0].message.content)"
```

### Check 3: Dataset Integrity

```bash
docker compose run --rm simple-evals python -c "
import pandas as pd
df = pd.read_csv('simpleqa-verified/simpleqa_verified.csv')
assert len(df) >= 1000, 'Dataset incomplete'
assert 'problem' in df.columns, 'Missing problem column'
assert 'answer' in df.columns, 'Missing answer column'
print('✓ Dataset verified')"
```

---

## D.8 Troubleshooting

### Issue: API Rate Limits

**Symptom:** `RateLimitError` during evaluation

**Solution:**
```bash
# Reduce parallelism
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py \
  --max-workers 4
```

### Issue: Vector Store Not Found

**Symptom:** `NotFoundError: vector store not found`

**Solution:**
1. Verify store ID in `.env`
2. Ensure store is in same OpenAI organization
3. Check store status at platform.openai.com

### Issue: CustomGPT Timeout

**Symptom:** `TimeoutError` on CustomGPT requests

**Solution:**
- CustomGPT has 5 concurrent request limit
- Reduce max_workers or add retry logic

---

## D.9 Version Tracking

### Model Versions (December 2025)

| Provider | Model | Version |
|----------|-------|---------|
| OpenAI | gpt-5.1 | 2025-12-14 snapshot |
| OpenAI | gpt-5-nano | 2025-12-14 snapshot |
| Google | gemini-3-pro-preview | 2025-12 |
| CustomGPT | gpt-5.1 (via CustomGPT) | 2025-12-14 |

**Important:** Model behavior may change with API updates. For exact reproduction, note the API access date.

### Framework Commits

```bash
git log -1 --format="%H %s"
# c2d9276 docs: Complete documentation overhaul for v3.0.0
```

---

*Appendix D: Reproducibility | Framework Version 3.0.0*
