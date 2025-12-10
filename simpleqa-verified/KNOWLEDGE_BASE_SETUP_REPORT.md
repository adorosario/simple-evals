# SimpleQA Knowledge Base Setup Report

**Date:** December 10, 2025
**Author:** Claude Code (automated)
**Status:** Complete and Verified

---

## Executive Summary

This report documents the setup, verification, and validation of the SimpleQA knowledge base across two RAG (Retrieval-Augmented Generation) providers: OpenAI Vector Store and CustomGPT.ai. The goal was to create identical, clean knowledge bases for fair RAG benchmark comparisons.

**Final Result:** 1,000 verified knowledge base files successfully indexed in both systems with perfect 1-to-1 alignment confirmed through rigorous file-by-file verification.

---

## 1. Knowledge Base Overview

### Source Data
- **Location:** `knowledge_base_verified/`
- **File Count:** 1,000 files
- **Naming Convention:** `verified_0001.txt` through `verified_1000.txt`
- **File Sizes:** Range from 2.9 KB to 1.77 MB
- **Total Size:** ~185 MB
- **Build Date:** December 8, 2025

### Target Systems

| System | Identifier | Purpose |
|--------|------------|---------|
| OpenAI Vector Store | `vs_6938645787788191bcff16ba2f298d45` | OpenAI RAG provider |
| CustomGPT.ai Project | `88141` (SimpleQA-Verified-KB-v1) | CustomGPT RAG provider |

---

## 2. Upload Process

### 2.1 OpenAI Vector Store

**Method:** Individual file uploads via OpenAI Files API + Vector Store attachment

**Process:**
1. Files uploaded individually using `client.files.create(purpose='assistants')`
2. File IDs stored in checkpoint file for recovery
3. Vector store created with `client.vector_stores.create()`
4. Files attached to vector store in batches (500 initial, then 100 at a time)

**Checkpoint File:** `checkpoints/upload_1765294410.checkpoint`
- Contains file paths, OpenAI file IDs, upload timestamps
- Enables recovery from interrupted uploads

**Script:** `scripts/robust_upload_knowledge_base.py`

**Key Decisions:**
- Individual file uploads (not merged) to preserve 1-to-1 filename mapping
- Checkpoint-based recovery to handle rate limits and interruptions
- 500 file limit per vector store creation call (OpenAI API constraint)

### 2.2 CustomGPT.ai

**Method:** File uploads via CustomGPT REST API

**Process:**
1. Project created: `SimpleQA-Verified-KB-v1` (ID: 88141)
2. Files uploaded via `/projects/{id}/sources/upload` endpoint
3. CustomGPT automatically indexes uploaded files

**Script:** `scripts/upload_kb_to_customgpt.py`

**Issue Encountered:** Multiple upload runs created 2,980 total pages (3x duplicates)
- Root cause: Script run multiple times with different `--start-from` flags
- Resolution: Cleanup script deleted 1,980 duplicates, leaving exactly 1,000 unique files

**Cleanup Script:** `scripts/cleanup_customgpt_duplicates.py`

---

## 3. Data Quality Issues Encountered

### 3.1 CustomGPT Duplicate Files

**Problem:** 2,980 pages existed instead of 1,000 (each file uploaded ~3 times)

**Analysis:**
- `customgpt_upload_results.json` showed only 990 files (misleading)
- Multiple upload sessions with `--start-from` flag
- No deduplication in CustomGPT API

**Resolution:**
1. Grouped pages by filename
2. Kept oldest successfully indexed copy of each file
3. Deleted all duplicates (1,977 ok + 3 failed = 1,980 deleted)

**Lessons Learned:**
- CustomGPT lacks built-in deduplication
- Need checkpoint/resume capability for CustomGPT uploads
- Always verify final state, not just upload results

### 3.2 Failed Indexing (CustomGPT)

**Problem:** 3 files initially failed indexing in CustomGPT

**Files Affected:**
- `verified_0020.txt` (841 KB)
- `verified_0925.txt` (377 KB)
- `verified_0942.txt` (580 KB)

**Resolution:** These were duplicate copies that got deleted during cleanup. The primary copies were successfully indexed.

---

## 4. Verification Process

### 4.1 Verification Script

**Script:** `scripts/verify_kb_alignment.py`

**Capabilities:**
- Collects filenames from all 3 sources (local KB, OpenAI, CustomGPT)
- Performs set comparison to find exact mismatches
- Auto-fix capability for missing files and orphans
- Generates audit trail in JSON format

### 4.2 Verification Results

**Final Verification Run:** December 10, 2025

```
KNOWLEDGE BASE ALIGNMENT VERIFICATION
======================================================================
  KB Directory: knowledge_base_verified
  OpenAI Vector Store: vs_6938645787788191bcff16ba2f298d45
  CustomGPT Project: 88141

STEP 1: Collecting filenames from all sources
--------------------------------------------------
[A] Local Knowledge Base: 1000 files
[B] OpenAI Vector Store: 1000 unique filenames (10 pages)
[C] CustomGPT Project: 1000 unique filenames

STEP 2: Comparing filenames across sources
--------------------------------------------------
  Files in all 3 sources: 1000
  Files in KB only (missing from OpenAI): 0
  Files in KB only (missing from CustomGPT): 0
  Orphans in OpenAI (not in KB): 0
  Orphans in CustomGPT (not in KB): 0

RESULT: PERFECT ALIGNMENT
======================================================================
```

### 4.3 Audit Trail

**File:** `knowledge_base_verified/alignment_audit.json`

```json
{
  "timestamp": "2025-12-10T13:06:49.729196",
  "status": "ALIGNED",
  "kb_files": 1000,
  "openai_files": 1000,
  "customgpt_files": 1000,
  "aligned_count": 1000,
  "mismatches": null
}
```

---

## 5. Architecture Decisions

### 5.1 Individual File Uploads vs. Merged Files

**Decision:** Upload files individually (not merged)

**Rationale:**
- Preserves 1-to-1 filename mapping for verification
- Enables granular debugging of retrieval issues
- Allows future selective updates
- Required for fair benchmark comparison

**Tradeoff:** More API calls, but better data integrity

### 5.2 Checkpoint-Based Recovery

**Decision:** Implement checkpoint files for upload sessions

**Rationale:**
- OpenAI rate limits can interrupt long uploads
- 1,000 files takes significant time to upload
- Enables resume from failure point

**Implementation:** Pickle-based checkpoint with file IDs and status

### 5.3 Auto-Fix vs. Manual Intervention

**Decision:** Implement auto-fix for mismatches

**Rationale:**
- Reduces manual intervention for common issues
- Faster iteration during development
- Can be disabled with `--dry-run` flag

**Auto-fix capabilities:**
- Re-upload missing files to OpenAI
- Re-upload missing files to CustomGPT
- Delete orphan files from OpenAI
- Delete orphan pages from CustomGPT

---

## 6. Configuration

### Environment Variables Required

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_VECTOR_STORE_ID=vs_6938645787788191bcff16ba2f298d45

# CustomGPT
CUSTOMGPT_API_KEY=9002|...
CUSTOMGPT_PROJECT=88141
```

### Key Files

| File | Purpose |
|------|---------|
| `knowledge_base_verified/` | Source knowledge base files |
| `knowledge_base_verified/build_manifest.json` | KB build metadata |
| `knowledge_base_verified/alignment_audit.json` | Verification audit trail |
| `checkpoints/upload_1765294410.checkpoint` | OpenAI upload checkpoint |
| `scripts/verify_kb_alignment.py` | Verification script |
| `scripts/cleanup_customgpt_duplicates.py` | Duplicate cleanup script |
| `scripts/upload_kb_to_customgpt.py` | CustomGPT upload script |

---

## 7. Reproducing This Setup

### Step 1: Upload to OpenAI Vector Store

```bash
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py \
  knowledge_base_verified --store-name "SimpleQA-Verified-KB-v1"
```

### Step 2: Upload to CustomGPT

```bash
docker compose run --rm simple-evals python scripts/upload_kb_to_customgpt.py
```

### Step 3: Verify Alignment

```bash
docker compose run --rm simple-evals python scripts/verify_kb_alignment.py
```

### Step 4: Clean Up Duplicates (if needed)

```bash
docker compose run --rm simple-evals python scripts/cleanup_customgpt_duplicates.py --dry-run
docker compose run --rm simple-evals python scripts/cleanup_customgpt_duplicates.py
```

---

## 8. Known Limitations

### 8.1 OpenAI Vector Store

- Maximum 500 file IDs per `vector_stores.create()` call
- Pagination required for listing files (100 per page)
- File retrieval needed to get actual filename (not included in vector store file list)

### 8.2 CustomGPT.ai

- No built-in deduplication
- No checkpoint/resume API for uploads
- Pagination via page numbers (not cursors)
- Sources endpoint returns different structure than expected (uploads.pages vs data.data)

### 8.3 General

- Large files (>1MB) may have longer indexing times
- Both systems have rate limits that require careful handling
- Verification requires ~10 API calls per system (1000 files / 100 per page)

---

## 9. Future Improvements

1. **Add content hash verification** - Compare MD5/SHA256 of source files with indexed content
2. **Implement CustomGPT checkpoint** - Enable resume capability for CustomGPT uploads
3. **Add file size validation** - Verify indexed file sizes match source
4. **Automate periodic verification** - Cron job to detect drift
5. **Add third provider** - Google Vertex AI for three-way comparison

---

## 10. Conclusion

The SimpleQA knowledge base has been successfully deployed to both OpenAI Vector Store and CustomGPT.ai with verified 1-to-1 file alignment. The rigorous verification process ensures:

- **Data Integrity:** All 1,000 source files are indexed in both systems
- **No Duplicates:** Exactly one copy of each file exists
- **No Orphans:** No extra files exist in either system
- **Reproducibility:** Process is documented and scriptable
- **Auditability:** Full audit trail preserved

The knowledge bases are ready for fair RAG benchmark comparisons.

---

## Appendix A: File Inventory

Total files: 1,000

| Range | Count | Example |
|-------|-------|---------|
| verified_0001.txt - verified_0100.txt | 100 | verified_0001.txt |
| verified_0101.txt - verified_0200.txt | 100 | verified_0150.txt |
| verified_0201.txt - verified_0300.txt | 100 | verified_0250.txt |
| verified_0301.txt - verified_0400.txt | 100 | verified_0350.txt |
| verified_0401.txt - verified_0500.txt | 100 | verified_0450.txt |
| verified_0501.txt - verified_0600.txt | 100 | verified_0550.txt |
| verified_0601.txt - verified_0700.txt | 100 | verified_0650.txt |
| verified_0701.txt - verified_0800.txt | 100 | verified_0750.txt |
| verified_0801.txt - verified_0900.txt | 100 | verified_0850.txt |
| verified_0901.txt - verified_1000.txt | 100 | verified_0950.txt |

---

## Appendix B: API Reference

### OpenAI Vector Store API

```python
# Create file
file = client.files.create(file=open(path, 'rb'), purpose='assistants')

# Create vector store
vs = client.vector_stores.create(name="...", file_ids=[...])

# Add file to vector store
client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)

# List files in vector store
client.vector_stores.files.list(vector_store_id, limit=100, after=cursor)

# Get file details
client.files.retrieve(file_id)
```

### CustomGPT API

```python
# Upload file
requests.post(
    f"https://app.customgpt.ai/api/v1/projects/{project_id}/sources/upload",
    headers={"Authorization": f"Bearer {api_key}"},
    files={"file": open(path, 'rb')}
)

# List sources
requests.get(
    f"https://app.customgpt.ai/api/v1/projects/{project_id}/sources",
    headers={"Authorization": f"Bearer {api_key}"}
)

# Delete page
requests.delete(
    f"https://app.customgpt.ai/api/v1/projects/{project_id}/pages/{page_id}",
    headers={"Authorization": f"Bearer {api_key}"}
)
```

---

*End of Report*
