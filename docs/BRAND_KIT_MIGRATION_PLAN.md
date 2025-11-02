# Brand Kit Migration Plan - Master Action Plan

## Executive Summary

**Goal**: Apply unified, Apple-inspired brand kit to ALL 300+ HTML files across the system.

**Why**: Currently each HTML file uses different inline styles, making the system look unprofessional and inconsistent. Users see different designs for each report type.

**Solution**: Systematically update all HTML generators to use `brand_kit.py` components.

---

## Current Status

### ✅ Already Using Brand Kit
1. **Main Dashboard** (`generate_main_dashboard.py`)
   - Generates: `index.html`
   - Status: Using brand kit ✅

2. **Quality Benchmark Reports** (`report_generators.py`)
   - Generates: `quality_benchmark_report_*.html`
   - Status: Using `generate_quality_benchmark_report_v2()` with brand kit ✅
   - Functions: `generate_statistical_analysis_report_v2()`, `generate_forensic_dashboard_v2()`

### ❌ NOT Using Brand Kit (CRITICAL)
3. **Forensic Reports** (`generate_forensic_reports.py`)
   - Generates: 270+ HTML files:
     - `forensic_dashboard.html` (10 instances)
     - `forensic_question_simpleqa_*.html` (262 instances)
     - `customgpt_engineering_report.html`
   - Status: Uses inline styles (lines 211-227)
   - **Impact**: Largest volume of inconsistent files
   - **Priority**: P0 - CRITICAL

4. **Universal Forensics** (`generate_universal_forensics.py`)
   - Generates: Provider-specific forensic analysis
   - Status: Unknown, needs inspection
   - **Priority**: P1 - HIGH

5. **Statistical Analysis** (`academic_statistical_analysis.py`)
   - Generates: `statistical_analysis_*.html`
   - Status: May use inline HTML
   - **Priority**: P1 - HIGH

6. **Flex Tier Comparison** (`flex_tier_comparison.py`)
   - Generates: `flex_tier_comparison_*.html`
   - Status: Has `_generate_html_report()` method with likely inline styles
   - **Priority**: P1 - HIGH

### Lower Priority (Legacy/Specialized)
7. **Multi-Provider Benchmark** (`multi_provider_benchmark.py`)
   - Status: Legacy v1.x
   - **Priority**: P2 - MEDIUM

8. **RAG Benchmark** (`rag_benchmark.py`)
   - Status: May be legacy
   - **Priority**: P2 - MEDIUM

9. **Blog Post Generators** (Various)
   - `create_beautiful_html_blog.py`
   - `create_narrative_blog_post.py`
   - `generate_adaptive_blog_post.py`
   - `generate_html_blog_post.py`
   - Status: Marketing content, may keep custom styles
   - **Priority**: P3 - LOW

---

## Action Plan - Systematic Updates

### Phase 1: Core Forensic System (P0 - IMMEDIATE)

#### Task 1.1: Update `generate_forensic_reports.py`
**Files Affected**: 270+ HTML files

**Changes Required**:
1. ✅ Import brand kit components (DONE)
2. ❌ Remove `get_html_template()` function (replace with brand kit)
3. ❌ Update `generate_forensic_dashboard()`:
   - Use `get_html_head()` for header
   - Use `get_navigation_bar(active_page='forensic')` for nav
   - Use `get_page_header()` for page title
   - Use brand kit CSS classes for cards/metrics
   - Add `get_datatable_script()` for tables
4. ❌ Update `generate_individual_question_report()`:
   - Same brand kit integration
   - Consistent navigation back to dashboard
5. ❌ Update `convert_engineering_report_to_html()`:
   - Use brand kit wrapper
6. ❌ Test with sample data

**Validation**:
- All 270+ forensic HTML files use consistent styling
- Navigation works between dashboard ↔ individual questions
- Links to quality benchmark and main dashboard work

---

### Phase 2: Analysis Reports (P1 - HIGH)

#### Task 2.1: Update `generate_universal_forensics.py`
**Inspection Required**: Check if this overlaps with `generate_forensic_reports.py` or serves different purpose

**Changes Required**:
1. Inspect current HTML generation
2. Import brand kit if generating HTML
3. Replace inline styles with brand kit components
4. Test integration

#### Task 2.2: Update `academic_statistical_analysis.py`
**Files Affected**: `statistical_analysis_*.html`

**Current State**: Has `generate_statistical_analysis_report()` function

**Changes Required**:
1. Check if currently generates HTML or just markdown/latex
2. If generates HTML, replace with `report_generators.generate_statistical_analysis_report_v2()`
3. OR update inline HTML generation to use brand kit
4. Test statistical reports

#### Task 2.3: Update `flex_tier_comparison.py`
**Files Affected**: `flex_tier_comparison_*.html`

**Current State**: Has `_generate_html_report()` method

**Changes Required**:
1. Import brand kit components
2. Replace `_generate_html_report()` to use brand kit
3. Ensure consistent styling with other reports
4. Test flex tier comparison reports

---

### Phase 3: Legacy/Other (P2-P3 - MEDIUM/LOW)

#### Task 3.1: Audit Individual Eval Modules
**Files**: `*_eval.py` modules (simpleqa_eval, mmlu_eval, etc.)

**Action**:
1. Check if they generate standalone HTML
2. If yes, update to use brand kit
3. If no, skip

#### Task 3.2: Legacy Benchmarks
- `multi_provider_benchmark.py`
- `rag_benchmark.py`

**Action**:
1. Determine if still in use
2. If yes, update to brand kit
3. If deprecated, document for removal

#### Task 3.3: Blog Post Generators
**Action**:
1. Review if these should use brand kit or keep custom styling
2. Decision: Likely keep custom styling since they're marketing content
3. Document decision

---

## Internal Linking Requirements

**All reports MUST link correctly to**:
1. **Home/Dashboard**:
   - `../index.html` or `index.html` depending on depth
   - Use: `get_navigation_bar(active_page='home')`

2. **Quality Benchmark**:
   - Find file: `quality_benchmark_report_*.html` in run dir
   - Link: Relative path
   - Use: `get_navigation_bar(active_page='quality')`

3. **Statistical Analysis**:
   - Find file: `statistical_analysis_run_*.html`
   - Link: Relative path
   - Use: `get_navigation_bar(active_page='statistical')`

4. **Forensics**:
   - Dashboard: `forensic_dashboard.html`
   - Individual: `forensic_question_*.html`
   - Use: `get_navigation_bar(active_page='forensic')`

**Navigation Component Usage**:
```python
from brand_kit import get_navigation_bar

# In forensic dashboard
nav_html = get_navigation_bar(active_page='forensic', run_id=run_id)

# In quality benchmark
nav_html = get_navigation_bar(active_page='quality', run_id=run_id)
```

---

## Testing Plan

### Unit Tests
1. Verify each generator imports brand kit correctly
2. Verify HTML output includes brand kit CSS
3. Verify navigation bar appears in all reports

### Integration Tests
1. Run full benchmark: `docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug`
2. Check all generated HTML files
3. Verify:
   - Consistent styling across all files
   - All internal links work
   - Navigation works between all report types
   - Mobile responsive
   - No console errors in browser

### Visual Inspection
1. Open `index.html` in browser
2. Navigate to each sub-report
3. Verify:
   - Consistent header/nav
   - Consistent colors/typography
   - Smooth animations
   - Apple-like polish

---

## Success Criteria

✅ **Code Quality**
- All generators import brand kit components
- No inline `<style>` tags except brand kit CSS
- Consistent use of brand kit functions across all generators

✅ **Visual Consistency**
- All HTML files look like they're from the same system
- Same navigation bar on all pages
- Same color scheme, typography, spacing
- Same hover effects and animations

✅ **Functionality**
- All internal links work
- Navigation between reports works seamlessly
- DataTables work on all tables
- Responsive on mobile/desktop

✅ **User Experience**
- Apple-like polish and attention to detail
- Fast, smooth animations
- Clear visual hierarchy
- Easy to navigate between reports

---

## Timeline

**Phase 1** (P0 - Forensic System):
- Estimated: 2-3 hours
- Must complete before moving to Phase 2

**Phase 2** (P1 - Analysis Reports):
- Estimated: 2-3 hours
- Can parallelize if multiple developers

**Phase 3** (P2-P3 - Legacy/Other):
- Estimated: 1-2 hours
- Can be done incrementally

**Total**: ~5-8 hours for complete migration

---

## Risk Mitigation

**Risk**: Breaking existing functionality
- **Mitigation**: Test each generator after update with `--debug` mode

**Risk**: Missing edge cases in linking
- **Mitigation**: Comprehensive link audit script

**Risk**: Inconsistent behavior between report types
- **Mitigation**: Centralized brand kit ensures consistency

---

## Deliverables

1. ✅ Updated `brand_kit.py` with Apple-inspired design
2. ✅ Catalog of all HTML report types
3. ✅ This migration plan
4. ❌ Updated `generate_forensic_reports.py`
5. ❌ Updated `generate_universal_forensics.py`
6. ❌ Updated `academic_statistical_analysis.py`
7. ❌ Updated `flex_tier_comparison.py`
8. ❌ Integration tests passing
9. ❌ Documentation of unified system

---

Generated: 2025-10-25
Status: Phase 1 In Progress (generate_forensic_reports.py)
