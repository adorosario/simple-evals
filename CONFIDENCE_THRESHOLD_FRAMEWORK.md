# Quality Benchmark Framework

## Overview

This framework implements the penalty-aware scoring methodology from OpenAI's "Why Language Models Hallucinate" paper (arXiv:2509.04664v1), using the recommended 80% confidence threshold to evaluate RAG providers with a focus on quality over volume.

## Key Problem Addressed

The paper identified that current evaluation systems **penalize uncertainty and reward guessing**, leading to the "I Don't Know" tax where conservative systems are unfairly penalized. Our quality benchmark addresses this by:

- **Quality Strategy**: Rewards appropriate uncertainty and calibration
- **Penalty-Aware Scoring**: Penalizes overconfident incorrect responses
- **Conservative Threshold**: Uses 80% confidence as recommended in the paper

## Quality Evaluation Framework

### Core Concept

The framework evaluates provider responses using **post-hoc penalty-aware analysis** with the research-recommended 80% confidence threshold. Providers give natural responses to questions, which are then evaluated against quality-focused criteria:

- **Providers**: Respond naturally to questions without threshold instructions
- **Evaluation**: Judge assesses responses against 80% confidence criteria post-hoc
- **Scoring**: Apply penalty-aware scoring (Correct=+1, Wrong=-4, IDK=0)

### Methodology Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Confidence Threshold** | 80% | Recommended by OpenAI research |
| **Penalty Ratio** | 4.0 | Balanced penalty for 80% threshold |
| **Strategy** | Conservative | Quality-focused approach |

### Scoring Systems Comparison

1. **Volume Strategy** (Traditional)
   - Correct: +1
   - Wrong: 0
   - IDK: 0
   - *Problem: Rewards guessing, penalizes uncertainty*

2. **Quality Strategy** (Penalty-Aware) ✅ **Used**
   - Correct: +1
   - Wrong: -4
   - IDK: 0
   - *Advantage: Rewards appropriate uncertainty, penalizes overconfidence*

## Usage

### Basic Usage

```bash
# Debug run with 5 examples
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Full evaluation with 20 examples
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 20

# Dry run to validate configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --dry-run
```

### Advanced Options

```bash
# Custom configuration with increased parallelism
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py \
  --examples 50 \
  --max-workers 8 \
  --output-dir custom_results
```

## Output Metrics

### Core Metrics

- **🏆 Quality Score**: Penalty-aware score (primary ranking metric)
- **📊 Volume Score**: Traditional binary accuracy (for comparison)
- **📈 Attempted Rate**: Percentage of questions attempted (not abstained)
- **✅ Success Rate**: Success rate on questions actually answered
- **🚫 Abstention Rate**: Percentage of questions where provider said "I don't know"
- **⚠️ Overconfidence Penalty**: Count of wrong answers (costly under penalty-aware scoring)

### Provider Analysis

- **Quality vs Volume Comparison**: Direct comparison of scoring methodologies
- **Strategy Assessment**: Quality-focused, Volume-focused, or Balanced
- **Provider Ranking**: Leaderboard sorted by Quality Score
- **Statistical Significance**: Pairwise provider comparisons with effect sizes

## Implementation Architecture

### Components

1. **ConfidenceThresholdSimpleQAEval**: Quality evaluation class with 80% confidence threshold
2. **confidence_threshold_benchmark.py**: Main runner script for provider quality comparison
3. **Enhanced Grader Template**: Context-aware judging with 80% confidence criteria
4. **Quality-Focused Evaluator**: Single-threshold evaluation optimized for performance

### Evaluation Flow

1. **Provider Response Collection**: Each provider called once per question with clean prompts
2. **Quality Assessment**: Natural responses evaluated against 80% confidence criteria
3. **Confidence Signal Analysis**: Judge assesses appropriate certainty/uncertainty in responses
4. **Penalty-Aware Scoring**: Quality strategy applied (Correct=+1, Wrong=-4, IDK=0)

### Key Configuration

```python
@dataclass
class ConfidenceThreshold:
    threshold: float      # 0.8 (80% confidence)
    penalty_ratio: float  # 4.0 (penalty ratio)
    name: str            # "Conservative"
```

## Theoretical Foundation

### OpenAI Paper Insights

1. **Binary Grading Problem**: Current evaluations reward guessing over uncertainty
2. **Statistical Reduction**: Hallucinations arise from supervised learning challenges
3. **Socio-Technical Solution**: Modify scoring rather than just adding hallucination evals
4. **Explicit Confidence Targets**: Clear penalty ratios in instructions

### Our Implementation

- **Post-Hoc Evaluation**: Providers give natural responses, evaluated against 80% threshold afterward
- **Single-Threshold Focus**: Streamlined evaluation using research-recommended 80% confidence
- **Penalty-Aware Scoring**: Wrong answers penalized at 4:1 ratio relative to correct answers
- **Natural Confidence Signals**: Judge evaluates inherent certainty/uncertainty in responses
- **Provider-Focused Analysis**: Direct comparison of RAG providers using quality methodology
- **Quality vs Volume Strategy**: Clear demonstration of penalty-aware benefits

## Expected Insights

### Quality vs Volume Strategy Comparison

- **Quality-Focused Providers**: Higher quality scores, appropriate abstention behavior
- **Volume-Focused Providers**: Higher volume scores, potential overconfidence penalties
- **Balanced Providers**: Similar quality and volume scores with optimal calibration

### Use Case Guidance

- **High-Stakes Applications**: Quality strategy recommended (penalizes wrong answers)
- **Volume Applications**: Traditional volume strategy may still be appropriate
- **Quality-Critical Systems**: 80% confidence threshold provides conservative evaluation

## Integration with Existing Pipeline

The framework is designed to enhance the existing benchmark with quality-focused evaluation:

- **Audit Logging**: Full compatibility with existing audit trail system
- **Provider-Focused Reports**: Enhanced leaderboard with quality vs volume analysis
- **JSON Output**: Structured results for further analysis
- **HTML Reports**: Clean, provider-focused visual analysis

## Performance Improvements

1. **~67% Faster Execution**: Single threshold eliminates redundant evaluations
2. **Increased Parallelism**: Default max_workers increased from 3 to 8
3. **Simplified Data Structure**: Streamlined results without threshold dimension
4. **Cleaner Statistical Analysis**: Direct provider-vs-provider comparisons

## Files

- `confidence_threshold_simpleqa_eval.py`: Core quality evaluation framework
- `scripts/confidence_threshold_benchmark.py`: Main quality benchmark runner
- `CONFIDENCE_THRESHOLD_FRAMEWORK.md`: This documentation (now quality-focused)
- Generated reports in `results/run_*/quality_benchmark_report_*.html`

## Research Impact

This implementation provides:

1. **Empirical Validation**: Testing OpenAI's penalty-aware scoring on real RAG systems
2. **Practical Guidelines**: Concrete implementation of 80% confidence threshold recommendation
3. **Industry Standard**: Quality-focused evaluation methodology for RAG providers
4. **Performance Optimization**: Streamlined implementation without multi-threshold overhead
5. **Provider-Focused Analysis**: Clear comparison of RAG systems using quality strategy