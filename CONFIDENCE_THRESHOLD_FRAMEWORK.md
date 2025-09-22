# Confidence Threshold Framework

## Overview

This framework implements the theoretical insights from OpenAI's "Why Language Models Hallucinate" paper (2509.04664v1), introducing penalty-aware scoring and behavioral calibration to better evaluate RAG systems.

## Key Problem Addressed

The paper identified that current evaluation systems **penalize uncertainty and reward guessing**, leading to the "I Don't Know" tax where conservative systems are unfairly penalized. Our benchmark analysis confirmed this:

- **Conservative providers**: Higher abstention rates, lower volume scores, potentially higher quality scores
- **Aggressive providers**: Lower abstention rates, higher volume scores, potential quality penalties
- **Traditional scoring**: Only rewards correct answers, ignores the cost of wrong answers

## Confidence Threshold Framework

### Core Concept

The framework evaluates provider responses using **post-hoc confidence threshold analysis** rather than instructing providers how they will be judged. Providers give natural responses to questions, which are then evaluated against different confidence threshold criteria:

- **Providers**: Respond naturally to questions without threshold instructions
- **Evaluation**: Judge assesses responses against threshold-specific criteria post-hoc
- **Scoring**: Apply different penalty ratios to the same responses based on confidence thresholds

### Threshold Configurations

| Strategy | Threshold | Penalty Ratio | Description |
|----------|-----------|---------------|-------------|
| **Balanced** | 50% | 1.0 | Traditional binary scoring |
| **Conservative** | 75% | 3.0 | Higher confidence required |
| **Cautious** | 90% | 9.0 | Very high confidence required |

### Scoring Systems

1. **Volume Strategy** (Traditional)
   - Correct: +1
   - Wrong: 0
   - IDK: 0
   - *Rewards guessing*

2. **Quality Strategy** (Penalty-Aware)
   - Correct: +1
   - Wrong: -k (penalty ratio)
   - IDK: 0
   - *Rewards appropriate uncertainty*

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
# Custom configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py \
  --examples 50 \
  --max-workers 3 \
  --output-dir custom_results
```

## Output Metrics

### Core Metrics

- **Volume Score**: Traditional binary accuracy
- **Quality Score**: Penalty-aware score accounting for overconfidence
- **Attempted Rate**: Percentage of questions attempted (not abstained)
- **Accuracy Given Attempted**: Success rate on questions actually answered
- **Abstention Rate**: Percentage of questions where provider said "I don't know"
- **Overconfidence Penalty**: Count of wrong answers when should have abstained

### Behavioral Analysis

- **Conservative Strategy Assessment**: How appropriately providers handle uncertainty
- **Behavioral Calibration**: Optimal response selection across thresholds
- **Cross-Threshold Performance**: Identifying best strategies for each provider

## Implementation Architecture

### Components

1. **ConfidenceThresholdSimpleQAEval**: Enhanced evaluation class with post-hoc threshold analysis
2. **confidence_threshold_benchmark.py**: Main runner script for multi-provider evaluation
3. **Enhanced Grader Template**: Context-aware judging that assesses natural confidence signals
4. **Multi-Threshold Evaluator**: Evaluates each provider response against multiple thresholds

### Evaluation Flow

1. **Provider Response Collection**: Each provider called once per question with clean prompts
2. **Multi-Threshold Assessment**: Natural responses evaluated against each threshold criterion
3. **Confidence Signal Analysis**: Judge assesses appropriate certainty/uncertainty in responses
4. **Penalty-Aware Scoring**: Different penalty ratios applied to same responses

### Key Classes

```python
@dataclass
class ConfidenceThreshold:
    threshold: float      # 0.5, 0.75, 0.9
    penalty_ratio: float  # k in penalty formula
    name: str            # "Conservative", "Balanced", "Aggressive"
```

## Theoretical Foundation

### OpenAI Paper Insights

1. **Binary Grading Problem**: Current evaluations reward guessing over uncertainty
2. **Statistical Reduction**: Hallucinations arise from supervised learning challenges
3. **Socio-Technical Solution**: Modify scoring rather than just adding hallucination evals
4. **Explicit Confidence Targets**: Clear penalty ratios in instructions

### Our Implementation

- **Post-Hoc Evaluation**: Providers give natural responses, evaluated against thresholds afterward
- **Multi-Threshold Testing**: Each response assessed across 3 confidence levels
- **Penalty-Aware Scoring**: Wrong answers penalized proportionally to confidence threshold
- **Natural Confidence Signals**: Judge evaluates inherent certainty/uncertainty in responses
- **Behavioral Calibration**: Measuring appropriate uncertainty recognition without contamination
- **Cross-Provider Analysis**: Identifying volume vs quality strategy preferences

## Expected Insights

### Conservative vs Aggressive Strategies

- **Conservative Providers**: Higher abstention rates, lower volume scores, higher quality scores
- **Aggressive Providers**: Lower abstention rates, higher volume scores, potential quality penalties
- **Optimal Calibration**: Providers that adapt appropriately to confidence thresholds

### Use Case Guidance

- **High-Stakes Applications**: Favor quality strategy with conservative thresholds
- **Volume Applications**: Traditional volume strategy may be appropriate
- **Balanced Applications**: 75% threshold provides good compromise

## Integration with Existing Pipeline

The framework is designed to complement the existing benchmark:

- **Audit Logging**: Full compatibility with existing audit trail system
- **Leaderboard Generation**: Enhanced reports with confidence threshold analysis
- **JSON Output**: Structured results for further analysis
- **HTML Reports**: Comprehensive visual analysis

## Future Extensions

1. **Custom Threshold Configurations**: User-defined confidence levels
2. **Domain-Specific Penalties**: Different penalty ratios for different question types
3. **Real-World Harm Modeling**: Penalties based on actual application costs
4. **Provider-Specific Optimization**: Threshold tuning per provider type

## Files

- `confidence_threshold_simpleqa_eval.py`: Core evaluation framework
- `scripts/confidence_threshold_benchmark.py`: Main runner script
- `CONFIDENCE_THRESHOLD_FRAMEWORK.md`: This documentation
- Generated reports in `results/run_*/confidence_threshold_report_*.html`

## Research Impact

This implementation provides:

1. **Empirical Validation**: Testing OpenAI's theoretical framework on real RAG systems using proper post-hoc methodology
2. **Practical Guidelines**: Concrete recommendations for penalty-aware evaluation methodology
3. **Industry Standard**: Proposed enhancement to existing benchmarking practices that avoids provider contamination
4. **Academic Contribution**: Bridge between theory and practical implementation with methodological rigor
5. **Methodological Advancement**: Demonstrates how to evaluate confidence thresholds without contaminating provider responses