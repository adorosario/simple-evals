#!/usr/bin/env python3
"""
Critic Verification Script: Google Gemini RAG Cost Calculation

This script manually verifies that the cost calculation logic is correct
by comparing calculated costs against reported costs from the audit log.
"""

# Manual verification of Gemini cost calculations
# Pricing: Input=$2.00/M, Output=$12.00/M

queries = [
    {"qid": "simpleqa_0001", "prompt": 140, "comp": 25, "thoughts": 0, "total": 2279, "reported_cost": 0.004808},
    {"qid": "simpleqa_0004", "prompt": 139, "comp": 45, "thoughts": 0, "total": 2154, "reported_cost": 0.004758},
    {"qid": "simpleqa_0000", "prompt": 119, "comp": 43, "thoughts": 182, "total": 3594, "reported_cost": 0.009438},
    {"qid": "simpleqa_0003", "prompt": 147, "comp": 95, "thoughts": 124, "total": 3146, "reported_cost": 0.008482},
    {"qid": "simpleqa_0002", "prompt": 153, "comp": 56, "thoughts": 216, "total": 2623, "reported_cost": 0.007966},
]

INPUT_RATE = 2.00  # $/M tokens
OUTPUT_RATE = 12.00  # $/M tokens

print("=" * 80)
print("CRITIC VERIFICATION: Google Gemini RAG Cost Calculation")
print("=" * 80)
print()

all_pass = True
total_diff = 0

for q in queries:
    # Calculate RAG context tokens (hidden in total but not in prompt/comp/thoughts)
    rag_context = q["total"] - q["prompt"] - q["comp"] - q["thoughts"]

    # Input = user prompt + RAG context
    input_tokens = q["prompt"] + rag_context

    # Output = completion + thinking
    output_tokens = q["comp"] + q["thoughts"]

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * INPUT_RATE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
    calculated_cost = input_cost + output_cost

    # Compare
    diff = abs(calculated_cost - q["reported_cost"])
    match = "✅ PASS" if diff < 0.000001 else "❌ FAIL"
    if diff >= 0.000001:
        all_pass = False
    total_diff += diff

    print(f"Query: {q['qid']}")
    print(f"  Token Breakdown:")
    print(f"    User prompt:     {q['prompt']:,} tokens")
    print(f"    RAG context:     {rag_context:,} tokens (hidden)")
    print(f"    Completion:      {q['comp']:,} tokens")
    print(f"    Thinking:        {q['thoughts']:,} tokens")
    print(f"    Total (API):     {q['total']:,} tokens")
    print(f"  ")
    print(f"  Billing Calculation:")
    print(f"    Input billed:    {input_tokens:,} tokens @ ${INPUT_RATE}/M = ${input_cost:.6f}")
    print(f"    Output billed:   {output_tokens:,} tokens @ ${OUTPUT_RATE}/M = ${output_cost:.6f}")
    print(f"    Calculated:      ${calculated_cost:.6f}")
    print(f"    Reported:        ${q['reported_cost']:.6f}")
    print(f"    Difference:      ${diff:.9f}")
    print(f"    Status:          {match}")
    print()

print("=" * 80)
print(f"OVERALL VERDICT: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
print(f"Total cumulative difference: ${total_diff:.9f}")
print("=" * 80)

# Summary statistics
print()
print("SUMMARY STATISTICS:")
print("-" * 40)
avg_rag_context = sum(q["total"] - q["prompt"] - q["comp"] - q["thoughts"] for q in queries) / len(queries)
avg_thoughts = sum(q["thoughts"] for q in queries) / len(queries)
avg_cost = sum(q["reported_cost"] for q in queries) / len(queries)
print(f"  Average RAG context tokens: {avg_rag_context:,.0f}")
print(f"  Average thinking tokens: {avg_thoughts:,.0f}")
print(f"  Average cost per query: ${avg_cost:.6f}")
print(f"  Projected cost for 100 queries: ${avg_cost * 100:.2f}")
