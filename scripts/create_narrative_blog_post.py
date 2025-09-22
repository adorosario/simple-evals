#!/usr/bin/env python3
"""
Create Compelling Narrative Blog Post using GPT-5
Focus on insights from the actual benchmark data with OpenAI-style narrative
"""

import json
from pathlib import Path
from openai import OpenAI

def create_narrative_blog_post(run_dir: str, output_path: str):
    """Create a compelling narrative blog post based on actual findings"""

    client = OpenAI()

    # Load benchmark data
    run_path = Path(run_dir)
    with open(run_path / "benchmark_results.json", 'r') as f:
        data = json.load(f)

    # Load some judge evaluations to understand the failure patterns
    judge_samples = []
    with open(run_path / "judge_evaluations.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # Just get first 5 examples
                judge_samples.append(json.loads(line))

    # Create comprehensive prompt for GPT-5
    prompt = f"""Write a compelling, narrative-driven blog post about a shocking discovery in RAG vs LLM performance.

NARRATIVE CONTEXT: We discovered something alarming - vanilla GPT-4.1 is failing catastrophically on simple factual questions, getting basic historical facts completely wrong. Meanwhile, RAG systems achieve perfect accuracy. This isn't just about performance - it's about reliability.

ACTUAL FINDINGS FROM BENCHMARK:
{json.dumps(data, indent=2)}

REAL EXAMPLES OF VANILLA LLM FAILURES:
{json.dumps(judge_samples, indent=2)}

WRITING STYLE: Follow OpenAI's blog post approach:
- Start with a compelling hook about the discovery
- Tell the story of what we found through data
- Use specific examples to illustrate points
- Balance technical depth with accessibility
- Focus on insights and implications
- Clear narrative arc with surprising revelations

KEY INSIGHTS TO EXPLORE:
1. The Hallucination Crisis: Vanilla GPT-4.1 confidently gave wrong answers to simple factual questions
   - Said Tanzan Ishibashi correctly but got Henri d'Angoulême's secretary wrong (said Jean de Nostredame instead of François de Malherbe)
   - Gave wrong birthplace for Muhammad Rafiq Tarar (Gujranwala vs Mandi Bahauddin)
   - Got Aniol Serrasolses' age wrong (28 vs 32)
   - Miscalculated vote difference by 82 votes (1,175 vs 1,093)

2. The RAG Revelation: Both RAG systems achieved perfect 100% accuracy on the same questions

3. The Speed vs Truth Dilemma: Vanilla is 7-14x faster but unreliable for facts

4. The Confidence Problem: The model answers confidently even when wrong

STRUCTURE:
- Compelling headline
- Opening hook about the discovery
- The experiment setup
- The shocking results (use specific examples)
- Why this matters for AI deployment
- The speed vs accuracy trade-off
- Implications for the future
- Interactive charts showing the data

TONE: Professional but engaging, like OpenAI's technical blog posts. Use data to tell a story about AI reliability.

Create a complete HTML blog post with embedded Chart.js visualizations that tells this compelling story."""

    try:
        print("📝 Creating compelling narrative blog post...")

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=8000
        )

        html_content = response.choices[0].message.content

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Narrative blog post created: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error creating narrative blog post: {e}")
        return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python create_narrative_blog_post.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_path = f"{run_dir}/narrative_blog_post.html"

    result = create_narrative_blog_post(run_dir, output_path)
    if result:
        print(f"🎉 Compelling narrative blog post ready!")
        print(f"📄 File: {result}")
        print(f"🌐 Open in browser to see the story unfold")