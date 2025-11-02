
# Example: Integrating Brand Kit into confidence_threshold_benchmark.py

## At the top of the file, add imports:

```python
from scripts.report_generators import generate_quality_benchmark_report_v2
from scripts.generate_main_dashboard import generate_main_dashboard
```

## Replace the old quality benchmark report generation:

```python
# OLD (line ~243):
generate_quality_benchmark_report(
    results=results,
    output_dir=output_dir,
    run_metadata=run_metadata
)

# NEW:
generate_quality_benchmark_report_v2(
    results=results,
    output_dir=output_dir,
    run_metadata=run_metadata
)
```

## At the END of the run (after all reports are generated):

```python
# Generate main dashboard hub
print("\n📊 Generating main dashboard hub...")
generate_main_dashboard(
    results_dir=output_dir,
    run_metadata={
        "run_id": run_metadata.get("run_id", ""),
        "timestamp": datetime.now(),
        "providers": providers,
        "total_questions": run_metadata.get("samples_per_provider", 200),
        "confidence_threshold": confidence_threshold
    }
)
print(f"✅ Main dashboard generated: {output_dir}/index.html")
```

## Complete example for main() function in confidence_threshold_benchmark.py:

```python
def main():
    args = parse_args()

    # ... existing setup code ...

    # Run evaluations
    results = run_benchmark(...)

    # Generate quality benchmark report (NEW v2)
    print("\n📊 Generating quality benchmark report...")
    generate_quality_benchmark_report_v2(
        results=results,
        output_dir=output_dir,
        run_metadata=run_metadata
    )

    # Generate statistical analysis (if applicable)
    # ... existing code ...

    # Generate forensic reports (if applicable)
    # ... existing code ...

    # Generate main dashboard hub (NEW - do this LAST)
    print("\n📊 Generating main dashboard hub...")
    generate_main_dashboard(
        results_dir=output_dir,
        run_metadata={
            "run_id": run_metadata.get("run_id", ""),
            "timestamp": datetime.now(),
            "providers": [r["sampler_name"] for r in results if r["success"]],
            "total_questions": run_metadata.get("samples_per_provider", 200),
            "confidence_threshold": 0.8
        }
    )

    print(f"\n✅ All reports generated successfully!")
    print(f"📁 Results directory: {output_dir}")
    print(f"🏠 Main dashboard: {output_dir}/index.html")
    print(f"   Open this file in your browser to navigate all reports")
```
