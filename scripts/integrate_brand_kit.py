"""
Brand Kit Integration Script
============================

Automatically integrates the new brand kit into existing report generation workflow.

Usage:
    python scripts/integrate_brand_kit.py --dry-run     # Preview changes
    python scripts/integrate_brand_kit.py               # Apply changes
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def update_confidence_threshold_benchmark():
    """Update confidence_threshold_benchmark.py to use new report generators"""

    file_path = Path("scripts/confidence_threshold_benchmark.py")
    content = file_path.read_text()

    # Check if already updated
    if "from scripts.report_generators import" in content:
        print("✅ confidence_threshold_benchmark.py already updated")
        return

    # Add imports at top of file
    import_section = """
# Brand kit report generators
from scripts.report_generators import generate_quality_benchmark_report_v2
from scripts.generate_main_dashboard import generate_main_dashboard
"""

    # Find where to insert imports (after existing imports)
    lines = content.split('\n')
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end_idx = i + 1

    lines.insert(import_end_idx, import_section)

    # Update the function call
    # Replace generate_quality_benchmark_report with generate_quality_benchmark_report_v2
    updated_content = '\n'.join(lines)
    updated_content = updated_content.replace(
        'generate_quality_benchmark_report(',
        'generate_quality_benchmark_report_v2('
    )

    # Add main dashboard generation at end of run
    # Find the location where we should add the dashboard generation
    # (typically at the end of the main function)

    print("✅ Updated confidence_threshold_benchmark.py")
    print("   - Added brand kit imports")
    print("   - Replaced quality benchmark report function")
    print("   ⚠️  MANUAL STEP NEEDED: Add generate_main_dashboard() call at end of run")

    return updated_content


def create_integration_example():
    """Create an example showing how to integrate the brand kit"""

    example = """
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
print("\\n📊 Generating main dashboard hub...")
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
    print("\\n📊 Generating quality benchmark report...")
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
    print("\\n📊 Generating main dashboard hub...")
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

    print(f"\\n✅ All reports generated successfully!")
    print(f"📁 Results directory: {output_dir}")
    print(f"🏠 Main dashboard: {output_dir}/index.html")
    print(f"   Open this file in your browser to navigate all reports")
```
"""

    output_path = Path("docs/integration_guide.md")
    output_path.write_text(example)
    print(f"✅ Created integration guide: {output_path}")


def main():
    """Main integration script"""

    parser = argparse.ArgumentParser(description="Integrate brand kit into report generators")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    print("=" * 60)
    print("Brand Kit Integration Script")
    print("=" * 60)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")

    # Create integration guide
    create_integration_example()
    print()

    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print()
    print("1. Review docs/integration_guide.md for detailed integration steps")
    print("2. Update scripts/confidence_threshold_benchmark.py:")
    print("   - Add imports for report_generators and generate_main_dashboard")
    print("   - Replace generate_quality_benchmark_report with _v2 version")
    print("   - Add generate_main_dashboard() call at end of run")
    print()
    print("3. Test with a debug run:")
    print("   docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug")
    print()
    print("4. Verify all HTML files:")
    print("   - index.html (main dashboard)")
    print("   - quality_benchmark_report_*.html")
    print("   - statistical_analysis_*.html")
    print("   - */forensic_dashboard.html")
    print()
    print("5. Check that navigation works between all pages")
    print()


if __name__ == "__main__":
    main()
