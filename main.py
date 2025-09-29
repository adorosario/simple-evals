#!/usr/bin/env python3
"""
Main RAG Benchmark Runner with Automatic Forensic Analysis

This script runs the complete RAG evaluation pipeline:
1. Confidence threshold benchmark evaluation
2. Penalty analysis for failed cases
3. GPT-5 deep forensic analysis
4. Engineering post-mortem report generation
5. Beautiful HTML forensic dashboard creation

Usage:
    python main.py [--examples N] [--debug] [--max-workers N] [--output-dir DIR]

Examples:
    python main.py --debug                    # Quick 10-question debug run
    python main.py --examples 200             # Full 200-question evaluation
    python main.py --examples 50 --debug      # 50 questions with verbose output
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description="", cwd=None):
    """Run a command and handle errors gracefully"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )

        duration = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {duration:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed after {duration:.1f}s")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 {description} crashed after {duration:.1f}s: {e}")
        return False


def find_latest_run_directory(output_dir: str = "results") -> Path:
    """Find the most recently created run directory"""
    results_path = Path(output_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {output_dir}")

    run_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {output_dir}")

    # Sort by creation time (most recent first)
    latest_run = max(run_dirs, key=lambda d: d.stat().st_ctime)
    return latest_run


def generate_forensic_pipeline(run_dir: Path, provider: str = "customgpt") -> bool:
    """Run the complete forensic analysis pipeline"""
    print(f"\n🔬 STARTING FORENSIC ANALYSIS PIPELINE")
    print(f"Run directory: {run_dir}")
    print(f"Provider: {provider}")
    print("=" * 80)

    # Step 1: Penalty deep dive analysis
    step1_success = run_command([
        "python", "scripts/customgpt_penalty_deep_dive.py",
        "--run-dir", str(run_dir)
    ], f"Step 1: {provider.upper()} penalty deep dive analysis")

    if not step1_success:
        print(f"⚠️  Step 1 failed, but continuing with forensic pipeline...")

    # Step 2: GPT-5 failure analysis
    step2_success = run_command([
        "python", "scripts/gpt5_failure_analysis.py",
        "--run-dir", str(run_dir)
    ], "Step 2: GPT-5 deep failure analysis")

    if not step2_success:
        print(f"⚠️  Step 2 failed, but continuing with forensic pipeline...")

    # Step 3: Engineering report generation
    step3_success = run_command([
        "python", "scripts/comprehensive_engineering_report.py",
        "--run-dir", str(run_dir)
    ], "Step 3: Engineering post-mortem report generation")

    if not step3_success:
        print(f"⚠️  Step 3 failed, but continuing with forensic pipeline...")

    # Step 4: HTML forensic dashboard generation
    step4_success = run_command([
        "python", "scripts/generate_forensic_reports.py",
        "--run-dir", str(run_dir),
        "--provider", provider
    ], "Step 4: HTML forensic dashboard generation")

    # Report results
    successful_steps = sum([step1_success, step2_success, step3_success, step4_success])

    print(f"\n📊 FORENSIC PIPELINE SUMMARY:")
    print(f"   ✅ Successful steps: {successful_steps}/4")
    print(f"   📁 Run directory: {run_dir}")

    if step4_success:
        dashboard_file = run_dir / "forensic_dashboard.html"
        print(f"   🌐 Forensic dashboard: {dashboard_file}")
        print(f"   🚀 To view: python -m http.server 8000 (navigate to {dashboard_file.relative_to(Path.cwd())})")

    return successful_steps >= 2  # At least half the steps should succeed


def main():
    """Main benchmark runner with automatic forensic analysis"""
    parser = argparse.ArgumentParser(
        description="Complete RAG Benchmark with Automatic Forensic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --debug                     # Quick 10-question debug run
  python main.py --examples 200              # Full 200-question evaluation
  python main.py --examples 50 --max-workers 4  # Custom configuration
        """
    )

    # Benchmark arguments
    parser.add_argument("--examples", type=int, help="Number of examples per provider (default: debug=10, normal=200)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--flex-tier", action="store_true", help="Use GPT-5 Flex tier (slower but cheaper)")

    # Pipeline control
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark, only run forensics on latest run")
    parser.add_argument("--skip-forensics", action="store_true", help="Skip forensic analysis, only run benchmark")
    parser.add_argument("--provider", default="customgpt", help="Provider to analyze for forensics")

    args = parser.parse_args()

    print("🏆 RAG BENCHMARK WITH AUTOMATIC FORENSIC ANALYSIS")
    print("=" * 70)
    print("Complete pipeline: Benchmark → Analysis → Forensics → HTML Reports")
    print("=" * 70)

    # Determine number of examples
    if args.examples is None:
        examples = 10 if args.debug else 200
    else:
        examples = args.examples

    print(f"📊 Configuration:")
    print(f"   Examples per provider: {examples}")
    print(f"   Debug mode: {args.debug}")
    print(f"   Max workers: {args.max_workers}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Flex tier: {args.flex_tier}")
    print(f"   Skip benchmark: {args.skip_benchmark}")
    print(f"   Skip forensics: {args.skip_forensics}")

    overall_start = time.time()

    # Phase 1: Run Benchmark (unless skipped)
    if not args.skip_benchmark:
        print(f"\n🎯 PHASE 1: RUNNING CONFIDENCE THRESHOLD BENCHMARK")
        print("=" * 70)

        benchmark_cmd = [
            "python", "scripts/confidence_threshold_benchmark.py",
            "--examples", str(examples),
            "--max-workers", str(args.max_workers),
            "--output-dir", args.output_dir
        ]

        if args.debug:
            benchmark_cmd.append("--debug")

        if args.flex_tier:
            benchmark_cmd.append("--flex-tier")

        benchmark_success = run_command(
            benchmark_cmd,
            f"Running RAG benchmark with {examples} examples per provider"
        )

        if not benchmark_success:
            print("❌ Benchmark failed! Cannot proceed with forensic analysis.")
            return 1
    else:
        print(f"\n⏭️  PHASE 1: SKIPPED (benchmark disabled)")
        benchmark_success = True

    # Find the run directory
    if not args.skip_forensics:
        print(f"\n🔍 FINDING LATEST RUN DIRECTORY")
        print("-" * 40)

        try:
            run_dir = find_latest_run_directory(args.output_dir)
            print(f"   📁 Latest run: {run_dir}")
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            return 1

        # Phase 2: Run Forensic Analysis Pipeline
        print(f"\n🔬 PHASE 2: FORENSIC ANALYSIS PIPELINE")
        print("=" * 70)

        forensics_success = generate_forensic_pipeline(run_dir, args.provider)

        if not forensics_success:
            print("⚠️  Forensic analysis had issues, but basic results should be available.")
    else:
        print(f"\n⏭️  PHASE 2: SKIPPED (forensics disabled)")
        forensics_success = True
        run_dir = find_latest_run_directory(args.output_dir) if not args.skip_benchmark else None

    # Final Summary
    overall_duration = time.time() - overall_start

    print(f"\n🎉 COMPLETE PIPELINE FINISHED")
    print("=" * 70)
    print(f"   ⏱️  Total duration: {overall_duration/60:.1f} minutes")
    print(f"   📊 Benchmark: {'✅' if benchmark_success else '❌' if not args.skip_benchmark else '⏭️'}")
    print(f"   🔬 Forensics: {'✅' if forensics_success else '❌' if not args.skip_forensics else '⏭️'}")

    if run_dir and not args.skip_forensics:
        print(f"\n📁 RESULTS LOCATION:")
        print(f"   Directory: {run_dir}")

        # Check what files were generated
        quality_report = list(run_dir.glob("quality_benchmark_report_*.html"))
        forensic_dashboard = run_dir / "forensic_dashboard.html"
        engineering_report = run_dir / f"{args.provider}_engineering_report.html"

        print(f"\n📋 GENERATED REPORTS:")
        if quality_report:
            print(f"   📊 Quality benchmark: {quality_report[0]}")
        if forensic_dashboard.exists():
            print(f"   🔍 Forensic dashboard: {forensic_dashboard}")
        if engineering_report.exists():
            print(f"   📝 Engineering report: {engineering_report}")

        print(f"\n🌐 TO VIEW RESULTS:")
        print(f"   1. cd {run_dir.parent}")
        print(f"   2. python -m http.server 8000")
        print(f"   3. Open http://localhost:8000/{run_dir.name}/")

        if forensic_dashboard.exists():
            print(f"   4. Navigate to forensic_dashboard.html for complete analysis")

    return 0 if (benchmark_success or args.skip_benchmark) and (forensics_success or args.skip_forensics) else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Pipeline crashed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)