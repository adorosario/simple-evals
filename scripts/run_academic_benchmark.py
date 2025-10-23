#!/usr/bin/env python3
"""
Academic Benchmark Orchestration Script
Single command to run complete academic-grade benchmark with all analysis.

Automatically runs:
1. Main confidence threshold benchmark
2. Universal penalty analysis (all providers)
3. Universal forensic generation (all providers)
4. Statistical analysis with confidence intervals
5. Reproducibility documentation

Output: Complete academic package ready for peer review.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class AcademicBenchmarkRunner:
    """Orchestrates complete academic benchmark pipeline"""

    def __init__(self, examples: int, output_dir: str = None, validate_rigor: bool = True):
        self.examples = examples
        self.output_dir = output_dir
        self.validate_rigor = validate_rigor
        self.run_dir = None
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "WARNING": "⚠",
            "ERROR": "✗",
            "STEP": "►"
        }.get(level, "•")

        print(f"[{timestamp}] {prefix} {message}")

    def run_command(self, description: str, command: list, critical: bool = True) -> bool:
        """
        Run a command and handle errors

        Args:
            description: What this command does
            command: Command list for subprocess
            critical: If True, stop pipeline on failure

        Returns:
            True if successful, False otherwise
        """
        self.log(f"{description}...", "STEP")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            self.log(f"{description} - Complete", "SUCCESS")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = f"{description} failed: {e.stderr[:200]}"
            self.log(error_msg, "ERROR")
            self.errors.append(error_msg)

            if critical:
                self.log("Critical step failed. Stopping pipeline.", "ERROR")
                sys.exit(1)

            return False

    def find_latest_run_dir(self) -> Path:
        """Find the most recently created run directory"""
        results_dir = Path("results")
        if not results_dir.exists():
            raise FileNotFoundError("Results directory not found")

        run_dirs = sorted(results_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not run_dirs:
            raise FileNotFoundError("No run directories found in results/")

        return run_dirs[0]

    def run_main_benchmark(self):
        """Step 1: Run main confidence threshold benchmark"""
        self.log("PHASE 1: Running Main Benchmark", "STEP")
        self.log(f"Examples: {self.examples}", "INFO")

        command = [
            "python", "scripts/confidence_threshold_benchmark.py",
            "--examples", str(self.examples)
        ]

        if self.output_dir:
            command.extend(["--output-dir", self.output_dir])

        success = self.run_command(
            "Main benchmark execution",
            command,
            critical=True
        )

        # Find the run directory that was just created
        self.run_dir = self.find_latest_run_dir()
        self.log(f"Run directory: {self.run_dir}", "INFO")

    def run_penalty_analysis(self, provider: str):
        """Run penalty analysis for a specific provider"""
        command = [
            "python", "scripts/universal_penalty_analyzer.py",
            "--run-dir", str(self.run_dir),
            "--provider", provider
        ]

        return self.run_command(
            f"Penalty analysis for {provider}",
            command,
            critical=False  # Not critical - provider might have no failures
        )

    def run_forensic_generation(self, provider: str):
        """Generate forensics for a specific provider"""
        command = [
            "python", "scripts/generate_universal_forensics.py",
            "--run-dir", str(self.run_dir),
            "--provider", provider
        ]

        return self.run_command(
            f"Forensic report generation for {provider}",
            command,
            critical=False  # Not critical - provider might have no failures
        )

    def run_all_provider_analysis(self):
        """Step 2 & 3: Run penalty analysis and forensics for all providers"""
        self.log("PHASE 2 & 3: Provider Analysis and Forensics", "STEP")

        providers = ['customgpt', 'openai_rag', 'openai_vanilla']

        for provider in providers:
            self.log(f"\nAnalyzing {provider.upper()}", "STEP")

            # Penalty analysis
            penalty_success = self.run_penalty_analysis(provider)

            # Forensics (only if penalty analysis found cases)
            if penalty_success:
                self.run_forensic_generation(provider)
            else:
                self.log(f"No penalty cases for {provider} - skipping forensics", "INFO")

    def run_statistical_analysis(self):
        """Step 4: Run statistical analysis"""
        self.log("PHASE 4: Statistical Analysis", "STEP")

        command = [
            "python", "scripts/academic_statistical_analysis.py",
            "--run-dir", str(self.run_dir)
        ]

        self.run_command(
            "Statistical analysis with confidence intervals",
            command,
            critical=False  # Warn but don't stop if this fails
        )

    def run_reproducibility_docs(self):
        """Step 5: Generate reproducibility documentation"""
        self.log("PHASE 5: Reproducibility Documentation", "STEP")

        command = [
            "python", "scripts/generate_reproducibility_docs.py",
            "--run-dir", str(self.run_dir)
        ]

        self.run_command(
            "Reproducibility manifest generation",
            command,
            critical=False
        )

    def validate_academic_completeness(self):
        """Validate that all required outputs exist"""
        if not self.validate_rigor:
            return

        self.log("PHASE 6: Academic Completeness Validation", "STEP")

        required_files = [
            # Core benchmark outputs
            ("run_metadata.json", "Run metadata"),
            ("quality_benchmark_results.json", "Quality benchmark results"),
            ("provider_requests.jsonl", "Provider audit log"),
            ("judge_evaluations.jsonl", "Judge evaluations"),
        ]

        optional_files = [
            # Statistical analysis
            (f"statistical_analysis_{self.run_dir.name}.md", "Statistical analysis report"),
            # Reproducibility
            (f"REPRODUCIBILITY_{self.run_dir.name}.md", "Reproducibility documentation"),
        ]

        # Check required files
        missing_required = []
        for filename, description in required_files:
            file_path = self.run_dir / filename
            if file_path.exists():
                self.log(f"✓ {description}", "SUCCESS")
            else:
                self.log(f"✗ {description} MISSING", "ERROR")
                missing_required.append(description)

        # Check optional files (warnings only)
        missing_optional = []
        for filename, description in optional_files:
            file_path = self.run_dir / filename
            if file_path.exists():
                self.log(f"✓ {description}", "SUCCESS")
            else:
                self.log(f"⚠ {description} missing", "WARNING")
                missing_optional.append(description)

        # Check provider-specific outputs
        providers = ['customgpt', 'openai_rag', 'openai_vanilla']
        for provider in providers:
            # Check if penalty analysis exists
            penalty_dir = self.run_dir / f"{provider}_penalty_analysis"
            if penalty_dir.exists():
                self.log(f"✓ {provider} penalty analysis", "SUCCESS")

                # Check if forensics exist
                forensics_dir = self.run_dir / f"{provider}_forensics"
                if forensics_dir.exists():
                    self.log(f"✓ {provider} forensics", "SUCCESS")
                else:
                    self.log(f"⚠ {provider} forensics missing", "WARNING")
            else:
                self.log(f"ℹ {provider} had no penalty cases", "INFO")

        # Summary
        if missing_required:
            self.log(f"\n✗ VALIDATION FAILED: {len(missing_required)} required files missing", "ERROR")
            sys.exit(1)
        elif missing_optional:
            self.log(f"\n⚠ VALIDATION WARNING: {len(missing_optional)} optional files missing", "WARNING")
            self.log("Benchmark complete but some analysis steps failed", "WARNING")
        else:
            self.log("\n✓ VALIDATION PASSED: All required outputs present", "SUCCESS")

    def generate_summary(self):
        """Generate final summary report"""
        self.log("\n" + "="*60, "INFO")
        self.log("ACADEMIC BENCHMARK COMPLETE", "SUCCESS")
        self.log("="*60, "INFO")
        self.log(f"Run Directory: {self.run_dir}", "INFO")
        self.log(f"Questions: {self.examples}", "INFO")

        if self.errors:
            self.log(f"\n⚠ Encountered {len(self.errors)} non-critical errors:", "WARNING")
            for error in self.errors:
                self.log(f"  - {error}", "WARNING")

        self.log("\nGenerated Outputs:", "INFO")
        self.log("  - Core benchmark results", "INFO")
        self.log("  - Provider penalty analysis (all providers)", "INFO")
        self.log("  - Forensic reports (all providers with failures)", "INFO")
        self.log("  - Statistical analysis with 95% CIs", "INFO")
        self.log("  - Reproducibility documentation", "INFO")

        self.log(f"\n► View results: {self.run_dir}/", "STEP")
        self.log(f"► Quality report: {self.run_dir}/quality_benchmark_report_*.html", "STEP")
        self.log(f"► Statistical analysis: {self.run_dir}/statistical_analysis_*.html", "STEP")
        self.log("="*60 + "\n", "INFO")

    def run(self):
        """Execute complete academic benchmark pipeline"""
        start_time = datetime.now()

        self.log("="*60, "INFO")
        self.log("ACADEMIC BENCHMARK PIPELINE", "INFO")
        self.log("="*60, "INFO")
        self.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        self.log(f"Questions: {self.examples}", "INFO")
        self.log(f"Validation: {'Enabled' if self.validate_rigor else 'Disabled'}", "INFO")
        self.log("="*60 + "\n", "INFO")

        try:
            # Phase 1: Main benchmark
            self.run_main_benchmark()

            # Phase 2 & 3: Provider analysis and forensics
            self.run_all_provider_analysis()

            # Phase 4: Statistical analysis
            self.run_statistical_analysis()

            # Phase 5: Reproducibility docs
            self.run_reproducibility_docs()

            # Phase 6: Validation
            if self.validate_rigor:
                self.validate_academic_completeness()

            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            self.log(f"\nTotal Duration: {duration}", "INFO")

            self.generate_summary()

        except KeyboardInterrupt:
            self.log("\nPipeline interrupted by user", "WARNING")
            sys.exit(1)

        except Exception as e:
            self.log(f"\nUnexpected error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete academic-grade benchmark with all analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 200-question pilot with full validation
  python run_academic_benchmark.py --examples 200

  # Run full benchmark (4300 questions)
  python run_academic_benchmark.py --examples 4300

  # Run without validation (faster, for testing)
  python run_academic_benchmark.py --examples 10 --no-validate

  # Custom output directory
  python run_academic_benchmark.py --examples 200 --output-dir custom_results/

Phases:
  1. Main benchmark (confidence_threshold_benchmark.py)
  2. Penalty analysis (universal_penalty_analyzer.py) for each provider
  3. Forensic generation (generate_universal_forensics.py) for each provider
  4. Statistical analysis with 95% confidence intervals
  5. Reproducibility documentation
  6. Academic completeness validation

Output:
  Complete academic package in results/run_XXXXXXX/ directory
        """
    )

    parser.add_argument(
        '--examples',
        type=int,
        required=True,
        help='Number of questions to evaluate'
    )

    parser.add_argument(
        '--output-dir',
        help='Custom output directory (default: auto-generated in results/)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip academic completeness validation'
    )

    args = parser.parse_args()

    # Create and run pipeline
    runner = AcademicBenchmarkRunner(
        examples=args.examples,
        output_dir=args.output_dir,
        validate_rigor=not args.no_validate
    )

    runner.run()


if __name__ == "__main__":
    main()
