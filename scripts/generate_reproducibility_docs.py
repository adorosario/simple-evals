#!/usr/bin/env python3
"""
Reproducibility Documentation Generator
Auto-generates complete reproducibility manifest for academic benchmark runs.

Captures:
- Complete environment (Python, packages, Docker)
- Data versions and hashes
- Configuration details
- Exact commands to reproduce
- Timestamps and metadata

Suitable for academic publication and peer review.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse
import hashlib


def get_python_version() -> Dict[str, str]:
    """Get Python version information"""
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro
        },
        "executable": sys.executable
    }


def get_package_versions() -> Dict[str, str]:
    """Get installed package versions"""
    try:
        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        packages = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==', 1)
                packages[name] = version
        return packages
    except Exception as e:
        return {"error": str(e)}


def get_docker_info() -> Dict[str, Any]:
    """Get Docker environment information"""
    docker_info = {}

    # Check if running in Docker
    if os.path.exists('/.dockerenv'):
        docker_info['running_in_docker'] = True

        # Try to get image info
        try:
            # Get Docker image ID
            result = subprocess.run(
                ['cat', '/proc/self/cgroup'],
                capture_output=True,
                text=True
            )
            docker_info['cgroup'] = result.stdout[:200]  # First 200 chars
        except Exception as e:
            docker_info['cgroup_error'] = str(e)
    else:
        docker_info['running_in_docker'] = False

    # Try to get docker-compose version
    try:
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True
        )
        docker_info['docker_compose_version'] = result.stdout.strip()
    except Exception:
        docker_info['docker_compose_version'] = 'not available'

    return docker_info


def get_git_info(repo_path: Path) -> Dict[str, str]:
    """Get Git repository information"""
    git_info = {}

    try:
        # Get current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True
        )
        git_info['commit_hash'] = result.stdout.strip()

        # Get commit date
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci'],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True
        )
        git_info['commit_date'] = result.stdout.strip()

        # Get branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True
        )
        git_info['branch'] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True
        )
        git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
        if git_info['has_uncommitted_changes']:
            git_info['warning'] = "Repository has uncommitted changes - results may not be exactly reproducible"

    except Exception as e:
        git_info['error'] = str(e)

    return git_info


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    if not file_path.exists():
        return "file_not_found"

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_directory_hash(dir_path: Path, pattern: str = "*") -> Dict[str, str]:
    """Calculate hashes of files in a directory"""
    hashes = {}
    if not dir_path.exists():
        return {"error": "directory_not_found"}

    for file_path in sorted(dir_path.glob(pattern)):
        if file_path.is_file():
            hashes[file_path.name] = get_file_hash(file_path)

    return hashes


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load run metadata from the benchmark run"""
    metadata_file = run_dir / "run_metadata.json"

    if not metadata_file.exists():
        return {"error": "run_metadata.json not found"}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def generate_reproducibility_manifest(
    run_dir: Path,
    repo_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate complete reproducibility manifest

    Args:
        run_dir: Path to evaluation run directory
        repo_root: Path to repository root (default: parent of run_dir)

    Returns:
        Complete manifest dict
    """
    if repo_root is None:
        # Assume run_dir is results/run_XXX, so repo_root is two levels up
        repo_root = run_dir.parent.parent

    manifest = {
        "reproducibility_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "run_id": run_dir.name,
    }

    # Environment
    print("Capturing environment information...")
    manifest["environment"] = {
        "python": get_python_version(),
        "packages": get_package_versions(),
        "docker": get_docker_info(),
        "platform": {
            "system": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine
        }
    }

    # Code version
    print("Capturing code version...")
    manifest["code"] = {
        "repository": get_git_info(repo_root),
        "benchmark_script": "scripts/confidence_threshold_benchmark.py",
        "script_hash": get_file_hash(repo_root / "scripts" / "confidence_threshold_benchmark.py")
    }

    # Data
    print("Capturing data information...")
    kb_hashes = get_directory_hash(repo_root / "knowledge_base_full", "*.txt")
    # Sample first 5 files
    kb_hashes_sample = dict(list(kb_hashes.items())[:5]) if isinstance(kb_hashes, dict) else kb_hashes
    manifest["data"] = {
        "dataset": "SimpleQA",
        "knowledge_base_directory": "knowledge_base_full",
        "knowledge_base_hashes": kb_hashes_sample
    }

    # Configuration
    print("Capturing configuration...")
    run_metadata = load_run_metadata(run_dir)
    manifest["configuration"] = {
        "run_metadata": run_metadata,
        "environment_variables": {
            "OPENAI_VECTOR_STORE_ID": os.getenv("OPENAI_VECTOR_STORE_ID", "not_set"),
            "CUSTOMGPT_PROJECT": os.getenv("CUSTOMGPT_PROJECT", "not_set")
        }
    }

    # Execution
    print("Capturing execution details...")
    manifest["execution"] = {
        "command": "docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples N",
        "working_directory": str(repo_root),
        "run_directory": str(run_dir)
    }

    return manifest


def generate_markdown_documentation(
    manifest: Dict[str, Any],
    output_file: Path
):
    """
    Generate human-readable markdown documentation

    Args:
        manifest: Reproducibility manifest dict
        output_file: Where to save the markdown file
    """
    doc = []

    doc.append("# Reproducibility Documentation")
    doc.append(f"**Run ID:** `{manifest['run_id']}`")
    doc.append(f"**Generated:** {manifest['generated_at']}")
    doc.append("")
    doc.append("---")
    doc.append("")

    doc.append("## Executive Summary")
    doc.append("")
    doc.append("This document contains complete information required to reproduce this benchmark run.")
    doc.append("All environment details, data versions, and configuration parameters are documented below.")
    doc.append("")

    # Environment
    doc.append("## Environment")
    doc.append("")
    doc.append("### Python")
    doc.append("")
    python_info = manifest["environment"]["python"]
    doc.append(f"- **Version:** {python_info['version_info']['major']}.{python_info['version_info']['minor']}.{python_info['version_info']['micro']}")
    doc.append(f"- **Executable:** `{python_info['executable']}`")
    doc.append("")

    doc.append("### Docker")
    doc.append("")
    docker_info = manifest["environment"]["docker"]
    doc.append(f"- **Running in Docker:** {docker_info['running_in_docker']}")
    doc.append(f"- **Docker Compose Version:** {docker_info.get('docker_compose_version', 'N/A')}")
    doc.append("")

    doc.append("### Platform")
    doc.append("")
    platform = manifest["environment"]["platform"]
    doc.append(f"- **System:** {platform['system']}")
    doc.append(f"- **Release:** {platform['release']}")
    doc.append(f"- **Machine:** {platform['machine']}")
    doc.append("")

    doc.append("### Key Package Versions")
    doc.append("")
    packages = manifest["environment"]["packages"]
    key_packages = ['openai', 'anthropic', 'requests', 'numpy', 'pandas']
    for pkg in key_packages:
        if pkg in packages:
            doc.append(f"- **{pkg}:** {packages[pkg]}")
    doc.append("")
    doc.append(f"*(Complete package list: {len(packages)} packages installed)*")
    doc.append("")

    # Code
    doc.append("## Code Version")
    doc.append("")
    repo_info = manifest["code"]["repository"]
    if "commit_hash" in repo_info:
        doc.append(f"- **Commit Hash:** `{repo_info['commit_hash']}`")
        doc.append(f"- **Commit Date:** {repo_info['commit_date']}")
        doc.append(f"- **Branch:** {repo_info['branch']}")

        if repo_info.get('has_uncommitted_changes'):
            doc.append("")
            doc.append("⚠️  **Warning:** Repository had uncommitted changes at runtime")
            doc.append("")
    else:
        doc.append("*Git information not available*")
    doc.append("")

    # Data
    doc.append("## Data")
    doc.append("")
    doc.append(f"- **Dataset:** {manifest['data']['dataset']}")
    doc.append(f"- **Knowledge Base:** `{manifest['data']['knowledge_base_directory']}`")
    doc.append("")
    kb_hashes = manifest['data'].get('knowledge_base_hashes', {})
    if kb_hashes and 'error' not in kb_hashes:
        doc.append("### Knowledge Base File Hashes (Sample)")
        doc.append("")
        for filename, file_hash in list(kb_hashes.items())[:5]:
            doc.append(f"- `{filename}`: `{file_hash[:16]}...`")
        doc.append("")

    # Configuration
    doc.append("## Configuration")
    doc.append("")
    run_config = manifest['configuration']['run_metadata']
    if run_config and 'error' not in run_config:
        doc.append(f"- **Total Questions:** {run_config.get('total_questions', 'N/A')}")
        doc.append(f"- **Start Time:** {run_config.get('start_time', 'N/A')}")
        doc.append(f"- **End Time:** {run_config.get('end_time', 'N/A')}")
        doc.append("")

        providers = run_config.get('providers', [])
        if providers:
            doc.append("### Providers Tested")
            doc.append("")
            if isinstance(providers, list):
                for provider in providers:
                    if isinstance(provider, dict):
                        # Extract provider name and config
                        provider_name = provider.get('name', 'Unknown')
                        provider_config = provider.get('config', {})

                        doc.append(f"#### {provider_name}")
                        doc.append("")

                        # Key configuration details
                        if 'model' in provider_config:
                            doc.append(f"- **Model:** `{provider_config['model']}`")
                        if 'temperature' in provider_config:
                            doc.append(f"- **Temperature:** {provider_config['temperature']}")
                        if 'max_tokens' in provider_config:
                            doc.append(f"- **Max Tokens:** {provider_config['max_tokens']}")

                        # Provider-specific details
                        if 'vector_store_id' in provider_config:
                            doc.append(f"- **Vector Store:** `{provider_config['vector_store_id']}`")

                        additional = provider_config.get('additional_params', {})
                        if 'project_id' in additional:
                            doc.append(f"- **CustomGPT Project:** `{additional['project_id']}`")

                        doc.append("")
                    else:
                        # Fallback for string providers
                        doc.append(f"- {provider}")
                        doc.append("")
            elif isinstance(providers, dict):
                for provider_name, provider_config in providers.items():
                    doc.append(f"- **{provider_name}**")
                    if isinstance(provider_config, dict):
                        if 'model' in provider_config:
                            doc.append(f"  - Model: `{provider_config['model']}`")
                        if 'temperature' in provider_config:
                            doc.append(f"  - Temperature: {provider_config['temperature']}")
            doc.append("")

    doc.append("### Environment Variables")
    doc.append("")
    env_vars = manifest['configuration']['environment_variables']
    for var, value in env_vars.items():
        # Mask sensitive values
        if value and value != 'not_set':
            masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
        else:
            masked_value = value
        doc.append(f"- `{var}`: `{masked_value}`")
    doc.append("")

    # Reproduction Instructions
    doc.append("## Reproduction Instructions")
    doc.append("")
    doc.append("To reproduce this benchmark run exactly:")
    doc.append("")
    doc.append("### 1. Environment Setup")
    doc.append("")
    doc.append("```bash")
    doc.append("# Clone repository")
    doc.append("git clone <repository_url>")
    doc.append("cd simple-evals")
    doc.append("")
    if "commit_hash" in manifest["code"]["repository"]:
        doc.append(f"# Checkout exact commit")
        doc.append(f"git checkout {manifest['code']['repository']['commit_hash']}")
        doc.append("")
    doc.append("# Build Docker environment")
    doc.append("docker compose build")
    doc.append("```")
    doc.append("")

    doc.append("### 2. Data Preparation")
    doc.append("")
    doc.append("```bash")
    doc.append("# Download knowledge base")
    doc.append("docker compose run --rm simple-evals python scripts/download_and_extract_kb.py")
    doc.append("```")
    doc.append("")

    doc.append("### 3. Set Environment Variables")
    doc.append("")
    doc.append("Create `.env` file with:")
    doc.append("```")
    doc.append("OPENAI_API_KEY=<your_key>")
    doc.append("CUSTOMGPT_API_KEY=<your_key>")
    doc.append(f"OPENAI_VECTOR_STORE_ID={env_vars.get('OPENAI_VECTOR_STORE_ID', '<your_store_id>')}")
    doc.append(f"CUSTOMGPT_PROJECT={env_vars.get('CUSTOMGPT_PROJECT', '<your_project_id>')}")
    doc.append("```")
    doc.append("")

    doc.append("### 4. Run Benchmark")
    doc.append("")
    doc.append("You have two options:")
    doc.append("")
    doc.append("**Option A: Full Academic Pipeline (Recommended)**")
    doc.append("")
    doc.append("Runs benchmark + penalty analysis + forensics + statistical analysis + reproducibility docs:")
    doc.append("```bash")
    total_questions = run_config.get('total_questions', 'N') if run_config else 'N'
    doc.append(f"docker compose run --rm simple-evals python scripts/run_academic_benchmark.py --examples {total_questions}")
    doc.append("```")
    doc.append("")
    doc.append("**Option B: Benchmark Only**")
    doc.append("")
    doc.append("Just runs the core benchmark evaluation (you'll need to run post-processing scripts manually):")
    doc.append("```bash")
    doc.append(f"docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples {total_questions}")
    doc.append("```")
    doc.append("")

    # Notes
    doc.append("## Notes")
    doc.append("")
    doc.append("- This benchmark uses **temperature=0** for deterministic results")
    doc.append("- Minor variations may still occur due to API-level sampling or model updates")
    doc.append("- API rate limits may affect execution time")
    doc.append("- Ensure sufficient API credits before running (estimate: ~$10-20 for 200 questions)")
    doc.append("")

    # Save
    with open(output_file, 'w') as f:
        f.write("\n".join(doc))

    print(f"✓ Markdown documentation saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate reproducibility documentation for benchmark runs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--repo-root', help='Path to repository root (default: auto-detect)')
    parser.add_argument('--output-dir', help='Output directory (default: run directory)')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    repo_root = Path(args.repo_root) if args.repo_root else None
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"Generating Reproducibility Documentation")
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}\n")

    # Generate manifest
    manifest = generate_reproducibility_manifest(run_dir, repo_root)

    # Save JSON manifest
    json_file = output_dir / f"reproducibility_manifest_{run_dir.name}.json"
    with open(json_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ JSON manifest saved: {json_file}")

    # Generate markdown documentation
    md_file = output_dir / f"REPRODUCIBILITY_{run_dir.name}.md"
    generate_markdown_documentation(manifest, md_file)

    print(f"\n{'='*60}")
    print(f"Reproducibility Documentation Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
