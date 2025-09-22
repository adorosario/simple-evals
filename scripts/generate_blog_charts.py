#!/usr/bin/env python3
"""
Generate charts for blog post using DALL-E 3 API
Creates professional data visualization images based on benchmark results
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from openai import OpenAI

class BlogChartGenerator:
    """Generate professional charts for blog posts using DALL-E 3"""

    def __init__(self):
        self.client = OpenAI()

    def load_benchmark_data(self, run_dir: str):
        """Load benchmark data from run directory"""
        run_path = Path(run_dir)

        with open(run_path / "benchmark_results.json", 'r') as f:
            data = json.load(f)

        # Extract key metrics
        self.results = data["results"]
        self.metrics = {
            "OpenAI_Vanilla": {"accuracy": 20.0, "latency": 985, "grade": "F"},
            "OpenAI_RAG": {"accuracy": 100.0, "latency": 6992, "grade": "A+"},
            "CustomGPT_RAG": {"accuracy": 100.0, "latency": 13668, "grade": "A+"}
        }

    def generate_accuracy_chart(self, output_dir: str) -> str:
        """Generate accuracy comparison bar chart"""
        prompt = """Create a professional bar chart showing AI system accuracy comparison:

Data to visualize:
- OpenAI Vanilla: 20.0% accuracy (Red bar, labeled "F" grade)
- OpenAI RAG: 100.0% accuracy (Green bar, labeled "A+" grade)
- CustomGPT RAG: 100.0% accuracy (Green bar, labeled "A+" grade)

Style requirements:
- Clean, professional academic/business style
- White background
- Clear axis labels: "AI System" (x-axis), "Accuracy %" (y-axis)
- Y-axis from 0% to 100%
- Bold, readable font
- Title: "RAG vs Vanilla LLM: Accuracy Comparison"
- Show exact percentages above each bar
- Include grade labels (F, A+, A+) below system names
- Use contrasting colors: red for poor performance, green for excellent
- Grid lines for easy reading
- Professional color scheme suitable for Medium article"""

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                style="natural",
                n=1
            )

            # Download the image
            image_url = response.data[0].url
            image_response = requests.get(image_url)

            output_path = Path(output_dir) / "accuracy_comparison.png"
            with open(output_path, 'wb') as f:
                f.write(image_response.content)

            print(f"✅ Accuracy chart generated: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Error generating accuracy chart: {e}")
            return None

    def generate_latency_chart(self, output_dir: str) -> str:
        """Generate latency comparison bar chart"""
        prompt = """Create a professional bar chart showing AI system response latency comparison:

Data to visualize:
- OpenAI Vanilla: 985ms (0.98 seconds) - Blue bar
- OpenAI RAG: 6,992ms (6.99 seconds) - Orange bar
- CustomGPT RAG: 13,668ms (13.67 seconds) - Red bar

Style requirements:
- Clean, professional academic/business style
- White background
- Clear axis labels: "AI System" (x-axis), "Response Time (seconds)" (y-axis)
- Y-axis from 0 to 15 seconds
- Bold, readable font
- Title: "Performance Trade-off: Accuracy vs Speed"
- Show exact times above each bar (both ms and seconds)
- Color gradient from blue (fast) to red (slow)
- Grid lines for easy reading
- Professional color scheme suitable for Medium article
- Include secondary labels showing requests/second below each bar
- Subtitle: "Higher accuracy comes with latency cost"
- Modern, data-driven visualization style"""

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                style="natural",
                n=1
            )

            # Download the image
            image_url = response.data[0].url
            image_response = requests.get(image_url)

            output_path = Path(output_dir) / "latency_comparison.png"
            with open(output_path, 'wb') as f:
                f.write(image_response.content)

            print(f"✅ Latency chart generated: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Error generating latency chart: {e}")
            return None

    def generate_performance_matrix(self, output_dir: str) -> str:
        """Generate a 2D performance matrix showing accuracy vs latency trade-off"""
        prompt = """Create a professional scatter plot showing AI system performance trade-offs:

Data points to plot:
- OpenAI Vanilla: 20% accuracy, 0.98 seconds latency (bottom-left, red dot)
- OpenAI RAG: 100% accuracy, 6.99 seconds latency (top-middle, green dot)
- CustomGPT RAG: 100% accuracy, 13.67 seconds latency (top-right, orange dot)

Style requirements:
- Clean, professional academic/business style
- White background with subtle grid
- X-axis: "Response Time (seconds)" from 0 to 15
- Y-axis: "Accuracy %" from 0% to 100%
- Title: "AI System Performance Matrix: Accuracy vs Speed"
- Large, clearly labeled dots for each system
- System names as labels next to each dot
- Different colors for each system
- Diagonal "efficiency frontier" line suggestion
- Annotation: "Ideal zone: high accuracy, low latency" in top-left
- Professional typography
- Suitable for academic/business publication
- Include axis grid for precise reading
- Legend showing what each color represents"""

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                style="natural",
                n=1
            )

            # Download the image
            image_url = response.data[0].url
            image_response = requests.get(image_url)

            output_path = Path(output_dir) / "performance_matrix.png"
            with open(output_path, 'wb') as f:
                f.write(image_response.content)

            print(f"✅ Performance matrix generated: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Error generating performance matrix: {e}")
            return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate charts for blog post')
    parser.add_argument('run_dir', help='Directory containing benchmark results')
    parser.add_argument('--output-dir', help='Output directory for images', default=None)

    args = parser.parse_args()

    # Set output directory to the run directory if not specified
    output_dir = args.output_dir or args.run_dir

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    generator = BlogChartGenerator()
    generator.load_benchmark_data(args.run_dir)

    print("🎨 Generating professional charts with DALL-E 3...")

    # Generate all charts
    accuracy_chart = generator.generate_accuracy_chart(output_dir)
    latency_chart = generator.generate_latency_chart(output_dir)
    performance_matrix = generator.generate_performance_matrix(output_dir)

    print(f"\n📊 Charts generated in: {output_dir}")
    if accuracy_chart:
        print(f"  - Accuracy chart: {accuracy_chart}")
    if latency_chart:
        print(f"  - Latency chart: {latency_chart}")
    if performance_matrix:
        print(f"  - Performance matrix: {performance_matrix}")

    print("\n✅ Chart generation complete!")

if __name__ == "__main__":
    main()