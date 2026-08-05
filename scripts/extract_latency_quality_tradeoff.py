#!/usr/bin/env python3
"""
Extract latency vs quality trade-off data from aggregated results.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_aggregated_results(results_dir: Path) -> Dict:
    """Load aggregated results from all models."""
    all_results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("results_"):
            continue
        
        model_name = model_dir.name.replace("results_", "")
        aggregated_file = model_dir / "aggregated_results.json"
        
        if aggregated_file.exists():
            with open(aggregated_file, "r") as f:
                all_results[model_name] = json.load(f)
    
    return all_results


def extract_latency_quality_data(all_results: Dict) -> List[Dict]:
    """Extract latency and quality metrics for all methods across all models and tasks."""
    rows = []
    
    method_names = {
        "entropy": "Entropy",
        "max_probability": "Max Probability",
        "sequence_probability": "Sequence Probability",
        "self_consistency": "Self-Consistency",
        "self_reflection": "Self-Reflection",
        "bsdetector": "BSDetector"
    }
    
    task_names = {
        "qa": "Factual QA (SQuAD)",
        "math": "Math Problems (GSM8K)",
        "summarization": "Summarization (CNN/DailyMail)",
        "open-ended": "Open-ended Generation (WritingPrompts)"
    }
    
    model_short = {
        "ministral-8b": "Ministral",
        "qwen3-4b": "Qwen3-4B",
        "qwen3-8b": "Qwen3-8B"
    }
    
    for model_name, model_results in all_results.items():
        model_display = model_short.get(model_name, model_name)
        
        for task_name, task_results in model_results.items():
            if task_name not in task_names:
                continue
            
            task_display = task_names[task_name]
            
            for method_name, method_results in task_results.items():
                if method_name not in method_names:
                    continue
                
                method_display = method_names[method_name]
                
                # Get latency
                perf = method_results.get("perf", {})
                latency = perf.get("avg_method_latency_s")
                
                if latency is None:
                    continue
                
                # Get quality metrics
                if task_name in ["qa", "math"]:
                    quality_metric = method_results.get("auroc")
                    quality_name = "AUROC"
                else:
                    quality_metric = method_results.get("correlation_spearman")
                    quality_name = "Correlation"
                
                ece = method_results.get("ece")
                
                rows.append({
                    "Model": model_display,
                    "Task": task_display,
                    "Method": method_display,
                    "Latency (s)": latency,
                    "Quality Metric": quality_name,
                    "Quality Value": quality_metric,
                    "ECE": ece
                })
    
    return rows


def generate_latency_quality_table(rows: List[Dict]) -> str:
    """Generate a markdown table showing latency vs quality trade-offs."""
    if not rows:
        return "No latency data found."
    
    df = pd.DataFrame(rows)
    
    # Group by method and compute averages across models and tasks
    method_stats = []
    
    for method in df["Method"].unique():
        method_df = df[df["Method"] == method]
        
        avg_latency = method_df["Latency (s)"].mean()
        # For quality, we need to handle AUROC and Correlation separately
        # Let's compute average quality across all tasks (normalize if needed)
        avg_quality = method_df["Quality Value"].mean()
        avg_ece = method_df["ECE"].mean()
        
        # Count tasks where this method performs well
        # For binary tasks: AUROC > 0.7 is good
        # For continuous tasks: Correlation > 0.2 is good
        binary_tasks = method_df[method_df["Quality Metric"] == "AUROC"]
        continuous_tasks = method_df[method_df["Quality Metric"] == "Correlation"]
        
        good_binary = (binary_tasks["Quality Value"] > 0.7).sum() if len(binary_tasks) > 0 else 0
        good_continuous = (continuous_tasks["Quality Value"] > 0.2).sum() if len(continuous_tasks) > 0 else 0
        
        method_stats.append({
            "Method": method,
            "Avg Latency (s)": avg_latency,
            "Avg Quality": avg_quality,
            "Avg ECE": avg_ece,
            "Good Performance Count": good_binary + good_continuous
        })
    
    # Sort by latency
    method_stats_df = pd.DataFrame(method_stats)
    method_stats_df = method_stats_df.sort_values("Avg Latency (s)")
    
    # Generate table
    output = []
    output.append("### Latency vs Quality Trade-off\n\n")
    output.append("*Average latency and quality metrics across all models and tasks.*\n\n")
    
    output.append("| Method | Avg Latency (s) | Avg Quality | Avg ECE | Good Performance Count |")
    output.append("|:-------|:---------------:|:-----------:|:-------:|:----------------------:|")
    
    for _, row in method_stats_df.iterrows():
        method = row["Method"]
        latency = row["Avg Latency (s)"]
        quality = row["Avg Quality"]
        ece = row["Avg ECE"]
        good_count = int(row["Good Performance Count"])
        
        # Format latency appropriately
        if latency < 0.001:
            latency_str = f"{latency*1000:.3f} ms"
        elif latency < 1.0:
            latency_str = f"{latency*1000:.1f} ms"
        else:
            latency_str = f"{latency:.3f} s"
        
        output.append(f"| {method} | {latency_str} | {quality:.3f} | {ece:.3f} | {good_count} |")
    
    output.append("\n")
    
    # Also create a per-task breakdown
    output.append("#### Per-Task Latency vs Quality\n\n")
    output.append("*Detailed breakdown by task showing latency and quality metrics.*\n\n")
    
    for task in df["Task"].unique():
        task_df = df[df["Task"] == task].copy()
        task_df = task_df.sort_values("Latency (s)")
        
        output.append(f"##### {task}\n\n")
        output.append("| Method | Model | Latency (s) | Quality | ECE |")
        output.append("|:-------|:-----|:-----------:|:-------:|:---:|")
        
        for _, row in task_df.iterrows():
            method = row["Method"]
            model = row["Model"]  # Already formatted as display name
            latency = row["Latency (s)"]
            quality = row["Quality Value"]
            ece = row["ECE"]
            
            # Format latency
            if latency < 0.001:
                latency_str = f"{latency*1000:.3f} ms"
            elif latency < 1.0:
                latency_str = f"{latency*1000:.1f} ms"
            else:
                latency_str = f"{latency:.3f} s"
            
            quality_str = f"{quality:.3f}" if pd.notna(quality) else "N/A"
            ece_str = f"{ece:.3f}" if pd.notna(ece) else "N/A"
            
            output.append(f"| {method} | {model} | {latency_str} | {quality_str} | {ece_str} |")
        
        output.append("\n")
    
    return "\n".join(output)


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    all_results = load_aggregated_results(results_dir)
    
    if not all_results:
        print("No aggregated results found.")
        return
    
    rows = extract_latency_quality_data(all_results)
    
    if not rows:
        print("No latency data found in results.")
        return
    
    table = generate_latency_quality_table(rows)
    print(table)


if __name__ == "__main__":
    main()

