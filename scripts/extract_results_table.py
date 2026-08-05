#!/usr/bin/env python3
"""
Extract results from JSON files and generate a comprehensive summary table.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_results(results_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all results from JSON files."""
    all_results = {}
    
    # Find all result directories
    for model_dir in results_dir.glob("results_*"):
        model_name = model_dir.name.replace("results_", "").replace("_", "-")
        all_results[model_name] = {}
        
        # Find all task directories
        for task_dir in model_dir.iterdir():
            if task_dir.is_dir() and not task_dir.name == "figures":
                task_name = task_dir.name
                results_file = task_dir / f"{task_name}_results.json"
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        task_results = json.load(f)
                        all_results[model_name][task_name] = task_results.get("methods", {})
    
    return all_results

def extract_metrics(all_results: Dict) -> List[Dict[str, Any]]:
    """Extract key metrics into a flat list."""
    rows = []
    
    for model_name, tasks in all_results.items():
        for task_name, methods in tasks.items():
            for method_name, method_data in methods.items():
                if isinstance(method_data, dict) and "error" not in method_data:
                    row = {
                        "Model": model_name,
                        "Task": task_name,
                        "Method": method_name,
                    }
                    
                    # Binary tasks (QA, Math)
                    if task_name in ["qa", "math"]:
                        row["AUROC"] = method_data.get("auroc", None)
                        row["ECE"] = method_data.get("ece", None)
                        row["Error Rate"] = method_data.get("error_rate", None)
                    
                    # Continuous tasks (Summarization, Open-ended)
                    elif task_name in ["summarization", "open-ended"]:
                        row["Correlation (Spearman)"] = method_data.get("correlation_spearman", None)
                        row["ECE"] = method_data.get("ece", None)
                        row["Error Rate"] = method_data.get("error_rate", None)
                    
                    # Performance metrics (if available)
                    if "perf" in method_data:
                        row["Avg Latency (s)"] = method_data["perf"].get("avg_method_latency_s", None)
                        row["Peak GPU (MB)"] = method_data["perf"].get("avg_method_peak_gpu_mem_mb", None)
                    
                    rows.append(row)
    
    return rows

def format_table_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table."""
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df_formatted = df.copy()
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    
    # Format markdown
    markdown = df_formatted.to_markdown(index=False, tablefmt="pipe")
    return markdown

def generate_summary_table(results_dir: Path) -> str:
    """Generate a comprehensive results table."""
    all_results = load_results(results_dir)
    rows = extract_metrics(all_results)
    
    if not rows:
        return "No results found."
    
    df = pd.DataFrame(rows)
    
    # Create separate tables for binary and continuous tasks
    binary_tasks = df[df["Task"].isin(["qa", "math"])].copy()
    continuous_tasks = df[df["Task"].isin(["summarization", "open-ended"])].copy()
    
    output = []
    output.append("## Comprehensive Results Table\n")
    
    # Binary tasks table
    if not binary_tasks.empty:
        output.append("### Binary Tasks (QA, Math)\n")
        binary_cols = ["Model", "Task", "Method", "AUROC", "ECE", "Error Rate"]
        if "Avg Latency (s)" in binary_tasks.columns:
            binary_cols.append("Avg Latency (s)")
        output.append(format_table_markdown(binary_tasks[binary_cols]))
        output.append("\n")
    
    # Continuous tasks table
    if not continuous_tasks.empty:
        output.append("### Continuous Tasks (Summarization, Open-ended)\n")
        continuous_cols = ["Model", "Task", "Method", "Correlation (Spearman)", "ECE", "Error Rate"]
        if "Avg Latency (s)" in continuous_tasks.columns:
            continuous_cols.append("Avg Latency (s)")
        output.append(format_table_markdown(continuous_tasks[continuous_cols]))
        output.append("\n")
    
    return "\n".join(output)

def main():
    """Main entry point."""
    results_dir = Path(__file__).parent.parent / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    table = generate_summary_table(results_dir)
    print(table)
    
    # Also save to file
    output_file = results_dir / "results_table.md"
    with open(output_file, 'w') as f:
        f.write(table)
    print(f"\nTable saved to: {output_file}")

if __name__ == "__main__":
    main()

