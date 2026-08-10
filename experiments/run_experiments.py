"""Main experiment runner for uncertainty quantification."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from llm_uq import Estimator, Benchmark
from llm_uq.datasets import load_builtin
from llm_uq import viz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str):
    return OmegaConf.load(config_path)


def run_experiments(config_path: str):
    """Run uncertainty quantification experiments from a YAML config file.

    Args:
        config_path: Path to experiment configuration YAML.
    """
    logger.info("=" * 80)
    logger.info("Starting Uncertainty Quantification Experiments")
    logger.info("=" * 80)

    cfg = _load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.model.name
    logger.info("Loading model: %s", model_name)

    est = Estimator.from_pretrained(
        model_name,
        device=OmegaConf.select(cfg, "model.device", default=None),
        load_in_8bit=OmegaConf.select(cfg, "model.load_in_8bit", default=False),
        load_in_4bit=OmegaConf.select(cfg, "model.load_in_4bit", default=False),
        semantic_model=OmegaConf.select(cfg, "semantic_model.name", default="all-MiniLM-L6-v2"),
    )

    bench = Benchmark(est)

    methods = list(cfg.uncertainty_methods)
    all_results: dict[str, Any] = {}

    for task_cfg in cfg.tasks:
        task_name = task_cfg.name
        logger.info("\n%s\n[%s] Task: %s\n%s", "=" * 80, model_name, task_name.upper(), "=" * 80)

        try:
            task_data = load_builtin(
                task=task_name,
                dataset=task_cfg.dataset if task_cfg.dataset != "default" else None,
                split=task_cfg.split,
                max_samples=task_cfg.max_samples,
            )
        except Exception as exc:
            logger.error("Failed to load %s dataset: %s", task_name, exc)
            continue

        result = bench.run(
            task=task_name,
            dataset=task_data,
            methods=methods,
            max_new_tokens=cfg.generation.max_length,
            temperature=cfg.generation.temperature,
            num_samples=cfg.generation.num_samples,
        )

        # Build per-method results dict in the same format as the original runner
        task_methods: dict[str, Any] = {}
        for method in methods:
            m_metrics = result.metrics.get(method, {})
            task_methods[method] = {
                **m_metrics,
                "uncertainties": result.uncertainties.get(method, []),
                "errors": result.errors,
            }
        all_results[task_name] = task_methods

        task_dir = results_dir / task_name
        task_figures_dir = task_dir / "figures"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_figures_dir.mkdir(parents=True, exist_ok=True)

        if OmegaConf.select(cfg, "output.save_individual_results", default=True):
            out_path = task_dir / f"{task_name}_results.json"
            result.to_json(str(out_path))
            logger.info("Saved results: %s", out_path)

        # Visualizations
        for method in methods:
            uncertainties = result.uncertainties.get(method, [])
            errors = result.errors
            if not uncertainties:
                continue

            if task_name in ("qa", "math"):
                viz.roc_curve(
                    uncertainties, errors,
                    method_name=f"{task_name} - {method}",
                    save_path=str(task_figures_dir / f"{method}_roc.png"),
                )
            else:
                viz.correlation(
                    uncertainties, [1.0 - e for e in errors],
                    method_name=f"{task_name} - {method}",
                    save_path=str(task_figures_dir / f"{method}_correlation.png"),
                )

            ece_val = result.metrics.get(method, {}).get("ece")
            viz.calibration_curve(
                uncertainties, errors,
                method_name=f"{task_name} - {method}",
                ece=ece_val,
                save_path=str(task_figures_dir / f"{method}_calibration.png"),
            )
            viz.distribution(
                uncertainties,
                method_name=f"{task_name} - {method}",
                save_path=str(task_figures_dir / f"{method}_distribution.png"),
            )

            # Token heatmaps (optional)
            plot_tokens = OmegaConf.select(cfg, "visualization.plot_token_heatmaps", default=False)
            if plot_tokens and result.token_level and method in result.token_level:
                tok_data = result.token_level[method]
                max_tok_samples = OmegaConf.select(
                    cfg, "visualization.token_heatmap_max_samples", default=10
                )
                try:
                    viz.token_heatmap(
                        tok_data["uncertainties"], tok_data["tokens"],
                        method_name=f"{task_name} - {method}",
                        max_samples=max_tok_samples,
                        save_path=str(task_figures_dir / f"{method}_token_heatmap.png"),
                    )
                except Exception as exc:
                    logger.warning("Token heatmap failed for %s/%s: %s", task_name, method, exc)

    # Aggregated results
    agg_path = results_dir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved aggregated results: %s", agg_path)

    # Summary
    logger.info("\n%s\nEXPERIMENT SUMMARY\n%s", "=" * 80, "=" * 80)
    for task_name, task_methods in all_results.items():
        logger.info("\n%s:", task_name.upper())
        for method, data in task_methods.items():
            ece = data.get("ece", float("nan"))
            if task_name in ("qa", "math"):
                auroc = data.get("auroc", float("nan"))
                logger.info("  %-20s | AUROC: %6.3f | ECE: %6.3f", method, auroc, ece)
            else:
                corr = data.get("correlation_spearman", float("nan"))
                logger.info("  %-20s | Corr:  %6.3f | ECE: %6.3f", method, corr, ece)

    logger.info("\n%s\nExperiments completed!\n%s", "=" * 80, "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run uncertainty quantification experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/qwen3_8b.yaml",
        help="Path to configuration YAML",
    )
    args = parser.parse_args()

    try:
        run_experiments(args.config)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Experiment failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
