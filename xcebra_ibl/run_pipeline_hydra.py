"""Hydra-backed entrypoint for the xCEBRA-IBL pipeline.

Use this when you want to override model or pipeline settings from the CLI
without hardcoding them into Python.

Examples:
    python -m xcebra_ibl.run_pipeline_hydra
    python -m xcebra_ibl.run_pipeline_hydra pipeline.all=true
    python -m xcebra_ibl.run_pipeline_hydra cebra.max_iterations=500 cebra.batch_size=256
    python -m xcebra_ibl.run_pipeline_hydra pipeline.wandb=true pipeline.wandb_project=thesis-xcebra
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
import hydra

from xcebra_ibl.run_pipeline import (
    run_download,
    run_preprocess,
    run_train,
    run_analyze,
    run_demo,
)


def _cfg_value(cfg: DictConfig, path: str, default=None):
    value = OmegaConf.select(cfg, path)
    return default if value is None else value


@hydra.main(version_base=None, config_path="configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if _cfg_value(cfg, "pipeline.demo", False):
        run_demo()
        return

    if _cfg_value(cfg, "pipeline.all", False) or _cfg_value(cfg, "pipeline.download", False):
        run_download(
            n_areas=_cfg_value(cfg, "pipeline.n_areas", None),
            n_sessions=_cfg_value(cfg, "pipeline.n_sessions", None),
        )

    all_sessions = None
    if _cfg_value(cfg, "pipeline.all", False) or _cfg_value(cfg, "pipeline.preprocess", False):
        all_sessions = run_preprocess()

    wandb_run = None
    if _cfg_value(cfg, "pipeline.wandb", False):
        # Import lazily so this entrypoint still works without wandb enabled.
        from xcebra_ibl.run_pipeline import init_wandb_run

        wandb_run = init_wandb_run(
            type(
                "Args",
                (),
                {
                    "wandb": True,
                    "wandb_project": _cfg_value(cfg, "pipeline.wandb_project", "xcebra-ibl"),
                    "wandb_entity": _cfg_value(cfg, "pipeline.wandb_entity", None),
                    "wandb_run_name": _cfg_value(cfg, "pipeline.wandb_run_name", None),
                    "wandb_mode": _cfg_value(cfg, "pipeline.wandb_mode", "online"),
                    "method": _cfg_value(cfg, "cebra.method", "per_variable"),
                    "max_iterations": _cfg_value(cfg, "cebra.max_iterations", None),
                    "batch_size": _cfg_value(cfg, "cebra.batch_size", None),
                    "n_attribution_samples": _cfg_value(cfg, "cebra.n_attribution_samples", 2000),
                    "comparison_mode": _cfg_value(cfg, "pipeline.comparison_mode", "fallback"),
                    "n_areas": _cfg_value(cfg, "pipeline.n_areas", None),
                    "n_sessions": _cfg_value(cfg, "pipeline.n_sessions", None),
                },
            )
        )

    if _cfg_value(cfg, "pipeline.all", False) or _cfg_value(cfg, "pipeline.train", False):
        _, neuron_df = run_train(
            all_sessions,
            method=_cfg_value(cfg, "cebra.method", "per_variable"),
            max_iterations=_cfg_value(cfg, "cebra.max_iterations", None),
            batch_size=_cfg_value(cfg, "cebra.batch_size", None),
            n_attribution_samples=_cfg_value(cfg, "cebra.n_attribution_samples", 2000),
            seed=_cfg_value(cfg, "pipeline.seed", 2025),
            model_kwargs={
                "model_architecture": _cfg_value(cfg, "cebra.model_architecture", "offset10-model"),
                "learning_rate": _cfg_value(cfg, "cebra.learning_rate", 3e-4),
                "temperature": _cfg_value(cfg, "cebra.temperature", 1.0),
                "num_hidden_units": _cfg_value(cfg, "cebra.num_hidden_units", 128),
                "time_offsets": _cfg_value(cfg, "cebra.time_offsets", 10),
                "embedding_dim_per_group": _cfg_value(cfg, "cebra.embedding_dim_per_group", 4),
            },
            wandb_run=wandb_run,
            wandb_log_interval=_cfg_value(cfg, "pipeline.wandb_log_interval", 50),
        )
    else:
        neuron_df = None

    if _cfg_value(cfg, "pipeline.all", False) or _cfg_value(cfg, "pipeline.analyze", False):
        run_analyze(
            neuron_df,
            comparison_mode=_cfg_value(cfg, "pipeline.comparison_mode", "fallback"),
            wandb_run=wandb_run,
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
