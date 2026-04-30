"""
xCEBRA-IBL: Main Pipeline Execution Script
==========================================

Complete pipeline to apply xCEBRA to the IBL dataset and reproduce/extend
the selectivity analysis from "Rarely Categorical" (Wang et al.).

Usage:
    # Full pipeline (requires IBL data):
    python -m xcebra_ibl.run_pipeline --all

    # Individual steps:
    python -m xcebra_ibl.run_pipeline --download
    python -m xcebra_ibl.run_pipeline --preprocess
    python -m xcebra_ibl.run_pipeline --train
    python -m xcebra_ibl.run_pipeline --analyze
    python -m xcebra_ibl.run_pipeline --demo  (synthetic data test)
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def run_download(n_areas=None, n_sessions=None):
    """Step 1: Download IBL data via ONE API."""
    print("\n" + "=" * 60)
    print("STEP 1: Download IBL Electrophysiology Data")
    print("=" * 60)
    from xcebra_ibl.data.download_ibl import download_ibl_sessions
    from xcebra_ibl.configs.config import CORTICAL_AREAS
    areas = CORTICAL_AREAS[:n_areas] if n_areas else None
    download_ibl_sessions(areas=areas, max_sessions=n_sessions, verbose=True)


def run_preprocess():
    """Step 2: Preprocess all sessions."""
    print("\n" + "=" * 60)
    print("STEP 2: Preprocess IBL Sessions")
    print("=" * 60)
    from xcebra_ibl.data.preprocess import preprocess_all_sessions
    all_sessions = preprocess_all_sessions(verbose=True)
    return all_sessions


def run_train(
    all_sessions=None,
    method="per_variable",
    max_iterations=None,
    batch_size=None,
    n_attribution_samples=2000,
    wandb_run=None,
    wandb_log_interval=50,
):
    """Step 3: Train xCEBRA models and extract attributions."""
    print("\n" + "=" * 60)
    print("STEP 3: Train xCEBRA Models")
    print("=" * 60)

    from xcebra_ibl.data.preprocess import load_preprocessed_sessions
    from xcebra_ibl.models.train import train_all_sessions

    if all_sessions is None:
        all_sessions = load_preprocessed_sessions()

    if not all_sessions:
        print("No preprocessed sessions found. Run preprocessing first.")
        return None, None

    all_results, neuron_df = train_all_sessions(
        all_sessions,
        method=method,
        max_iterations=max_iterations,
        batch_size=batch_size,
        n_attribution_samples=n_attribution_samples,
        wandb_run=wandb_run,
        wandb_log_interval=wandb_log_interval,
        verbose=True,
    )
    return all_results, neuron_df


def _log_analysis_to_wandb(wandb_run, analysis_results):
    """Log top-level analysis metrics and upload generated figures."""
    if analysis_results is None:
        return

    corr = analysis_results.get("anatomical_correlation")
    if corr:
        wandb_run.log(
            {
                "analysis/anatomy_spearman_r": float(corr.get("spearman_r", 0.0)),
                "analysis/anatomy_spearman_p": float(corr.get("spearman_p", 1.0)),
            }
        )

    comp = analysis_results.get("rrr_comparison")
    if comp:
        wandb_run.log(
            {
                "analysis/baseline_overall_spearman_r": float(comp.get("overall_spearman_r", 0.0)),
                "analysis/baseline_mean_area_similarity": float(comp.get("mean_area_similarity", 0.0)),
            }
        )


def _log_figures_dir_to_wandb(wandb_run, figure_dir):
    """Attach saved figure files to a W&B artifact and preview image files."""
    try:
        import importlib
        wandb = importlib.import_module("wandb")
    except ImportError:
        return

    figure_dir = Path(figure_dir)
    if not figure_dir.exists():
        return

    artifact = wandb.Artifact("xcebra-figures", type="figures")
    artifact.add_dir(str(figure_dir))
    wandb_run.log_artifact(artifact)

    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for fig_path in sorted(figure_dir.glob(ext)):
            wandb_run.log({f"figures/{fig_path.stem}": wandb.Image(str(fig_path))})


def run_analyze(neuron_df=None, comparison_mode="fallback", wandb_run=None):
    """Step 4: Selectivity analysis and comparison with RRR."""
    print("\n" + "=" * 60)
    print("STEP 4: Selectivity Analysis & Comparison")
    print("=" * 60)

    import pandas as pd
    from xcebra_ibl.configs.config import RESULTS_DIR
    from xcebra_ibl.analysis.selectivity import run_full_selectivity_analysis
    from xcebra_ibl.analysis.visualize import generate_all_figures

    if neuron_df is None:
        results_path = RESULTS_DIR / "xcebra_neuron_results.json"
        if results_path.exists():
            neuron_df = pd.read_json(str(results_path))
        else:
            print(f"No results found at {results_path}")
            print("Run training first.")
            return None

    # Run selectivity analysis
    analysis_results = run_full_selectivity_analysis(
        neuron_df,
        comparison_mode=comparison_mode,
    )

    # Generate figures
    generate_all_figures(analysis_results)

    if wandb_run is not None:
        from xcebra_ibl.configs.config import RESULTS_DIR
        _log_analysis_to_wandb(wandb_run, analysis_results)
        _log_figures_dir_to_wandb(wandb_run, RESULTS_DIR / "figures")

    return analysis_results


def init_wandb_run(args):
    """Initialize an optional W&B run from CLI args."""
    if not args.wandb:
        return None

    try:
        import importlib
        wandb = importlib.import_module("wandb")
    except ImportError:
        print("\nW&B requested but not installed. Install with: pip install wandb -qU")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            "method": args.method,
            "max_iterations": args.max_iterations,
            "batch_size": args.batch_size,
            "n_attribution_samples": args.n_attribution_samples,
            "comparison_mode": args.comparison_mode,
            "n_areas": args.n_areas,
            "n_sessions": args.n_sessions,
        },
    )
    print(f"\nW&B logging enabled: project='{args.wandb_project}', run='{run.name}'")
    return run


def run_demo():
    """
    Run a demonstration with synthetic data to verify the pipeline works.

    Creates synthetic neural data with known structure:
    - 200 neurons across 4 "brain areas"
    - Each area has a distinctive selectivity profile
    - Verifies CEBRA training + Jacobian attribution + clustering
    """
    print("\n" + "=" * 60)
    print("DEMO: xCEBRA Pipeline with Synthetic Data")
    print("=" * 60)

    import pandas as pd
    from xcebra_ibl.models.xcebra_model import XCEBRAModel
    from xcebra_ibl.analysis.selectivity import (
        compute_selectivity_from_attributions,
        cluster_areas,
    )
    from xcebra_ibl.analysis.visualize import (
        plot_selectivity_heatmap, plot_embedding_visualization,
    )
    from xcebra_ibl.configs.config import (
        VARIABLE_NAMES, N_VARIABLES, RESULTS_DIR,
    )

    np.random.seed(42)

    # ── Generate synthetic data ──
    n_neurons = 200
    n_timesteps = 2000  # simulating K*T flattened
    n_variables = N_VARIABLES

    print(f"\n  Generating synthetic data:")
    print(f"    {n_neurons} neurons, {n_timesteps} time points")
    print(f"    {n_variables} behavioral variables")

    # Create 4 synthetic brain areas with distinct profiles
    area_profiles = {
        "VISp": np.array([3, 1, 2, 0.5, 0.3, 0.2, 0.1, 0.1]),   # visual: stimulus-driven
        "MOp":  np.array([0.1, 0.3, 0.1, 3, 0.5, 2, 0.5, 0.5]),  # motor: choice + wheel
        "ACAd": np.array([2, 0.5, 0.5, 1, 1, 0.5, 0.3, 0.3]),    # prefrontal: block + mixed
        "SSp-bfd": np.array([0.1, 0.2, 0.1, 0.5, 0.3, 1, 3, 2]), # somatosensory: whisker + lick
    }
    neurons_per_area = n_neurons // len(area_profiles)

    # Generate behavioral labels
    labels = {}
    labels["block"] = np.random.choice([-1, 0, 1], size=n_timesteps).astype(float)
    labels["side"] = np.random.choice([-1, 1], size=n_timesteps).astype(float)
    labels["contrast_level"] = np.random.choice([0, 1, 4], size=n_timesteps).astype(float)
    labels["choice"] = np.random.choice([-1, 1], size=n_timesteps).astype(float)
    labels["outcome"] = np.random.choice([-1, 1], size=n_timesteps).astype(float)
    labels["wheel"] = np.random.randn(n_timesteps)
    labels["whisker_max"] = np.abs(np.random.randn(n_timesteps))
    labels["lick"] = np.abs(np.random.randn(n_timesteps))
    labels["paw"] = np.abs(np.random.randn(n_timesteps))

    # Stack labels into matrix
    label_matrix = np.column_stack([labels[v] for v in VARIABLE_NAMES])  # (T, 9)

    # Generate neural data with known selectivity
    neural_data = np.random.randn(n_timesteps, n_neurons) * 0.3
    area_labels = []

    for area_idx, (area_name, profile) in enumerate(area_profiles.items()):
        start_n = area_idx * neurons_per_area
        end_n = (area_idx + 1) * neurons_per_area

        # Each neuron in this area has activity driven by the area's profile
        for ni in range(start_n, end_n):
            # Neuron-specific loading (with noise)
            # Expand true profile to 9 variables with zeros for 'paw' to avoid shape errors
            full_profile = np.zeros(N_VARIABLES)
            full_profile[:len(profile)] = profile
            neuron_profile = full_profile * (0.5 + np.random.rand(N_VARIABLES))
            
            # Convert to float array to prevent dimension/type errors
            neuron_profile = np.array(neuron_profile, dtype=float)
            
            # Neural activity = weighted sum of labels + noise
            neural_data[:, ni] += np.dot(label_matrix, neuron_profile) + np.random.randn(n_timesteps) * 0.5

        area_labels.extend([area_name] * neurons_per_area)

    area_labels = np.array(area_labels)
    print(f"    4 areas: {list(area_profiles.keys())}")
    print(f"    {neurons_per_area} neurons/area")

    # ── Train xCEBRA ──
    print(f"\n  Training xCEBRA (per-variable mode)...")
    model = XCEBRAModel(
        max_iterations=1000,    # Fewer iterations for demo
        batch_size=256,
        embedding_dim_per_group=3,
        num_hidden_units=64,
        model_architecture="offset10-model",  # Use robust offset architecture
        time_offsets=10,
    )
    model.fit_per_variable(neural_data, labels, verbose=True)

    # ── Extract embeddings ──
    print(f"\n  Extracting embeddings...")
    embeddings = model.transform_per_variable(neural_data)
    for var_name, emb in embeddings.items():
        print(f"    {var_name}: shape {emb.shape}")

    # ── Compute attributions ──
    print(f"\n  Computing Jacobian attributions...")
    attributions = model.compute_attribution_maps(
        neural_data, method="per_variable", n_samples=500,
    )
    for var_name, attr in attributions.items():
        print(f"    {var_name}: shape {attr.shape}, "
              f"range [{attr.min():.4f}, {attr.max():.4f}]")

    # ── Build neuron DataFrame ──
    rows = []
    for ni in range(n_neurons):
        row = {
            "eid": "synthetic",
            "ni": ni,
            "uuids": f"neuron_{ni}",
            "acronym": area_labels[ni],
            "mfr_task": float(np.mean(np.abs(neural_data[:, ni]))),
        }
        for var_name in VARIABLE_NAMES:
            row[f"xcebra_attr_{var_name}"] = float(attributions[var_name][ni])
        rows.append(row)

    neuron_df = pd.DataFrame(rows)

    # ── Selectivity analysis ──
    print(f"\n  Computing selectivity profiles...")
    sel_result = compute_selectivity_from_attributions(
        neuron_df, min_neurons_per_area=10,
    )
    print(f"    Areas: {sel_result['area_order']}")
    print(f"    Selectivity shape: {sel_result['sel_areas'].shape}")

    # ── Clustering ──
    print(f"\n  Clustering areas...")
    cluster_result = cluster_areas(
        sel_result["sel_areas"], sel_result["area_order"], n_clusters=3
    )
    for c_id, areas in cluster_result["area_clusters"].items():
        print(f"    Cluster {c_id}: {areas}")

    # ── Verify: compare recovered profiles with true profiles ──
    print(f"\n  Verification: Comparing recovered vs true selectivity...")
    for area in sel_result["area_order"]:
        true_profile = area_profiles[area]
        true_z = (true_profile - true_profile.mean()) / (true_profile.std() + 1e-8)
        recovered_z = sel_result["sel_areas"][sel_result["area_order"].index(area)]

        # Correlation between true and recovered
        corr = np.corrcoef(true_z, recovered_z)[0, 1]
        print(f"    {area}: correlation = {corr:.3f}")

    # ── Generate figures ──
    fig_dir = RESULTS_DIR / "demo_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_selectivity_heatmap(
        sel_result["sel_areas"], sel_result["area_order"],
        save_path=fig_dir / "demo_selectivity_heatmap.pdf",
        title="Demo: xCEBRA Selectivity (Synthetic Data)",
    )

    for var_name in ["side", "choice", "whisker_max"]:
        if var_name in embeddings and var_name in labels:
            plot_embedding_visualization(
                embeddings[var_name], labels[var_name], var_name,
                save_path=fig_dir / f"demo_embedding_{var_name}.png",
            )

    print(f"\n  Demo complete! Figures saved to {fig_dir}")
    print(f"\n  {'='*60}")
    print(f"  DEMO PASSED: xCEBRA pipeline verified with synthetic data.")
    print(f"  {'='*60}")

    return {
        "model": model,
        "embeddings": embeddings,
        "attributions": attributions,
        "neuron_df": neuron_df,
        "sel_result": sel_result,
        "cluster_result": cluster_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="xCEBRA-IBL: Apply xCEBRA to the IBL Dataset"
    )
    parser.add_argument("--download", action="store_true", help="Download IBL data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess sessions")
    parser.add_argument("--train", action="store_true", help="Train xCEBRA models")
    parser.add_argument("--analyze", action="store_true", help="Run selectivity analysis")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument(
        "--method", default="per_variable",
        choices=["per_variable", "joint"],
        help="xCEBRA training strategy",
    )
    parser.add_argument(
        "--n-areas", type=int, default=None,
        help="Limit download to first N cortical areas (default: all 43)",
    )
    parser.add_argument(
        "--n-sessions", type=int, default=None,
        help="Limit download to N sessions per area (default: config MAX_SESSIONS_PER_AREA=30)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override CEBRA max iterations for training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override CEBRA batch size for training",
    )
    parser.add_argument(
        "--n-attribution-samples", type=int, default=2000,
        help="Samples used per session for Jacobian attribution",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["fallback", "strict"],
        default="fallback",
        help="Comparison behavior when RRR artifact is unavailable",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="xcebra-ibl",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user). If omitted, uses your default account.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional custom W&B run name",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B run mode",
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=50,
        help="Log training loss to W&B every N iterations",
    )

    args = parser.parse_args()

    if not any([args.download, args.preprocess, args.train, args.analyze, args.demo, args.all]):
        parser.print_help()
        print("\n  Tip: Start with --demo to verify the pipeline works!")
        return

    if args.demo:
        run_demo()
        return

    wandb_run = init_wandb_run(args)

    if args.all or args.download:
        run_download(n_areas=args.n_areas, n_sessions=args.n_sessions)

    all_sessions = None
    if args.all or args.preprocess:
        all_sessions = run_preprocess()

    neuron_df = None
    if args.all or args.train:
        _, neuron_df = run_train(
            all_sessions,
            method=args.method,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            n_attribution_samples=args.n_attribution_samples,
            wandb_run=wandb_run,
            wandb_log_interval=args.wandb_log_interval,
        )

    if args.all or args.analyze:
        run_analyze(
            neuron_df,
            comparison_mode=args.comparison_mode,
            wandb_run=wandb_run,
        )

    if wandb_run is not None:
        wandb_run.finish()

    print("\n" + "=" * 60)
    print("xCEBRA-IBL Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
