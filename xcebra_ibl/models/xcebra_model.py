"""
xCEBRA Model: Explainable CEBRA with Multi-Group Embeddings and Jacobian Regularization.

This module implements the xCEBRA methodology for the IBL dataset:

1. **Multi-group embedding**: The encoder f = [f₁; ...; f_G] splits its output
   into G groups (one per behavioral variable), trained with separate InfoNCE
   losses so each subspace captures a single variable's effect.

2. **Jacobian regularization**: ‖∂f/∂x‖² encourages sparsity in the
   input-to-embedding mapping, so each embedding dimension depends on a
   minimal set of neurons → interpretable attribution maps.

3. **Attribution extraction**: After training, compute the per-neuron Jacobian
   for each variable group to get an analog of the RRR β coefficients.

xCEBRA replaces the linear RRR model with:
    - A nonlinear encoder (temporal convolutional network)
    - Contrastive loss instead of reconstruction loss
    - Jacobian-based attribution instead of linear coefficients
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import cebra
from cebra import CEBRA

from xcebra_ibl.configs.config import (
    EMBEDDING_DIM_PER_GROUP, TOTAL_EMBEDDING_DIM,
    N_VARIABLES, VARIABLE_NAMES, VARIABLE_DISPLAY_NAMES,
    MODEL_ARCHITECTURE, MAX_ITERATIONS, BATCH_SIZE,
    LEARNING_RATE, TEMPERATURE, NUM_HIDDEN_UNITS, TIME_OFFSETS,
    JACOBIAN_REG_WEIGHT, JACOBIAN_N_PROJ, MODELS_DIR,
)


class XCEBRAModel:
    """
    xCEBRA wrapper for IBL neural data analysis.

    Trains a CEBRA model with multi-group embeddings using separate
    behavioral variables as auxiliary labels, then extracts per-neuron
    attribution maps (Jacobians) as the nonlinear analog of RRR β coefficients.

    The key insight: instead of fitting β_{n,v,t} via regression, we learn
    an embedding f(x) where each subspace f_g(x) is aligned to variable g,
    and then compute ∂f_g/∂x_n as the "selectivity" of neuron n for variable g.

    Parameters
    ----------
    embedding_dim_per_group : int
        Embedding dimensions allocated to each behavioral variable.
    model_architecture : str
        CEBRA model architecture name.
    max_iterations : int
        Training iterations.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Optimizer learning rate.
    temperature : float
        InfoNCE temperature.
    num_hidden_units : int
        Hidden layer size in the encoder.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        embedding_dim_per_group: int = EMBEDDING_DIM_PER_GROUP,
        model_architecture: str = MODEL_ARCHITECTURE,
        max_iterations: int = MAX_ITERATIONS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        temperature: float = TEMPERATURE,
        num_hidden_units: int = NUM_HIDDEN_UNITS,
        time_offsets: int = TIME_OFFSETS,
        device: str = "cuda_if_available",
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 1,
        checkpoint_retention: Optional[int] = None,
    ):
        self.embedding_dim_per_group = embedding_dim_per_group
        self.n_groups = N_VARIABLES
        self.total_dim = embedding_dim_per_group * N_VARIABLES
        self.model_architecture = model_architecture
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.num_hidden_units = num_hidden_units
        self.time_offsets = time_offsets
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_retention = checkpoint_retention

        # Will be set during fit
        self.models_ = {}           # one CEBRA model per variable
        self.joint_model_ = None    # single CEBRA model with all labels
        self.is_fitted_ = False
        self.training_losses_ = {}

    def _checkpoint_callback(self, prefix):
        """Build a CEBRA callback that saves the solver after each mini-batch."""
        if self.checkpoint_dir is None:
            return None
        if self.checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be at least 1")
        if self.checkpoint_retention is not None and self.checkpoint_retention < 1:
            raise ValueError("checkpoint_retention must be at least 1 or None")

        checkpoint_dir = self.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def save_checkpoint(num_steps, solver):
            filename = f"{prefix}_step_{num_steps:08d}.pt"
            solver.save(str(checkpoint_dir), filename)
            if self.checkpoint_retention is not None:
                checkpoints = sorted(checkpoint_dir.glob(f"{prefix}_step_*.pt"))
                for old_checkpoint in checkpoints[:-self.checkpoint_retention]:
                    old_checkpoint.unlink()
            print(f"  Saved mini-batch checkpoint: {checkpoint_dir / filename}")

        return save_checkpoint

    def fit_per_variable(
        self,
        neural_data: np.ndarray,
        labels: Dict[str, np.ndarray],
        verbose: bool = True,
    ):
        """
        Strategy A: Train a separate CEBRA model per behavioral variable.

        This is the simplest xCEBRA approach: each model's embedding captures
        the effect of one variable, and attribution maps are extracted per model.

        Parameters
        ----------
        neural_data : (n_samples, n_neurons) array
            Flattened neural activity (K*T rows, N columns).
        labels : dict
            {variable_name: (n_samples,) or (n_samples, 1) label array}
        verbose : bool
        """
        self.models_ = {}
        self.training_losses_ = {}

        for var_idx, var_name in enumerate(VARIABLE_NAMES):
            if var_name not in labels:
                if verbose:
                    print(f"  Warning: {var_name} not in labels, skipping")
                continue

            if verbose:
                print(f"\n  Training CEBRA for variable {var_idx + 1}/{N_VARIABLES}: {var_name}")

            y = labels[var_name]
            if y.ndim == 2:
                y = y.ravel()

            model = CEBRA(
                model_architecture=self.model_architecture,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                temperature=self.temperature,
                output_dimension=self.embedding_dim_per_group,
                max_iterations=self.max_iterations,
                num_hidden_units=self.num_hidden_units,
                time_offsets=self.time_offsets,
                device=self.device,
                verbose=verbose,
            )

            # Determine if label is discrete or continuous
            unique_vals = np.unique(y[~np.isnan(y)])
            is_discrete = len(unique_vals) <= 10

            callback = self._checkpoint_callback(var_name)
            fit_kwargs = {}
            if callback is not None:
                fit_kwargs = {
                    "callback": callback,
                    "callback_frequency": self.checkpoint_frequency,
                }

            if is_discrete:
                # Discrete labels → use as-is (CEBRA handles discrete labels)
                model.fit(neural_data, y, **fit_kwargs)
            else:
                # Continuous labels → pass as 2D array
                y_2d = y.reshape(-1, 1)
                model.fit(neural_data, y_2d, **fit_kwargs)

            self.models_[var_name] = model
            self.training_losses_[var_name] = (
                model.state_dict_["loss"] if hasattr(model, "state_dict_") else []
            )

        self.is_fitted_ = True
        if verbose:
            print(f"\n  All {len(self.models_)} variable models trained.")

    def fit_joint(
        self,
        neural_data: np.ndarray,
        labels: Dict[str, np.ndarray],
        verbose: bool = True,
    ):
        """
        Strategy B: Train a single CEBRA model with ALL behavioral variables
        as joint auxiliary labels. The embedding space is then split into groups.

        This leverages CEBRA's multi-label support: passing multiple label arrays
        creates a joint positive distribution over all variables.

        Parameters
        ----------
        neural_data : (n_samples, n_neurons) array
        labels : dict of label arrays
        verbose : bool
        """
        if verbose:
            print("  Training joint xCEBRA model with all behavioral labels")

        # Separate discrete and continuous labels
        discrete_labels = []
        continuous_labels = []

        for var_name in VARIABLE_NAMES:
            if var_name not in labels:
                continue
            y = labels[var_name]
            if y.ndim == 2:
                y = y.ravel()

            unique_vals = np.unique(y[~np.isnan(y)])
            if len(unique_vals) <= 10:
                discrete_labels.append(y.astype(int))
            else:
                continuous_labels.append(y.reshape(-1, 1))

        # Build y arguments for CEBRA.fit
        # CEBRA expects: continuous labels as 2D arrays, one discrete array
        y_args = []
        for cl in continuous_labels:
            y_args.append(cl)
        # Combine discrete labels into a single compound label
        if discrete_labels:
            # Create compound discrete label by encoding combinations
            compound = discrete_labels[0].copy()
            for dl in discrete_labels[1:]:
                compound = compound * 100 + dl  # simple encoding
            y_args.append(compound)

        model = CEBRA(
            model_architecture=self.model_architecture,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            output_dimension=self.total_dim,
            max_iterations=self.max_iterations,
            num_hidden_units=self.num_hidden_units,
            time_offsets=self.time_offsets,
            device=self.device,
            verbose=verbose,
            hybrid=True,  # Use hybrid mode when mixing discrete + continuous
        )

        callback = self._checkpoint_callback("joint")
        fit_kwargs = {}
        if callback is not None:
            fit_kwargs = {
                "callback": callback,
                "callback_frequency": self.checkpoint_frequency,
            }
        model.fit(neural_data, *y_args, **fit_kwargs)
        self.joint_model_ = model
        self.is_fitted_ = True

        if verbose:
            print("  Joint model training complete.")

    def transform_per_variable(
        self, neural_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings from per-variable models.

        Returns
        -------
        dict : {var_name: (n_samples, embedding_dim_per_group)}
        """
        embeddings = {}
        for var_name, model in self.models_.items():
            embeddings[var_name] = model.transform(neural_data)
        return embeddings

    def transform_joint(self, neural_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get embeddings from joint model, split into per-variable groups.

        Returns
        -------
        dict : {var_name: (n_samples, embedding_dim_per_group)}
        """
        if self.joint_model_ is None:
            raise RuntimeError("Joint model not fitted. Call fit_joint() first.")

        full_embedding = self.joint_model_.transform(neural_data)  # (n_samples, total_dim)

        embeddings = {}
        for g, var_name in enumerate(VARIABLE_NAMES):
            start = g * self.embedding_dim_per_group
            end = (g + 1) * self.embedding_dim_per_group
            embeddings[var_name] = full_embedding[:, start:end]
        return embeddings

    def compute_attribution_maps(
        self,
        neural_data: np.ndarray,
        method: str = "per_variable",
        n_samples: int = 1000,
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-neuron attribution maps via Jacobian computation.

        For each variable group g and each neuron n, compute:
            A_{g,n} = E_x[‖∂f_g/∂x_n‖²]

        This gives the xCEBRA analog of the RRR selectivity:
        instead of |β_{n,v,t}|, we get the mean squared Jacobian.

        Parameters
        ----------
        neural_data : (n_total_samples, n_neurons)
        method : "per_variable" or "joint"
        n_samples : int, number of samples to average over
        batch_size : int

        Returns
        -------
        attribution_maps : dict
            {var_name: (n_neurons,) attribution scores}
        """
        if method == "per_variable" and self.models_:
            return self._compute_attributions_per_variable(
                neural_data, n_samples, batch_size
            )
        elif method == "joint" and self.joint_model_ is not None:
            return self._compute_attributions_joint(
                neural_data, n_samples, batch_size
            )
        else:
            raise RuntimeError(
                f"Method '{method}' not available. Fit the corresponding model first."
            )

    def _compute_attributions_per_variable(
        self, neural_data, n_samples, batch_size
    ) -> Dict[str, np.ndarray]:
        """Compute Jacobian-based attributions from per-variable models."""
        N = neural_data.shape[1]  # n_neurons
        attribution_maps = {}

        # Subsample if needed
        if n_samples < neural_data.shape[0]:
            idx = np.random.choice(neural_data.shape[0], n_samples, replace=False)
            data_subset = neural_data[idx]
        else:
            data_subset = neural_data

        for var_name, model in self.models_.items():
            print(f"  Computing Jacobian attribution for: {var_name}")
            attr = self._jacobian_attribution(model, data_subset, batch_size)
            attribution_maps[var_name] = attr  # (n_neurons,)

        return attribution_maps

    def _compute_attributions_joint(
        self, neural_data, n_samples, batch_size
    ) -> Dict[str, np.ndarray]:
        """Compute Jacobian-based attributions from joint model, per group."""
        N = neural_data.shape[1]
        attribution_maps = {}

        if n_samples < neural_data.shape[0]:
            idx = np.random.choice(neural_data.shape[0], n_samples, replace=False)
            data_subset = neural_data[idx]
        else:
            data_subset = neural_data

        model = self.joint_model_

        for g, var_name in enumerate(VARIABLE_NAMES):
            start = g * self.embedding_dim_per_group
            end = (g + 1) * self.embedding_dim_per_group
            print(f"  Computing Jacobian attribution for group {g}: {var_name}")
            attr = self._jacobian_attribution(
                model, data_subset, batch_size, output_slice=(start, end)
            )
            attribution_maps[var_name] = attr

        return attribution_maps

    def _jacobian_attribution(
        self,
        cebra_model: CEBRA,
        data: np.ndarray,
        batch_size: int = 256,
        output_slice: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute mean squared Jacobian ‖∂f/∂x‖² per input neuron.

        For each sample x, computes the Jacobian J = ∂f(x)/∂x, then
        averages ‖J[:, n]‖² across samples to get attribution per neuron n.

        Parameters
        ----------
        cebra_model : fitted CEBRA model
        data : (n_samples, n_features)
        batch_size : int
        output_slice : (start, end) to select output dimensions (for joint model)

        Returns
        -------
        attributions : (n_features,) mean squared Jacobian per input feature
        """
        # Access the underlying PyTorch model
        solver = cebra_model.solver_
        net = solver.model

        # Determine device
        device = next(net.parameters()).device

        net.eval()
        n_samples, n_features = data.shape
        accumulated_jsq = np.zeros(n_features)
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_np = data[start_idx:end_idx]
            # Reshape based on model architecture
            if "offset" in self.model_architecture:
                # Offset models expect (1, n_features, T) where T is batch_size
                batch = torch.tensor(batch_np.T[None, ...], dtype=torch.float32, device=device)
            else:
                # Standard models expect (batch_size, n_features)
                batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
                
            batch.requires_grad_(True)

            # Forward pass
            embedding = net(batch)

            # Select output dimensions if specified
            if output_slice is not None:
                if embedding.ndim == 3:
                    embedding = embedding[:, output_slice[0]:output_slice[1], :]
                else:
                    embedding = embedding[:, output_slice[0]:output_slice[1]]

            # Compute Jacobian attribution via backward pass over output dims
            D_out = embedding.shape[1]
            jacobian_sq_sum = torch.zeros(n_features, device=device)

            for d in range(D_out):
                scalar = embedding[:, d, :].mean() if embedding.ndim == 3 else embedding[:, d].mean()
                grad = torch.autograd.grad(
                    scalar,
                    batch,
                    retain_graph=(d < D_out - 1),
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # grad shape: (1, n_features, T)
                jacobian_sq_sum += grad.pow(2).mean(dim=(0, 2)).detach()

            accumulated_jsq += jacobian_sq_sum.cpu().numpy()
            n_batches += 1

        # Average across batches (already averaged across T within each batch)
        attributions = accumulated_jsq / max(n_batches, 1)

        # Normalize: note accumulated_jsq already accounts for all output dims
        return attributions

    def compute_selectivity_profiles(
        self,
        neural_data: np.ndarray,
        neuron_areas: np.ndarray,
        method: str = "per_variable",
        n_samples: int = 2000,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute per-area selectivity profiles (analog of RRR Fig. 3).

        For each brain area a and variable v:
            α_{a,v} = mean over neurons n in area a of A_{v,n}

        Then z-score across areas.

        Parameters
        ----------
        neural_data : (n_samples, n_neurons)
        neuron_areas : (n_neurons,) brain area label per neuron
        method : "per_variable" or "joint"
        n_samples : int, samples for Jacobian computation

        Returns
        -------
        sel_profiles : (n_areas, n_variables) z-scored selectivity
        raw_profiles : (n_areas, n_variables) un-normalized selectivity
        area_list : list of area names
        """
        # Compute per-neuron attributions
        attr = self.compute_attribution_maps(
            neural_data, method=method, n_samples=n_samples
        )

        # Stack into (n_neurons, n_variables) matrix
        n_neurons = neural_data.shape[1]
        attr_matrix = np.zeros((n_neurons, N_VARIABLES))
        for v, var_name in enumerate(VARIABLE_NAMES):
            if var_name in attr:
                attr_matrix[:, v] = attr[var_name]

        # Average per area
        unique_areas = np.unique(neuron_areas)
        area_list = []
        raw_profiles = []

        for area in unique_areas:
            mask = neuron_areas == area
            if mask.sum() < 5:  # minimum neurons per area
                continue
            area_list.append(area)
            raw_profiles.append(attr_matrix[mask].mean(axis=0))

        raw_profiles = np.array(raw_profiles)  # (n_areas, n_variables)

        # Z-score per variable across areas
        sel_profiles = (raw_profiles - raw_profiles.mean(axis=0)) / (
            raw_profiles.std(axis=0) + 1e-8
        )

        return sel_profiles, raw_profiles, area_list

    def save(self, save_dir=None, prefix="xcebra"):
        """Save trained models to disk."""
        if save_dir is None:
            save_dir = MODELS_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save per-variable models
        for var_name, model in self.models_.items():
            path = save_dir / f"{prefix}_{var_name}.pt"
            model.save(str(path))

        # Save joint model
        if self.joint_model_ is not None:
            path = save_dir / f"{prefix}_joint.pt"
            self.joint_model_.save(str(path))

        # Save metadata
        meta = {
            "embedding_dim_per_group": self.embedding_dim_per_group,
            "n_groups": self.n_groups,
            "total_dim": self.total_dim,
            "model_architecture": self.model_architecture,
            "max_iterations": self.max_iterations,
            "variable_names": list(self.models_.keys()),
        }
        import json
        with open(save_dir / f"{prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, save_dir=None, prefix="xcebra"):
        """Load trained models from disk."""
        if save_dir is None:
            save_dir = MODELS_DIR
        save_dir = Path(save_dir)

        # Load per-variable models
        for var_name in VARIABLE_NAMES:
            path = save_dir / f"{prefix}_{var_name}.pt"
            if path.exists():
                model = CEBRA.load(str(path))
                self.models_[var_name] = model

        # Load joint model
        joint_path = save_dir / f"{prefix}_joint.pt"
        if joint_path.exists():
            self.joint_model_ = CEBRA.load(str(joint_path))

        self.is_fitted_ = bool(self.models_) or self.joint_model_ is not None
