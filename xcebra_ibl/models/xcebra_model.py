"""Regularized per-variable CEBRA adaptation with inverted-gradient attribution.

Independent supervised encoders do not implement canonical multiobjective
xCEBRA. Attribution measures model sensitivity, not causal effects or RRR
encoding coefficients. Jacobian norm regularization alone does not guarantee
sparsity or identifiability on IBL recordings.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    JACOBIAN_REG_WEIGHT, JACOBIAN_N_PROJ, JACOBIAN_PINV_RCOND, MODELS_DIR,
    RANDOM_SEED,
)


def _install_trial_safe_expander(dataset, trial_ids, trial_length):
    """Make a CEBRA dataset clamp offset windows inside their trial."""
    import types

    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    starts = {}
    ends = {}
    for position, trial in enumerate(trial_ids):
        trial = int(trial)
        starts.setdefault(trial, position)
        ends[trial] = position + 1
    offset = dataset.offset

    def expand_index(self, index):
        index_tensor = torch.as_tensor(index, dtype=torch.long)
        centers = index_tensor.detach().cpu().numpy().reshape(-1)
        expanded = []
        for center in centers:
            center = int(center)
            trial = int(trial_ids[center])
            lower = starts[trial] + int(offset.left)
            upper = ends[trial] - int(offset.right)
            clipped = min(max(center, lower), upper)
            expanded.append(
                [clipped + delta for delta in range(-int(offset.left), int(offset.right))]
            )
        return torch.as_tensor(expanded, dtype=torch.long, device=index_tensor.device)

    dataset.expand_index = types.MethodType(expand_index, dataset)


class XCEBRAModel:
    """
    xCEBRA wrapper for IBL neural data analysis.

    Trains independent supervised encoders and computes mean absolute
    pseudoinverse-Jacobian attribution. These scores are not RRR coefficients
    and do not isolate a variable's unique effect when labels are correlated.

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
        jacobian_reg_weight: Optional[float] = JACOBIAN_REG_WEIGHT,
        jacobian_n_proj: int = JACOBIAN_N_PROJ,
        jacobian_pinv_rcond: float = JACOBIAN_PINV_RCOND,
        random_seed: int = RANDOM_SEED,
        use_xcebra: bool = True,
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
        self.jacobian_reg_weight = jacobian_reg_weight
        self.jacobian_n_proj = jacobian_n_proj
        self.jacobian_pinv_rcond = jacobian_pinv_rcond
        self.random_seed = random_seed
        self.use_xcebra = use_xcebra

        # Training and inference must use the same trial boundaries.  Without
        # this, an offset convolution sees the last bins of one trial next to
        # the first bins of the following trial.
        self.trial_ids_ = None
        self.time_ids_ = None
        self.trial_length_ = None

        # Will be set during fit
        self.models_ = {}           # one CEBRA model per variable
        self.joint_model_ = None    # single CEBRA model with all labels
        self.is_fitted_ = False
        self.training_losses_ = {}
        self.label_classes_ = {}

    def _set_trial_structure(self, trial_ids=None, time_ids=None, trial_length=None):
        """Store trial metadata used to keep temporal windows within trials."""
        if trial_ids is None or time_ids is None or trial_length is None:
            self.trial_ids_ = self.time_ids_ = self.trial_length_ = None
            return
        trial_ids = np.asarray(trial_ids)
        time_ids = np.asarray(time_ids)
        if trial_ids.ndim != 1 or time_ids.ndim != 1 or len(trial_ids) != len(time_ids):
            raise ValueError("trial_ids and time_ids must be matching 1D arrays")
        self.trial_ids_ = trial_ids.astype(np.int64, copy=False)
        self.time_ids_ = time_ids.astype(np.int64, copy=False)
        self.trial_length_ = int(trial_length)

    def _fit_cebra(self, model, neural_data, y, callback, fit_kwargs):
        """Fit ordinary CEBRA or the official CEBRA 0.6 regularized solver."""
        if not self.use_xcebra or self.jacobian_reg_weight is None:
            model.fit(neural_data, y, **fit_kwargs)
            return

        # The regularized solver is part of the official xCEBRA API in CEBRA
        # 0.6.  Build the estimator's dataset/loader once, then replace only
        # its solver with RegularizedSolver so save/transform remain standard
        # CEBRA operations.
        try:
            import cebra
            state = model._prepare_fit(neural_data, y)
            base_solver, encoder, loader, is_multisession = state
            if is_multisession:
                raise ValueError("Regularized xCEBRA requires single-session input")
            if self.trial_ids_ is not None:
                _install_trial_safe_expander(loader.dataset, self.trial_ids_, self.trial_length_)
            solver = cebra.solver.init(
                "regularized-solver",
                model=encoder,
                criterion=base_solver.criterion,
                optimizer=base_solver.optimizer,
                tqdm_on=model.verbose,
                lambda_JR=self.jacobian_reg_weight,
            )
            solver.to(model.device_)
            model._partial_fit(
                solver,
                encoder,
                loader,
                is_multisession,
                callback=callback,
                callback_frequency=(self.checkpoint_frequency if callback else None),
            )
        except (AttributeError, KeyError, ImportError) as exc:
            raise RuntimeError(
                "Official regularized xCEBRA requires cebra==0.6.0 or newer; "
                "ordinary CEBRA would not match this project's stated method."
            ) from exc

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
        trial_ids: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        trial_length: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Strategy A: Train a separate CEBRA model per behavioral variable.

        Each independent model is conditioned on one label. This adaptation
        does not assign identifiable variable slices within a shared encoder.

        Parameters
        ----------
        neural_data : (n_samples, n_neurons) array
            Flattened neural activity (K*T rows, N columns).
        labels : dict
            {variable_name: (n_samples,) or (n_samples, 1) label array}
        verbose : bool
        """
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        self.models_ = {}
        self.training_losses_ = {}
        self._set_trial_structure(trial_ids, time_ids, trial_length)

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

            # Dtype is the CEBRA API contract for label semantics.  The
            # preprocessing stage intentionally preserves categorical labels
            # as integer arrays; cardinality-based detection misclassifies
            # z-scored categorical labels and binary continuous signals.
            is_discrete = np.issubdtype(y.dtype, np.integer)

            callback = self._checkpoint_callback(var_name)
            fit_kwargs = {}
            if callback is not None:
                fit_kwargs = {
                    "callback": callback,
                    "callback_frequency": self.checkpoint_frequency,
                }

            if is_discrete:
                # Discrete labels must be one-dimensional integer arrays.
                y = y.astype(np.int64, copy=False).reshape(-1)
                # CEBRA's discrete distribution uses np.bincount internally,
                # so labels must be non-negative and densely indexed.  This
                # also makes the public wrapper safe for callers that provide
                # raw categorical codes such as {-1, 0, 1}.
                classes = np.unique(y)
                y = np.searchsorted(classes, y).astype(np.int64, copy=False)
                self.label_classes_[var_name] = classes.tolist()
            else:
                # Continuous labels must be two-dimensional arrays.
                y = y.astype(np.float32, copy=False).reshape(-1, 1)

            self._fit_cebra(model, neural_data, y, callback, fit_kwargs)

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
        trial_ids: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        trial_length: Optional[int] = None,
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
        self._set_trial_structure(trial_ids, time_ids, trial_length)
        if self.use_xcebra and self.jacobian_reg_weight is not None:
            raise ValueError(
                "Regularized xCEBRA is implemented for the per-variable "
                "single-objective models. Use method='per_variable' for the "
                "scientific xCEBRA analysis, or set use_xcebra=False."
            )

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

            if np.issubdtype(y.dtype, np.integer):
                y = y.astype(np.int64, copy=False).reshape(-1)
                classes = np.unique(y)
                discrete_labels.append(
                    np.searchsorted(classes, y).astype(np.int64, copy=False)
                )
            else:
                continuous_labels.append(y.reshape(-1, 1))

        # Build y arguments for CEBRA.fit
        # CEBRA expects: continuous labels as 2D arrays, one discrete array
        y_args = []
        for cl in continuous_labels:
            y_args.append(cl)
        # Combine discrete labels into a single compound label
        if discrete_labels:
            # Encode each observed combination densely.  Fixed-base arithmetic
            # can collide when a variable has more than 100 categories.
            discrete_matrix = np.column_stack(discrete_labels)
            _, compound = np.unique(discrete_matrix, axis=0, return_inverse=True)
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

    def _transform_trial_safe(self, model, neural_data):
        """Transform each trial separately when temporal context is used."""
        if self.trial_ids_ is None:
            return model.transform(neural_data)
        outputs = []
        for trial in np.unique(self.trial_ids_):
            idx = np.flatnonzero(self.trial_ids_ == trial)
            outputs.append(model.transform(neural_data[idx]))
        return np.concatenate(outputs, axis=0)

    def transform_per_variable(
        self, neural_data: np.ndarray, trial_ids=None, time_ids=None, trial_length=None
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings from per-variable models.

        Returns
        -------
        dict : {var_name: (n_samples, embedding_dim_per_group)}
        """
        self._set_trial_structure(trial_ids, time_ids, trial_length)
        embeddings = {}
        for var_name, model in self.models_.items():
            embeddings[var_name] = self._transform_trial_safe(model, neural_data)
        return embeddings

    def transform_joint(
        self, neural_data: np.ndarray, trial_ids=None, time_ids=None, trial_length=None
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings from joint model, split into per-variable groups.

        Returns
        -------
        dict : {var_name: (n_samples, embedding_dim_per_group)}
        """
        if self.joint_model_ is None:
            raise RuntimeError("Joint model not fitted. Call fit_joint() first.")

        self._set_trial_structure(trial_ids, time_ids, trial_length)
        full_embedding = self._transform_trial_safe(self.joint_model_, neural_data)

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
        trial_ids=None,
        time_ids=None,
        trial_length=None,
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
        self._set_trial_structure(trial_ids, time_ids, trial_length)
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
        attribution_maps = {}

        data_subset = self._attribution_windows(neural_data, n_samples)

        for var_name, model in self.models_.items():
            print(f"  Computing Jacobian attribution for: {var_name}")
            attr = self._jacobian_attribution(model, data_subset, batch_size)
            attribution_maps[var_name] = attr  # (n_neurons,)

        return attribution_maps

    def _compute_attributions_joint(
        self, neural_data, n_samples, batch_size
    ) -> Dict[str, np.ndarray]:
        """Compute Jacobian-based attributions from joint model, per group."""
        attribution_maps = {}

        data_subset = self._attribution_windows(neural_data, n_samples)

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

    def _attribution_windows(self, neural_data, n_samples):
        """Select valid centers and build true temporal input windows.

        The previous implementation sampled rows and transposed them into a
        pseudo-sequence, so neighboring rows were unrelated trials.  This
        helper creates the same receptive-field windows used by CEBRA and
        never crosses a trial boundary.
        """
        neural_data = np.asarray(neural_data, dtype=np.float32)
        if self.models_:
            fitted_model = next(iter(self.models_.values()))
        else:
            fitted_model = self.joint_model_
        net = fitted_model.solver_.model
        if hasattr(net, "get_offset"):
            offset = net.get_offset()
            left, right = int(offset.left), int(offset.right)
        else:
            left, right = 0, 1

        if self.trial_ids_ is None:
            # Without explicit trial metadata, treat the supplied array as
            # one continuous sequence. This preserves the temporal input
            # shape, while making the boundary assumption explicit.
            valid_centers = np.arange(left, len(neural_data) - right + 1)
        elif left == 0 and right == 1:
            valid_centers = np.arange(len(neural_data))
        else:
            valid_centers = np.flatnonzero(
                (self.time_ids_ >= left)
                & (self.time_ids_ < self.trial_length_ - right + 1)
            )

        if valid_centers.size == 0:
            raise ValueError("No valid attribution centers remain inside trial boundaries")
        count = min(int(n_samples), valid_centers.size)
        rng = np.random.default_rng(self.random_seed)
        centers = (
            valid_centers
            if count == valid_centers.size
            else rng.choice(valid_centers, size=count, replace=False)
        )
        if left == 0 and right == 1:
            return neural_data[centers]
        windows = np.stack(
            [neural_data[c - left : c + right].T for c in centers], axis=0
        )
        return windows

    def _jacobian_attribution(
        self,
        cebra_model: CEBRA,
        data: np.ndarray,
        batch_size: int = 256,
        output_slice: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute the Inverted Neuron Gradient (mean absolute Jacobian
        pseudo-inverse) per input neuron.

        For each sample x, computes the encoder Jacobian J = ∂f(x)/∂x,
        averages over the temporal receptive field, computes the Moore-Penrose
        pseudo-inverse J⁺, and averages |J⁺| across output dimensions and
        samples. This is the attribution defined by xCEBRA; it is distinct
        from a squared forward gradient.

        Parameters
        ----------
        cebra_model : fitted CEBRA model
        data : (n_samples, n_features)
        batch_size : int
        output_slice : (start, end) to select output dimensions (for joint model)

        Returns
        -------
        attributions : (n_features,) mean absolute inverted-gradient score
        """
        # Access the underlying PyTorch model
        solver = cebra_model.solver_
        net = solver.model

        # Determine device
        device = next(net.parameters()).device

        net.eval()
        if data.ndim == 2:
            data = np.asarray(data, dtype=np.float32)
        elif data.ndim == 3:
            data = np.asarray(data, dtype=np.float32)
        else:
            raise ValueError(f"Expected 2D samples or 3D temporal windows, got {data.shape}")
        n_samples = data.shape[0]
        n_features = data.shape[1]
        accumulated_inverse = np.zeros(n_features, dtype=np.float64)
        n_accumulated = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_np = data[start_idx:end_idx]
            if batch_np.ndim == 3:
                batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            elif "offset" in self.model_architecture:
                # This path is retained for callers that provide 2D data
                # directly. New session analysis uses explicit windows above.
                batch = torch.tensor(
                    batch_np.T[None, ...], dtype=torch.float32, device=device
                )
            else:
                batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            batch.requires_grad_(True)

            embedding = net(batch)
            if embedding.ndim == 3:
                # A receptive-field window should produce one center output;
                # averaging is a safe compatibility path for older CEBRA
                # versions that return a singleton temporal dimension.
                embedding = embedding.mean(dim=-1)

            if output_slice is not None:
                embedding = embedding[:, output_slice[0] : output_slice[1]]

            # Compute one exact output basis per batch.  The exact Jacobian is
            # required before taking its pseudo-inverse; a Hutchinson sketch
            # cannot recover the inverted neuron gradient.
            D_out = embedding.shape[1]
            if self.jacobian_n_proj != -1:
                raise ValueError(
                    "Inverted Neuron Gradient requires jacobian_n_proj=-1 "
                    "so the full encoder Jacobian is available."
                )
            projections = torch.eye(D_out, device=device).unsqueeze(1).expand(
                D_out, embedding.shape[0], D_out
            )
            try:
                grads = torch.autograd.grad(
                    embedding,
                    batch,
                    grad_outputs=projections,
                    is_grads_batched=True,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            except TypeError:
                grads = torch.stack(
                    [
                        torch.autograd.grad(
                            embedding,
                            batch,
                            grad_outputs=projection,
                            retain_graph=i < len(projections) - 1,
                            create_graph=False,
                            allow_unused=False,
                        )[0]
                        for i, projection in enumerate(projections)
                    ],
                    dim=0,
                )
            # grads has shape (output_dim, batch, input_dim[, offset]).
            # Match the reference implementation by reducing a temporal
            # receptive field before inversion, not after inversion.
            if grads.ndim == 4:
                grads = grads.mean(dim=-1)
            jacobian = grads.permute(1, 0, 2).detach().cpu().numpy()
            jacobian = np.asarray(jacobian, dtype=np.float64)
            inverted = np.linalg.pinv(jacobian, rcond=self.jacobian_pinv_rcond)
            if not np.isfinite(inverted).all():
                raise FloatingPointError("Non-finite inverted Jacobian")
            accumulated_inverse += np.abs(inverted).mean(axis=2).sum(axis=0)
            n_accumulated += len(inverted)

        # Weight samples equally, including an incomplete final batch.
        attributions = accumulated_inverse / max(n_accumulated, 1)
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
            "method": "regularized_cebra_adaptation",
            "embedding_dim_per_group": self.embedding_dim_per_group,
            "n_groups": self.n_groups,
            "total_dim": self.total_dim,
            "model_architecture": self.model_architecture,
            "max_iterations": self.max_iterations,
            "variable_names": list(self.models_.keys()),
            "use_xcebra": self.use_xcebra,
            "jacobian_reg_weight": self.jacobian_reg_weight,
            "jacobian_n_proj": self.jacobian_n_proj,
            "jacobian_pinv_rcond": self.jacobian_pinv_rcond,
            "random_seed": self.random_seed,
            "label_classes": self.label_classes_,
            "trial_safe_temporal_context": self.trial_ids_ is not None,
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
                # These files are generated by this run and are therefore a
                # trusted checkpoint source.  PyTorch >=2.6 defaults to a
                # weights-only unpickler, which cannot read CEBRA 0.6's
                # sklearn metadata (notably NumPy dtype objects).
                model = CEBRA.load(str(path), weights_only=False)
                self.models_[var_name] = model

        # Load joint model
        joint_path = save_dir / f"{prefix}_joint.pt"
        if joint_path.exists():
            self.joint_model_ = CEBRA.load(str(joint_path), weights_only=False)

        self.is_fitted_ = bool(self.models_) or self.joint_model_ is not None
