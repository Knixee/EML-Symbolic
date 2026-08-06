import numpy as np
from typing import List, Optional, Tuple, Union
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .network import _EMLInternalNetwork

class EMLSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Differentiable Symbolic Regressor compliant with the scikit-learn API.
    
    Optimizes mathematical expressions via continuous relaxation of the graph structure (Phase 1)
    and refines numeric parameters after freezing the discrete structural routing (Phase 2).
    """

    def __init__(
        self,
        n_layers: int = 2,
        nodes_per_layer: int = 2,
        epochs: int = 15000,
        split_ratio: float = 0.80,
        lr_phase1: float = 0.01,
        lr_phase2: float = 0.002,
        l1_lambda: float = 0.15,
        gate_entropy_lambda: float = 0.0005,
        gate_l1_lambda: float = 0.01,
        pruning_threshold: float = 0.03,
        clamp_min: float = -12.0,
        clamp_max: float = 8.0,
        grad_clip: float = 0.5,
        eps: float = 1e-8,
        device: str = "auto",
        loss_fn: Union[str, nn.Module] = "mse",
        random_state: Optional[int] = None,
        verbose: Union[int, bool] = 1000,
    ):
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.lr_phase1 = lr_phase1
        self.lr_phase2 = lr_phase2
        self.l1_lambda = l1_lambda
        self.gate_entropy_lambda = gate_entropy_lambda
        self.gate_l1_lambda = gate_l1_lambda
        self.pruning_threshold = pruning_threshold
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.grad_clip = grad_clip
        self.eps = eps
        self.device = device
        self.loss_fn = loss_fn
        self.random_state = random_state
        self.verbose = verbose
    
    def _log_print(self, msg: str):
        print(msg, flush=True)

    def fit(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor],
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_nonfinite_streak: int = 20,
    ):
        """
        Parameters
        ----------
        max_nonfinite_streak : int, default=20
            Number of consecutive non-finite (NaN/Inf) loss values to
            tolerate before aborting with a RuntimeError. This is a runtime
            training-diagnostics knob (like `eval_set`), not a model
            hyperparameter, so it lives on fit() rather than __init__().
            Set higher to be more tolerant of transient instability, or
            lower to fail fast during experimentation.
        """
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # y_arr is already validated/converted by check_X_y; deriving ndim
        # from the raw `y` argument breaks whenever the caller passes a
        # plain list or pandas Series (neither has a .ndim attribute), which
        # scikit-learn's estimator API is expected to accept.
        self.y_ndim_ = y_arr.ndim

        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if self.device == "auto":
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device
        
        if isinstance(self.loss_fn, str):
            if self.loss_fn.lower() == "mse":
                criterion = nn.MSELoss()
            elif self.loss_fn.lower() == "mae":
                criterion = nn.L1Loss()
            elif self.loss_fn.lower() == "huber":
                criterion = nn.HuberLoss()
            else:
                raise ValueError(self.loss_fn)
        else:
            criterion = self.loss_fn
        
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device_)
        
        has_val = eval_set is not None
        if has_val:
            X_v, y_v = eval_set
            if self.device == "auto" and torch.cuda.is_available():
                warnings.warn(
                    "Validation metrics are computed on the CPU via scikit-learn. "
                    "Data transfer overhead may occur during logging steps.",
                    UserWarning
                )

        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device_)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).to(self.device_)
        
        if has_val:
            X_v_tensor = torch.tensor(X_v, dtype=torch.float32).to(self.device_)
            y_v_tensor = torch.tensor(y_v, dtype=torch.float32).to(self.device_)
            if y_v_tensor.ndim == 1:
                y_v_tensor = y_v_tensor.reshape(-1, 1)

        in_dim = X_arr.shape[1]

        self.model_ = _EMLInternalNetwork(
            in_dim=in_dim,
            n_layers=self.n_layers,
            nodes_per_layer=self.nodes_per_layer,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            eps=self.eps,
        ).to(self.device_)

        split_epoch = int(self.epochs * self.split_ratio)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr_phase1)

        start_temp = 2.0
        end_temp = 0.2
        fine_tune_mode = False
        self.fixed_structure_ = None
        self.pruning_mask_ = None
        
        start_time = time.time()
        moving_avg_step_time = 0.0 
        alpha = 0.05
        nonfinite_streak = 0
        
        if isinstance(self.verbose, bool):
            log_frequency = 1000 if self.verbose else None
        else:
            log_frequency = self.verbose if self.verbose > 0 else None

        if log_frequency:
            self._log_print(f"Training on device={self.device_}, epochs={self.epochs}")
            header = f"{'epoch':>8} | {'phase':<22} | {'loss':>14}"
            if has_val:
                header += f" | {'val_loss':>14}"
            header += f" | {'eta':>10}"
            self._log_print(header)

        for epoch in range(self.epochs):
            step_start = time.time()

            if epoch == split_epoch and not fine_tune_mode:
                fine_tune_mode = True
                self.fixed_structure_ = []

                for layer_nodes in self.model_.layers:
                    layer_idxs = []
                    for node in layer_nodes:
                        idx_x = torch.argmax(node.gate_x).item()
                        idx_y = torch.argmax(node.gate_y).item()
                        layer_idxs.append((idx_x, idx_y))
                    self.fixed_structure_.append(layer_idxs)

                with torch.no_grad():
                    self.pruning_mask_ = (
                        torch.abs(self.model_.final_projection.weight) > self.pruning_threshold
                    )
                    self.model_.final_projection.weight.data *= self.pruning_mask_

                # Column layout of the feature pool is: original input
                # features, the bias column, then one column per node in
                # arrival order. Nodes whose output column was pruned out of
                # final_projection contribute nothing to the prediction, so
                # freeze their scale_x/scale_y instead of wasting gradient
                # steps fine-tuning parameters that no longer matter.
                pruning_mask_flat = self.pruning_mask_[0]
                col = in_dim + 1
                for layer_nodes in self.model_.layers:
                    for node in layer_nodes:
                        node_is_active = bool(pruning_mask_flat[col])
                        node.scale_x.requires_grad_(node_is_active)
                        node.scale_y.requires_grad_(node_is_active)
                        col += 1

                ft_params = [
                    self.model_.final_projection.weight,
                    self.model_.final_projection.bias,
                ]
                for layer_nodes in self.model_.layers:
                    for node in layer_nodes:
                        if node.scale_x.requires_grad:
                            ft_params.append(node.scale_x)
                        if node.scale_y.requires_grad:
                            ft_params.append(node.scale_y)

                optimizer = optim.Adam(ft_params, lr=self.lr_phase2)

            optimizer.zero_grad()

            if not fine_tune_mode:
                temp = max(
                    end_temp,
                    start_temp * (end_temp / start_temp) ** (epoch / max(1, split_epoch)),
                )
            else:
                temp = end_temp

            outputs, _ = self.model_(X_tensor, temp, fine_tune_mode, self.fixed_structure_)
            loss = criterion(outputs, y_tensor)
            last_loss_val = loss.item()

            if not fine_tune_mode:
                l1_penalty = torch.sum(torch.abs(self.model_.final_projection.weight))

                gate_penalty = 0.0
                for layer_nodes in self.model_.layers:
                    for node in layer_nodes:
                        p_x = torch.softmax(node.gate_x / temp, dim=-1)
                        p_y = torch.softmax(node.gate_y / temp, dim=-1)
                        gate_penalty += -torch.sum(p_x * torch.log(p_x + self.eps))
                        gate_penalty += -torch.sum(p_y * torch.log(p_y + self.eps))
                        gate_penalty += self.gate_l1_lambda * (
                            torch.sum(torch.abs(node.gate_x)) + torch.sum(torch.abs(node.gate_y))
                        )

                total_loss = (
                    loss + self.l1_lambda * l1_penalty + self.gate_entropy_lambda * gate_penalty
                )
            else:
                total_loss = loss

            if not torch.isfinite(total_loss):
                nonfinite_streak += 1
                warnings.warn(
                    f"Non-finite loss encountered at epoch {epoch} (streak={nonfinite_streak}); "
                    "skipping optimizer step.",
                    UserWarning,
                    stacklevel=2,
                )
                optimizer.zero_grad()
                if nonfinite_streak >= max_nonfinite_streak:
                    raise RuntimeError(
                        f"Training diverged: loss was non-finite for "
                        f"{nonfinite_streak} consecutive steps (last at epoch "
                        f"{epoch}). Try lowering lr_phase1/lr_phase2, "
                        f"reducing clamp_max, or increasing grad_clip "
                        f"strictness."
                    )
                continue
            nonfinite_streak = 0

            total_loss.backward()

            if fine_tune_mode and self.pruning_mask_ is not None:
                with torch.no_grad():
                    self.model_.final_projection.weight.grad *= self.pruning_mask_

            nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=self.grad_clip)
            optimizer.step()
            
            step_duration = time.time() - step_start
            if epoch == 0:
                moving_avg_step_time = step_duration
            else:
                moving_avg_step_time = alpha * step_duration + (1 - alpha) * moving_avg_step_time

            if fine_tune_mode and self.pruning_mask_ is not None:
                with torch.no_grad():
                    self.model_.final_projection.weight.data *= self.pruning_mask_

            if log_frequency and (epoch % log_frequency == 0 or epoch == self.epochs - 1):
                remaining_steps = self.epochs - epoch - 1
                eta_seconds = remaining_steps * moving_avg_step_time
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                
                mode_str = "fine_tuning" if fine_tune_mode else "relaxation"

                row = f"{epoch:>8d} | {mode_str:<22} | {last_loss_val:>14.7f}"
                if has_val:
                    self.model_.eval()
                    with torch.no_grad():
                        v_outputs, _ = self.model_(X_v_tensor, temp, fine_tune_mode, self.fixed_structure_)
                        val_loss = criterion(v_outputs, y_v_tensor).item()
                    self.model_.train()
                    row += f" | {val_loss:>14.7f}"
                row += f" | {eta_str:>10}"

                self._log_print(row)

        if self.fixed_structure_ is None:
            self.fixed_structure_ = []
            for layer_nodes in self.model_.layers:
                layer_idxs = []
                for node in layer_nodes:
                    idx_x = torch.argmax(node.gate_x).item()
                    idx_y = torch.argmax(node.gate_y).item()
                    layer_idxs.append((idx_x, idx_y))
                self.fixed_structure_.append(layer_idxs)

        if log_frequency:
            elapsed = time.time() - start_time
            self._log_print(f"Training finished in {elapsed:.2f}s, final training loss={last_loss_val:.7f}")

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        check_is_fitted(self, attributes=["is_fitted_", "model_"])
        X_arr = check_array(X, accept_sparse=False)

        self.model_.eval()
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device_)

        with torch.no_grad():
            predictions, _ = self.model_(
                X_tensor,
                temperature=0.1,
                fine_tune_mode=True,
                fixed_structure=self.fixed_structure_,
            )

        preds = predictions.cpu().numpy()
        if hasattr(self, "y_ndim_") and self.y_ndim_ == 1:
            return preds.ravel()
        return preds

    def to_symbolic(self, feature_names: Optional[List[str]] = None) -> str:
        """Decodes the optimized discrete computational graph paths into an explicit equation string."""
        check_is_fitted(self, attributes=["is_fitted_", "model_"])

        in_features_dim = self.model_.layers[0][0].gate_x.shape[0] - 1
        names = (
            feature_names
            if feature_names and len(feature_names) == in_features_dim
            else [f"x{i}" for i in range(in_features_dim)]
        )
        pool_formulas = names + ["1.0"]

        for i, layer_nodes in enumerate(self.model_.layers):
            for j, node in enumerate(layer_nodes):
                if self.fixed_structure_:
                    idx_x, idx_y = self.fixed_structure_[i][j]
                else:
                    idx_x = torch.argmax(node.gate_x).item()
                    idx_y = torch.argmax(node.gate_y).item()

                arg_x = pool_formulas[idx_x]
                arg_y = pool_formulas[idx_y]

                sx = node.scale_x.item()
                sy = node.scale_y.item()

                str_x = f"{sx:.4f}" if arg_x == "1.0" else f"({sx:.4f} * {arg_x})"
                str_y = f"{sy:.4f}" if arg_y == "1.0" else f"({sy:.4f} * {arg_y})"

                formula = f"(exp({str_x}) - log(abs({str_y})))"
                pool_formulas.append(formula)

        w = self.model_.final_projection.weight.detach().cpu().numpy()[0]
        b = self.model_.final_projection.bias.detach().cpu().numpy()[0]

        terms = []
        for i, coef in enumerate(w):
            if abs(coef) > self.pruning_threshold:
                terms.append(f"({coef:.4f} * {pool_formulas[i]})")

        final_eq = " + ".join(terms) + f" + ({b:.4f})"
        return final_eq
    
    def to_standard_equation(
        self,
        feature_names: Optional[List[str]] = None,
        max_sympify_length: int = 20_000,
    ) -> str:
        """Parses and simplifies the nested expression graph using SymPy.

        Parameters
        ----------
        max_sympify_length : int, default=20_000
            Expressions longer than this (in characters) are unlikely to
            simplify in reasonable time, since each layer nests the full
            formulas of all previous layers. Above this length, sympy is
            skipped entirely and the raw, unsimplified formula is returned
            instead of risking a hang. This is a call-time safety knob for
            this specific method, so it lives here rather than on
            __init__() or fit().
        """
        import sympy as sp
        check_is_fitted(self, attributes=["is_fitted_", "model_"])

        raw_eml_string = self.to_symbolic(feature_names=feature_names)

        if len(raw_eml_string) > max_sympify_length:
            warnings.warn(
                "Extracted expression is too large to simplify safely "
                f"({len(raw_eml_string)} characters); returning the raw "
                "unsimplified formula. Consider reducing n_layers/"
                "nodes_per_layer, raising pruning_threshold, or passing a "
                "larger max_sympify_length.",
                UserWarning,
            )
            return raw_eml_string

        try:
            parsed_expr = sp.sympify(raw_eml_string)
            simplified_expr = sp.simplify(parsed_expr)
            simplified_expr = sp.trigsimp(simplified_expr)
            return str(simplified_expr)
        except Exception as e:
            warnings.warn(
                f"SymPy simplification failed ({type(e).__name__}: {e}); returning raw expression.",
                UserWarning,
                stacklevel=2,
            )
            return raw_eml_string