import torch
import torch.nn as nn
from typing import Optional, Tuple

class ParametricSafeEMLNode(nn.Module):
    """
    Computational node implementing the EML (Exponent-Minus-Log) operator.
    
    Computes: f(x, y) = exp(x) - ln(|y| + eps).
    Uses the Straight-Through Gumbel-Softmax estimator for differentiable 
    discrete selection of inputs from the feature pool.
    """

    def __init__(
        self,
        num_inputs: int,
        clamp_min: float = -12.0,
        clamp_max: float = 8.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = eps

        # Structural routing parameters (logits for selection)
        self.gate_x = nn.Parameter(torch.randn(num_inputs) * 0.02)
        self.gate_y = nn.Parameter(torch.randn(num_inputs) * 0.02)

        # Scaling coefficients for arguments
        self.scale_x = nn.Parameter(torch.tensor(1.0))
        self.scale_y = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        pool: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        fixed_idx: Optional[Tuple[int, int]] = None,
        pure_mode: bool = False,
    ) -> torch.Tensor:
        if fixed_idx is not None:
            idx_x, idx_y = fixed_idx
            x_selected = pool[:, idx_x]
            y_selected = pool[:, idx_y]
        else:
            w_x = torch.nn.functional.gumbel_softmax(self.gate_x, tau=temperature, hard=hard)
            w_y = torch.nn.functional.gumbel_softmax(self.gate_y, tau=temperature, hard=hard)
            x_selected = torch.matmul(pool, w_x)
            y_selected = torch.matmul(pool, w_y)

        x = x_selected * self.scale_x
        y = y_selected * self.scale_y

        # The exponent argument is always clamped, in both the continuous
        # relaxation phase and the fine-tuning ("pure_mode") phase. scale_x
        # is a free parameter with no explicit penalty once the structure is
        # frozen, so without this clamp it can drift to values that make
        # exp(x) overflow to inf and poison the loss with NaNs. The symbolic
        # formula extracted in to_symbolic()/to_standard_equation() is built
        # directly from scale_x/scale_y and is unaffected by this clamp; it
        # only protects the numerical forward pass used for loss/predict.
        x_safe = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return torch.exp(x_safe) - torch.log(torch.abs(y) + self.eps)