import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .nodes import ParametricSafeEMLNode

class _EMLInternalNetwork(nn.Module):
    """
    Multi-layer neural network architecture with nested EML nodes 
    and a sparse linear projection output layer.
    """

    def __init__(
        self,
        in_dim: int,
        n_layers: int,
        nodes_per_layer: int,
        clamp_min: float,
        clamp_max: float,
        eps: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer

        self.layers = nn.ModuleList()
        current_pool_size = in_dim + 1  # Features + bias intercept (constant 1.0)

        for _ in range(n_layers):
            node_list = nn.ModuleList(
                [
                    ParametricSafeEMLNode(
                        num_inputs=current_pool_size,
                        clamp_min=clamp_min,
                        clamp_max=clamp_max,
                        eps=eps,
                    )
                    for _ in range(nodes_per_layer)
                ]
            )
            self.layers.append(node_list)
            current_pool_size += nodes_per_layer

        self.final_projection = nn.Linear(current_pool_size, 1)
        nn.init.normal_(self.final_projection.weight, std=0.1)

    def forward(
        self,
        X_tensor: torch.Tensor,
        temperature: float,
        fine_tune_mode: bool,
        fixed_structure: Optional[List[List[Tuple[int, int]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X_tensor.shape[0]
        ones = torch.ones(batch_size, 1, device=X_tensor.device)
        pool = torch.cat([X_tensor, ones], dim=1)

        for i, layer_nodes in enumerate(self.layers):
            layer_outputs = []
            for j, node in enumerate(layer_nodes):
                f_idx = fixed_structure[i][j] if fine_tune_mode and fixed_structure else None
                out = node(
                    pool,
                    temperature=temperature,
                    hard=True,
                    fixed_idx=f_idx,
                    pure_mode=fine_tune_mode,
                )
                layer_outputs.append(out.unsqueeze(1))
            pool = torch.cat([pool] + layer_outputs, dim=1)

        return self.final_projection(pool), pool