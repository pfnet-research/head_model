from typing import Any, Tuple

import torch
import torch.nn as nn

# https://cyberagent.ai/blog/research/11863/


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input_forward: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return input_forward

    @staticmethod
    def backward(
        ctx: Any, grad_backward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (scale,) = ctx.saved_tensors
        return scale * -grad_backward, None


class GradientReversal(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)
