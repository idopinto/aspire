import torch
from torch.nn.modules.loss import _Loss
from typing import Optional, Callable
from torch import Tensor
from torch.nn import  PairwiseDistance

'''
The dimension checking for the anchor, positive, and negative tensors in the TripletMarginWithDistanceLoss 
function was introduced in PyTorch version 1.7.0. 
The original requirement 1.8.1
'''


class CustomTripletMarginWithDistanceLoss(_Loss):
    __constants__ = ["margin", "swap", "reduction"]
    margin: float
    swap: bool

    def __init__(
            self,
            *,
            distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
            margin: float = 1.0,
            swap: bool = False,
            reduction: str = "mean",
    ):
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        if margin <= 0:
            raise ValueError(
                f"TripletMarginWithDistanceLoss: expected margin to be greater than 0, got {margin} instead"
            )
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = (
            distance_function if distance_function is not None else PairwiseDistance()
        )
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return custom_triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )


def custom_triplet_margin_with_distance_loss(anchor: torch.Tensor,
                                             positive: torch.Tensor,
                                             negative: torch.Tensor,
                                             *,
                                             distance_function: Optional[
                                                 Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                                             margin: float = 1.0,
                                             swap: bool = False,
                                             reduction: str = "mean",
                                             ) -> torch.Tensor:
    r"""Compute the triplet margin loss for input tensors using a custom distance function.

    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    # if has_torch_function_variadic(anchor, positive, negative):
    #     return handle_torch_function(
    #         triplet_margin_with_distance_loss,
    #         (anchor, positive, negative),
    #         anchor,
    #         positive,
    #         negative,
    #         distance_function=distance_function,
    #         margin=margin,
    #         swap=swap,
    #         reduction=reduction,
    # #     )

    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # Check validity of margin
    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")

    # # Check dimensions
    # a_dim = anchor.ndim
    # p_dim = positive.ndim
    # n_dim = negative.ndim
    # if not (a_dim == p_dim and p_dim == n_dim):
    #     raise RuntimeError(
    #         f"The anchor, positive, and negative tensors are expected to have "
    #         f"the same number of dimensions, but got: anchor {a_dim}D, "
    #         f"positive {p_dim}D, and negative {n_dim}D inputs"
    #     )

    # Calculate loss
    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)

    # Apply reduction
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss

