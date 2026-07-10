import torch.nn as nn
import torch
from torch.nn import HuberLoss
from typing import Tuple


def class_balanced_weights(samples_per_class, beta: float = 0.9999, eps: float = 1e-12):
    """
    Class-balanced weights with "effective number of samples" (Cui et al., 2019):
        w_c = (1 - beta) / (1 - beta^{n_c})
    Then normalize so that mean(w) = 1 over present classes.
    """
    if samples_per_class is None:
        raise ValueError("samples_per_class is required")
    if isinstance(samples_per_class, torch.Tensor):
        n = samples_per_class.detach().cpu().double()
    else:
        n = torch.tensor(list(samples_per_class), dtype=torch.double)
    if n.numel() == 0:
        raise ValueError("samples_per_class is empty")
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0,1), got {beta}")

    # Missing classes (n=0) get weight 0.
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.double), n.clamp_min(0))
    w = (1.0 - beta) / (effective_num + eps)
    w = torch.where(n > 0, w, torch.zeros_like(w))

    present = (n > 0)
    denom = (w[present].mean().clamp_min(eps) if bool(present.any()) else w.mean().clamp_min(eps))
    w = w / denom
    return w.to(dtype=torch.float32)


class ClassBalancedCELoss(nn.Module):
    """
    Class-balanced CE for one-hot targets (expects `preds` are probabilities, i.e., already softmaxed).
    """

    _epsilon = 1e-6

    def __init__(self, samples_per_class, beta: float = 0.9999) -> None:
        super().__init__()
        weight = class_balanced_weights(samples_per_class=samples_per_class, beta=beta)
        print(f"[{self._get_name()}] CB beta={beta}, samples_per_class={list(map(int, list(samples_per_class)))}")
        print(f"[{self._get_name()}] Loss Weights:", weight.detach().cpu().tolist())
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss



class CELoss(nn.Module):
    """
    Cross Entropy loss
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BCELoss(nn.Module):
    """
    Binary cross entropy loss for phase-picking and detection
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, weight=None, has_softmax=True):
        """
        Args:
            gamma (float): Coefficient.
            weight (list|Tensor): Weight of each class. Defaults to None.
            has_softmax (bool): If True, softmax will be applied for the input `preds`. Defaults to True.
        """
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

        self.has_softmax = has_softmax

    def forward(self, preds, targets):
        """Input shape: (N,C,L) or (N,Classes)"""
        if self.has_softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= torch.pow((1 - preds), self.gamma)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal loss (binary)

    note: the input `preds` must be the output of `sigmoid`.
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, alpha=1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""

        loss = -(
            self.alpha
            * torch.pow((1 - preds), self.gamma)
            * targets
            * torch.log(preds + self._epsilon)
            + (1 - self.alpha)
            * torch.pow(preds, self.gamma)
            * (1 - targets)
            * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss


class MSELoss(nn.Module):
    """
    MSE Loss.
    """

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        """Input shape: (N,C,L)"""
        loss = (preds - targets) ** 2
        loss *= self.weight
        loss = loss.mean()
        return loss


class ConbinationLoss(nn.Module):
    """
    For multi-task learning.
    """

    def __init__(self, losses: list, losses_weights: list = None) -> None:
        """
        note: Use `functools.partial` if there are arguments that need to be passed to the loss module in `losses`.
        """
        super().__init__()

        assert len(losses) > 0

        if len(losses) == 1:
            raise Exception(
                f"Expected number of losses `>=2`, got {len(losses)}."
                f" `ConbinationLoss` is used for multi-task training, and requires at least two loss modules."
                f" Use `{losses[0]}` instead."
            )

        if losses_weights is not None:
            assert len(losses) == len(losses_weights)
            self.losses_weights = losses_weights
        else:
            self.losses_weights = [1.0] * len(losses)

        self.losses = nn.ModuleList([Loss() for Loss in losses])

    def forward(self, preds: Tuple[torch.Tensor], targets: Tuple[torch.Tensor]):
        sum_loss = 0.0
        for i, (pred, target, lossfn, weight) in enumerate(
            zip(preds, targets, self.losses, self.losses_weights)
        ):
            sum_loss += lossfn(pred, target) * weight

        return sum_loss


class MousaviLoss(nn.Module):
    """
    Loss module for the following models:
    
        [1] MagNet. Mousavi et al. 2019
        [2] dist-PT Network. Mousavi et al. 2020
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        y_hat = preds[:, 0].reshape(-1, 1)
        s = preds[:, 1].reshape(-1, 1)
        loss = torch.sum(
            0.5 * torch.exp(-1 * s) * torch.square(torch.abs(targets - y_hat)) + 0.5 * s
        )
        return loss
