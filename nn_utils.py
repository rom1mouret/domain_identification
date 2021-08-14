from typing import List
import torch
import torch.nn as nn


class MaskedConv1d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int=-1, mask: List[int]=[]) -> None:
        super(MaskedConv1d, self).__init__()
        if mask == []:
            assert kernel_size > 1, "kernel size must be provided when mask is not"
            mask = [1] * kernel_size
            mask[in_dim // 2 + 1] = 0
        else:
            kernel_size = len(mask)

        assert kernel_size % 2 == 1, "kernel size must be an odd number"

        layer = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size)
        self._kernel = nn.Parameter(layer.weight.data)
        self._bias = nn.Parameter(layer.bias.data)
        self._mask = torch.ones_like(self._kernel)
        for i, b in enumerate(mask):
            self._mask[:, :, i] = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._kernel * self._mask
        return nn.functional.conv1d(x, kernel, bias=self._bias)

    def _apply(self, fn):
        # so that tensor.device(device) applies device() to the mask
        super(MaskedConv1d, self)._apply(fn)
        self._mask = fn(self._mask)
        return self


class Permute(nn.Module):
    def __init__(self, *dims) -> None:
        super(Permute, self).__init__()
        self._dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._dims)


class Triangle(torch.nn.Module):
    def forward(self, x):
        return x.abs()


def masked_mean(x: torch.Tensor, sentences: list) -> torch.Tensor:
    mask = torch.zeros_like(x)
    shift = max(map(len, sentences)) - x.size(1)
    for i, sentence in enumerate(sentences):
        mask[i, :len(sentence)-shift] = 1

    return (x * mask).sum() / mask.sum()


def freeze(net: nn.Module):
    net.eval()  # freezes batchnorms and dropouts
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net: nn.Module):
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
