"""Function-matched direct and full-expressivity balanced SVD parameterizations."""
import torch
from torch import nn


def initial_matrix(seed, outputs=40, inputs=82, scale=0.01):
    return scale * torch.randn(outputs, inputs, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


class Model(nn.Module):
    def __init__(self, kind, w0):
        super().__init__()
        self.kind = kind
        if kind == 'direct':
            self.W = nn.Parameter(w0.clone())
        elif kind == 'factorized':
            u, s, vh = torch.linalg.svd(w0, full_matrices=False)
            self.A = nn.Parameter(u * s.sqrt())
            self.B = nn.Parameter(vh.T * s.sqrt())
        else:
            raise ValueError(kind)

    def matrix(self):
        return self.W if self.kind == 'direct' else self.A @ self.B.T

    def forward(self, x):
        return x @ self.matrix().T
