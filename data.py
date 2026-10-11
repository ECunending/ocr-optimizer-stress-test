"""Deterministic symbolic OCR data; opaque, independently permuted subject IDs."""
import torch


def make_data(n=20, subjects_per_group=4, data_seed=1729):
    m = subjects_per_group
    generator = torch.Generator().manual_seed(data_seed)
    ids = torch.randperm(n*m, generator=generator).reshape(n, m)
    train, test = [], []
    for i in range(n):
        for j in range(m):
            s = int(ids[i, j])
            train.append((s, n*m, i))
            (train if j == 0 else test).append((s, n*m+1, n+i))
    def tensors(rows):
        triples = torch.tensor(rows)
        x = torch.zeros(len(rows), n*m+2, dtype=torch.float64)
        x.scatter_(1, triples[:, :2], 1.)
        return x, triples[:, 2]
    x, y = tensors(train)
    xt, yt = tensors(test)
    return x, y, xt, yt
