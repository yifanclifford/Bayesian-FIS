import numpy as np
import torch
from torch import nn


def create_mask(M):
    M = M.type(torch.int)
    num_row = M.shape[0] - 1
    num_column = int(torch.sum(M[:num_row]).item())
    nonzeros = torch.sum(M, 1).type(torch.int)
    M = torch.zeros(num_row, num_column, dtype=torch.bool)
    N = 0
    for n in range(num_row):
        N += nonzeros[n]
        M[n, :N] = 1
    return M


def ind2hot(X, N):
    m, n = X.shape
    col = X.flatten().cpu().numpy()
    row = np.arange(m)
    row = row.repeat(n)
    H = torch.zeros([m, N], dtype=torch.uint8)
    H[row, col] = 1
    return H


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


def entropy(x, eps=1e-8):
    return torch.sum(torch.log(x + eps) * x + torch.log(-x + 1 + eps) * (-x + 1))


def cross_entropy(x, y, eps=1e-8):
    return -torch.sum(torch.log(x + eps) * y + torch.log(-x + 1 + eps) * (-y + 1))


class Embedding(nn.Module):
    def __init__(self, size, padding_idx, device='cpu', std_init=0.01):
        super(Embedding, self).__init__()
        self._embedding = nn.Parameter(torch.empty(size), requires_grad=True)
        nn.init.normal_(self._embedding, 0, std_init)
        size[padding_idx] = 1
        self.padding = torch.zeros(size).to(device)
        self.padding_idx = padding_idx
        self.std_init = std_init

    def initial(self):
        nn.init.normal_(self._embedding, 0, self.std_init)

    def embedding(self):
        return self._embedding

    def forward(self, x):
        V = torch.cat([self._embedding, self.padding], dim=self.padding_idx)
        if len(self._embedding.shape) == 3:
            return V[:, x]
        elif len(self._embedding.shape) <= 2:
            return V[x]
        else:
            return None

    def extra_repr(self):
        if len(self._embedding.shape) == 1:
            n = self._embedding.shape[0]
            return 'num_feature={}, dimension=1'.format(n)
        elif len(self._embedding.shape) == 2:
            n, d = self._embedding.shape
            return 'num_feature={}, dimension={}'.format(n, d)
        elif len(self._embedding.shape) == 3:
            m, n, d = self._embedding.shape
            return 'num_order={}, num_feature={}, dimension={}'.format(m, n, d)


if __name__ == '__main__':
    pro = torch.full((5, 10), 0.5)
