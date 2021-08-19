import torch
from torch import nn
from model.utils import trace, entropy, Embedding
import abc


def reparameterize(mu, sigma):
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    return (eps * std) + mu
    # return mu


class FM(nn.Module):
    def __init__(self, M, **inputs):
        super(FM, self).__init__()
        self.w0 = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.normal_(self.w0, 0, 0.01)
        self.w = Embedding([inputs['n']], 0, inputs['device'])
        self.V = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])
        self.M = M
        self.device = inputs['device']
        self.alpha = inputs['alpha']
        self.beta = inputs['beta']

    def forward(self, **inputs):
        V = self.M @ self.V(inputs['x'])
        w = self.w(inputs['x']) @ self.M.t()
        sumV = torch.sum(V, dim=1)
        sqV = torch.sum(V * V, dim=1)
        predict = torch.sum(w, dim=1) + self.w0
        predict += self.inference(sumV * sumV - sqV) / 2
        return predict, self.regularization(w, V)

    def predict(self, **inputs):
        V = self.M @ self.V(inputs['x'])
        w = self.w(inputs['x']) @ self.M.t()
        sumV = torch.sum(V, dim=1)
        sqV = torch.sum(V * V, dim=1)
        predict = torch.sum(w, dim=1) + self.w0
        predict += self.evaluate(sumV * sumV - sqV) / 2
        return predict

    def inference(self, V):
        return torch.sum(V, -1)

    def evaluate(self, V):
        return self.inference(V)

    def regularization(self, w, V):
        return self.alpha * trace(w) + self.beta * trace(V)


class PFM(FM):
    def __init__(self, M, **inputs):
        super(PFM, self).__init__(M, **inputs)
        self.mu = nn.Linear(inputs['d'], 1)
        self.sigma = nn.Linear(inputs['d'], 1)

    def inference(self, V):
        mu = self.mu(V).squeeze(-1)
        sigma = self.sigma(V).squeeze(-1)
        return reparameterize(mu, sigma)

    def evaluate(self, V):
        return self.mu(V).squeeze(-1)


class SparseFM(FM):
    def __init__(self, M, **inputs):
        super(SparseFM, self).__init__(M, **inputs)

    def regularization(self, w, V):
        return self.alpha * torch.norm(w, p=1) + self.beta * torch.norm(V, p=1)


class NeurFM(FM):
    def __init__(self, M, **inputs):
        super(NeurFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        self.network.add_module('dropout', nn.Dropout(inputs['drop']))
        for l in range(inputs['num_layer']):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(inputs['d'], 1))

    def inference(self, V):
        return self.network(V).squeeze(-1) / 2


class PNFM(FM):
    def __init__(self, M, **inputs):
        super(PNFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        for l in range(inputs['order'] - 2):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.mu = nn.Linear(inputs['layer'][-1], 1)
        self.sigma = nn.Linear(inputs['layer'][-1], 1)

    def inference(self, V):
        output = self.network(V)
        mu = self.mu(output).squeeze(-1)
        sigma = self.sigma(output).squeeze(-1)
        return reparameterize(mu, sigma)

    def evaluate(self, V):
        output = self.network(V)
        return self.mu(output).squeeze(-1) / 2


class FIS(nn.Module):
    def __init__(self, M, **inputs):
        super(FIS, self).__init__()
        self.w0 = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.normal_(self.w0, 0, 0.01)
        self.w = Embedding([inputs['n']], 0, inputs['device'])
        self.V = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])
        self.probability = nn.Parameter(torch.full((inputs['m'], inputs['n']), inputs['rate']), requires_grad=True)
        self.num_field = M.shape[0]
        self.num_interaction = int(self.num_field * (self.num_field - 1) / 2)
        idx = torch.tensor([[i, j] for i in range(self.num_field) for j in range(i + 1, self.num_field)],
                           dtype=torch.int64).to(inputs['device'])
        self.row = idx[:, 0]
        self.col = idx[:, 1]
        self.M = M
        self.alpha = inputs['alpha']
        self.beta = inputs['beta']

    def initial(self):
        self.V.initial()
        self.w.initial()
        nn.init.normal_(self.w0, 0, 0.01)

    def clamp(self):
        self.probability.data[self.probability.data < 0] = 0
        self.probability.data[self.probability.data > 1] = 1

    def selection(self, pi, Pi):
        eps = torch.rand_like(pi)
        s = (pi >= eps).type(torch.float32)
        eps = torch.rand_like(Pi)
        S = s.unsqueeze(1) + s.unsqueeze(2)
        S = S[:, self.row, self.col] + (Pi >= eps).type(torch.float32)
        S[S < 2] = 0.0
        S[S > 0] = 1.0
        return S

    def forward_weight(self, users, x):
        V = self.M @ self.V(x)
        w = self.w(x) @ self.M.t()
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, sigma = self.inference(interaction[:, self.row, self.col])
        interaction = reparameterize(mu, sigma)

        pi = self.probability[users]
        last_index = pi.shape[1] - 1
        x = x.clone()
        x[x == -1] = last_index
        pi = torch.gather(pi, 1, x)
        Pi = pi.unsqueeze(2) @ pi.unsqueeze(1)
        Pi = Pi[:, self.row, self.col]
        S = self.selection(pi, Pi)

        y = torch.sum(w, dim=1) + self.w0 + torch.sum(interaction * S, dim=1)
        regular = -0.5 * self.beta * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return y, regular

    def forward_select(self, u, x):
        V = self.M @ self.V(x)
        w = self.w(x) @ self.M.t()
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, sigma = self.inference(interaction[:, self.row, self.col])
        interaction = reparameterize(mu, sigma)

        # pi = self.probability[u][x]
        pi = self.probability[u]
        last_index = pi.shape[1] - 1
        x = x.clone()
        x[x == -1] = last_index
        pi = torch.gather(pi, 1, x)

        Pi = pi.unsqueeze(2) @ pi.unsqueeze(1)
        Pi = Pi[:, self.row, self.col]
        S = self.selection(pi, Pi)

        y = torch.sum(w.detach(), dim=1).unsqueeze(-1) + self.w0.detach()
        interaction = interaction.detach()
        sumV = torch.sum(interaction * S, dim=1).unsqueeze(-1)
        y = y + sumV + Pi * (-S + 1) * interaction - (- Pi + 1) * S * interaction
        regular = self.alpha * entropy(pi)
        return y, regular

    def forward(self, **inputs):
        x = inputs['x']
        u = inputs['u']
        y, regular = self.forward_select(u, x) if inputs['select'] else self.forward_weight(u, x)
        return y, regular

    def predict(self, **inputs):
        x = inputs['x']
        u = inputs['u']
        V = self.M @ self.V(x).detach()
        w = self.w(x).detach() @ self.M.t()
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, _ = self.inference(interaction[:, self.row, self.col])

        pi = self.probability[u]
        last_index = pi.shape[1] - 1
        x = x.clone()
        x[x == -1] = last_index
        pi = torch.gather(pi, 1, x)

        pij = pi.unsqueeze(2) @ pi.unsqueeze(1)
        Pi = pij * (1 + pi.unsqueeze(1) + pi.unsqueeze(2) - 2 * pij)
        Pi = Pi[:, self.row, self.col]
        y = torch.sum(w, dim=1) + self.w0 + torch.sum(mu * Pi, dim=1)
        return y

    @abc.abstractmethod
    def inference(self, V):
        pass


class PFIS(FIS):
    def __init__(self, M, **inputs):
        super(PFIS, self).__init__(M, **inputs)
        self.mu_linear = nn.Linear(inputs['d'], 1)
        self.sigma_linear = nn.Linear(inputs['d'], 1)

    def inference(self, V):
        # mu = self.mu_linear(V).squeeze(-1)
        mu = torch.sum(V, dim=-1)
        sigma = self.sigma_linear(V).squeeze(-1)
        return mu, sigma


class PNFIS(FIS):
    def __init__(self, M, **inputs):
        super(PNFIS, self).__init__(M, **inputs)
        self.mu_linear = nn.Linear(inputs['d'], 1)
        self.sigma_linear = nn.Linear(inputs['d'], 1)
        self.network = nn.Sequential()
        for l in range(inputs['order'] - 2):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())

    def inference(self, V):
        output = self.network(V)
        mu = self.mu_linear(output).squeeze(-1)
        sigma = self.sigma_linear(output).squeeze(-1)
        return mu, sigma


if __name__ == '__main__':
    embedding = Embedding([2, 8], 0)
    print(embedding([0, 1, 1]))
