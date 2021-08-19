import abc

import torch
from torch import nn

from model.utils import Embedding, trace, create_mask, cross_entropy


class FM(nn.Module):
    def __init__(self, M, **inputs):
        super(FM, self).__init__()
        self.w0 = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.normal_(self.w0, 0, 0.01)
        self.w = Embedding([inputs['n']], 0, inputs['device'])
        self.order = inputs['order']
        self.device = inputs['device']
        self.alpha = inputs['alpha']
        self.beta = inputs['beta']
        self.num_field = M.shape[0]
        self.M = M

    def inference(self, V):
        return torch.sum(V, -1)

    def predict(self, **inputs):
        y, _ = self(**inputs)
        return y.detach()

    def regularization(self, w, V):
        return self.alpha * trace(w) + self.beta * trace(V)


class AFM(FM):
    def __init__(self, M, **inputs):
        super(AFM, self).__init__(M, **inputs)
        self.V = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])
        self.att_net = nn.Sequential()
        self.att_net.add_module('Input', nn.Linear(inputs['d'], inputs['d']))
        self.att_net.add_module('ReLU', nn.ReLU())
        self.att_net.add_module('Output', nn.Linear(inputs['d'], 1))
        self.drop = nn.Dropout(inputs['drop'])
        self.q = nn.Parameter(torch.empty(inputs['d']), requires_grad=True)
        nn.init.normal_(self.q, 0, 0.01)

    def forward(self, **inputs):
        w = self.w(inputs['x'])
        V = self.V(inputs['x'])
        V0 = self.M @ V
        V = V0[:, :-1]
        V0 = V0[:, 1:]
        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field)).to(self.device)
        V = V0.unsqueeze(-2) * V.unsqueeze(-3)
        W = self.inference(V[:, mask])
        y += torch.sum(self.drop(W), dim=-1)
        return y, self.regularization(w, V)

    def inference(self, V):
        return V @ self.q

    def predict(self, **inputs):
        y, _ = self(**inputs)
        return y

    def regularization(self, w, V):
        return self.alpha * trace(w) + self.beta * trace(V)


class SFM(FM):
    def __init__(self, M, **inputs):
        super(SFM, self).__init__(M, **inputs)
        self.V = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])

    def forward(self, **inputs):
        w = self.w(inputs['x'])
        V = self.V(inputs['x'])
        V0 = self.M @ V
        V = V0[:, :-1]
        V0 = V0[:, 1:]
        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field)).to(self.device)
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            W = self.inference(V[:, mask])
            y += torch.sum(W, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[:, 1:]
            V = V[:, M_minus]
            mask = create_mask(M_minus)
        return y, self.regularization(w, V)


class SNFM(SFM):
    def __init__(self, M, **inputs):
        super(SNFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        for l in range(inputs['order'] - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(inputs['d'], 1))

    def inference(self, V):
        return self.network(V).squeeze(-1)


class SDeepFM(SFM):
    def __init__(self, M, **inputs):
        super(SDeepFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        layer = [M.shape[1] * inputs['d']] + inputs['deep_layer']
        for l in range(len(layer) - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(layer[l], layer[l + 1]))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(layer[-1], 1))

    def forward(self, **inputs):
        fm_score, fm_reg = super(SDeepFM, self).forward(**inputs)
        V = torch.flatten(self.M @ self.V(inputs['x']), 1, 2)
        deep_score = self.network(V).squeeze(-1)
        return fm_score + deep_score, fm_reg


class HFM(FM):
    def __init__(self, M, **inputs):
        super(HFM, self).__init__(M, **inputs)
        self.V = Embedding([inputs['order'] - 1, inputs['n'], inputs['d']], 1, inputs['device'])

    def forward(self, **inputs):
        w = self.w(inputs['x'])
        V = self.V(inputs['x'])
        V0 = self.M @ V
        V = V0[:, :, :-1]
        V0 = V0[:, :, 1:]
        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field)).to(self.device)
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            W = self.inference(V[0][:, mask])
            y += torch.sum(W, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[1:, :, 1:]
            V = V[1:, :, M_minus]
            mask = create_mask(M_minus)
        return y, self.regularization(w, V)


class HNFM(HFM):
    def __init__(self, M, **inputs):
        super(HNFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        for l in range(inputs['num_layer'] - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(inputs['d'], 1))

    def inference(self, V):
        return self.network(V).squeeze(-1)


class HDeepFM(HFM):
    def __init__(self, M, **inputs):
        super(HDeepFM, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        layer = [M.shape[1] * inputs['d']] + inputs['deep_layer']
        for l in range(len(layer) - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(layer[l], layer[l + 1]))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(layer[-1], 1))

    def forward(self, **inputs):
        fm_score, fm_reg = super(HDeepFM, self).forward(**inputs)
        V = self.M @ self.V(inputs['x'])[0]
        V = torch.flatten(V, 1, 2)
        deep_score = self.network(V).squeeze(-1)
        return fm_score + deep_score, fm_reg


class FIS(nn.Module):
    def __init__(self, M, **inputs):
        super(FIS, self).__init__()
        self.w0 = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.normal_(self.w0, 0, 0.01)
        self.w = Embedding([inputs['n']], 0, inputs['device'])
        self.Z = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])
        self.order = inputs['order']
        self.device = inputs['device']
        self.alpha = inputs['alpha']
        self.beta = inputs['beta']
        self.rate = inputs['rate']
        self.num_field = M.shape[0]
        self.M = M

        # self.probability = nn.Parameter(torch.full(inputs['n'], inputs['rate']), requires_grad=True)
        self.probability = nn.Sequential()
        layer = [inputs['d']] + inputs['policy_layer']
        for l in range(len(layer) - 1):
            self.probability.add_module('Layer {}'.format(l), nn.Linear(layer[l], layer[l + 1]))
            self.probability.add_module('ReLU {}'.format(l), nn.ReLU())
        self.probability.add_module('Output', nn.Linear(layer[-1], 1))
        self.probability.add_module('Sigmoid', nn.Sigmoid())

    def selection(self, Z):
        pro = self.probability(Z).squeeze(-1)
        eps = torch.rand_like(pro)
        S = (pro >= eps).type(torch.float32)
        return S, pro

    # def clamp(self):
    #     self.probability.data[self.probability.data < 0] = 0
    #     self.probability.data[self.probability.data > 1] = 1

    @abc.abstractmethod
    def forward_infer(self, x):
        pass

    @abc.abstractmethod
    def forward_policy(self, x):
        pass

    def forward(self, **inputs):
        x = inputs['x']
        y, regular = self.forward_policy(x) if inputs['select'] else self.forward_infer(x)
        return y, regular

    def predict(self, **inputs):
        pass

    def clamp(self):
        pass


class SFIS(FIS):
    def __init__(self, M, **inputs):
        super(SFIS, self).__init__(M, **inputs)
        self.V = Embedding([inputs['n'], inputs['d']], 0, inputs['device'])

    def inference(self, V):
        return torch.sum(V, -1)

    def forward_infer(self, x):
        w = self.w(x)
        V = self.V(x)
        Z = self.Z(x).detach()

        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :-1]
        V0 = V0[:, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]

        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field).to(self.device))
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[:, mask])
            S, _ = self.selection(Z[:, mask])
            y += torch.sum(W * S, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[:, 1:]
            V = V[:, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        return y, self.beta * (trace(w) + trace(V))

    def forward_policy(self, x):
        w = self.w(x).detach()
        V = self.V(x).detach()
        Z = self.Z(x)

        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :-1]
        V0 = V0[:, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]

        mask = create_mask(torch.eye(self.num_field).to(self.device))
        y = self.w0.detach() + torch.sum(w, dim=1).unsqueeze(-1)
        predictions = []
        regular = 0
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[:, mask])
            S, pro = self.selection(Z[:, mask])
            target = torch.full_like(pro, self.rate)
            regular += cross_entropy(pro, target)
            y += torch.sum(W * S, dim=-1).unsqueeze(-1)
            # y += sumV
            # y += torch.sum(W * S, dim=-1)
            pred = pro * (- S + 1) * W - (- pro + 1) * S * W
            predictions.append(pred)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[:, 1:]
            V = V[:, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        predictions = torch.cat(predictions, dim=1) + y
        return predictions, self.alpha * regular

    def predict(self, **inputs):
        w = self.w(inputs['x']).detach()
        V = self.V(inputs['x']).detach()
        Z = self.Z(inputs['x']).detach()

        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :-1]
        V0 = V0[:, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]

        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field).to(self.device))
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[:, mask])
            _, pro = self.selection(Z[:, mask])
            y += torch.sum(W * pro, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[:, 1:]
            V = V[:, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        return y


class SNFIS(SFIS):
    def __init__(self, M, **inputs):
        super(SNFIS, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        for l in range(inputs['num_layer'] - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(inputs['d'], 1))

    def inference(self, V):
        return self.network(V).squeeze(-1)


class SDeepFIS(SFIS):
    def __init__(self, M, **inputs):
        super(SDeepFIS, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        layer = [M.shape[1] * inputs['d']] + inputs['deep_layer']
        for l in range(len(layer) - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(layer[l], layer[l + 1]))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(layer[-1], 1))

    def forward(self, **inputs):
        V = torch.flatten(self.M @ self.V(inputs['x']), 1, 2)
        deep_score = self.network(V)
        if not inputs['select']:
            deep_score = deep_score.squeeze(-1).detach()
        fm_score, regular = super(SDeepFIS, self).forward(**inputs)
        return fm_score + deep_score, regular


class HFIS(FIS):
    def __init__(self, M, **inputs):
        super(HFIS, self).__init__(M, **inputs)
        self.V = Embedding([inputs['order'] - 1, inputs['n'], inputs['d']], 1, inputs['device'])

    def inference(self, V):
        return torch.sum(V, -1)

    def forward_infer(self, x):
        w = self.w(x)
        V = self.V(x)
        Z = self.Z(x).detach()
        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :, :-1]
        V0 = V0[:, :, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]

        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field).to(self.device))
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[0][:, mask])
            S, _ = self.selection(Z[:, mask])
            y += torch.sum(W * S, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[1:, :, 1:]
            V = V[1:, :, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        return y, self.beta * (trace(w) + trace(V))

    def forward_policy(self, x):
        w = self.w(x).detach()
        V = self.V(x).detach()
        Z = self.Z(x)
        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :, :-1]
        V0 = V0[:, :, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]
        mask = create_mask(torch.eye(self.num_field).to(self.device))
        y = self.w0.detach() + torch.sum(w, dim=1).unsqueeze(-1)
        predictions = []
        regular = 0
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[0][:, mask])
            S, pro = self.selection(Z[:, mask])
            target = torch.full_like(pro, self.rate)
            regular += cross_entropy(pro, target)
            y += torch.sum(W * S, dim=-1).unsqueeze(-1)
            pred = pro * (- S + 1) * W - (- pro + 1) * S * W
            predictions.append(pred)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[1:, :, 1:]
            V = V[1:, :, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        predictions = torch.cat(predictions, dim=1) + y
        return predictions, self.alpha * regular

    def predict(self, **inputs):
        w = self.w(inputs['x']).detach()
        V = self.V(inputs['x']).detach()
        Z = self.Z(inputs['x']).detach()
        V0 = self.M @ V
        Z0 = self.M @ Z
        V = V0[:, :, :-1]
        V0 = V0[:, :, 1:]
        Z = Z0[:, :-1]
        Z0 = Z0[:, 1:]

        y = self.w0 + torch.sum(w, dim=1)
        mask = create_mask(torch.eye(self.num_field).to(self.device))
        for t in range(2, self.order + 1):
            V = V0.unsqueeze(-2) * V.unsqueeze(-3)
            Z = Z0.unsqueeze(-2) * Z.unsqueeze(-3)
            W = self.inference(V[0][:, mask])
            _, pro = self.selection(Z[:, mask])
            y += torch.sum(W * pro, dim=-1)
            M_minus = mask.clone()
            M_minus[-1, :] = 0
            V0 = V0[1:, :, 1:]
            V = V[1:, :, M_minus]
            Z0 = Z0[:, 1:]
            Z = Z[:, M_minus]
            mask = create_mask(M_minus)
        return y


class HNFIS(HFIS):
    def __init__(self, M, **inputs):
        super(HNFIS, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        for l in range(inputs['num_layer'] - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(inputs['d'], inputs['d']))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(inputs['d'], 1))

    def inference(self, V):
        return self.network(V).squeeze(-1)


class HDeepFIS(HFIS):
    def __init__(self, M, **inputs):
        super(HDeepFIS, self).__init__(M, **inputs)
        self.network = nn.Sequential()
        layer = [M.shape[1] * inputs['d']] + inputs['deep_layer']
        for l in range(len(layer) - 1):
            self.network.add_module('Layer {}'.format(l), nn.Linear(layer[l], layer[l + 1]))
            self.network.add_module('ReLU {}'.format(l), nn.ReLU())
        self.network.add_module('Output', nn.Linear(layer[-1], 1))

    def forward(self, **inputs):
        V = torch.flatten(self.M @ self.V(inputs['x'])[0], 1, 2)
        deep_score = self.network(V)
        if not inputs['select']:
            deep_score = deep_score.squeeze(-1).detach()
        fm_score, regular = super(HDeepFIS, self).forward(**inputs)
        return fm_score + deep_score, regular
