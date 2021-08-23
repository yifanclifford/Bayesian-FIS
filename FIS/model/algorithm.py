import math

import torch
from torch.nn import functional
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from lib.utils import Evaluator, log_loss_logits
import numpy as np

metric_names = {'HR': 'recall', 'ARHR': 'recip_rank_cut', 'nDCG': 'ndcg_cut'}


class ProgramFM:
    def __init__(self, **inputs):
        self.device = inputs['device']
        self.progress = inputs['progress']
        self.num_candidate = inputs['num_candidate']
        self.N = inputs['topn']
        self.evaluator = Evaluator({metric_names[metric] for metric in inputs['metrics']})
        self.metrics = inputs['metrics']

    def train(self, model, loader, optimizer, num_sample, loss_func):
        train_loss = 0
        model.train()
        for x, y in tqdm(loader, disable=not self.progress, unit='batch'):
            optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            score, regular = model(x=x, select=False)
            loss = loss_func(score, y, reduction='sum')
            train_loss += loss.item()
            loss += regular
            loss.backward()
            optimizer.step()
        return train_loss / num_sample

    def test(self, model, loader, num_sample, loss_func):
        test_loss = 0
        model.eval()
        for features, ratings in tqdm(loader, disable=not self.progress, unit='batch'):
            features = features.to(self.device)
            ratings = ratings.to(self.device)
            score = model.predict(x=features).detach()
            loss = loss_func(score, ratings, reduction='sum')
            test_loss += loss.item()
        return test_loss / num_sample

    def predict(self, model, loader):
        scores = []
        for features, _ in tqdm(loader, disable=not self.progress):
            features = features.to(self.device)
            score = model.predict(x=features).detach().cpu().numpy()
            scores.append(score)
        return np.concatenate(scores)

    def select(self, model, loader, optimizer, num_sample, loss_func):
        train_loss = 0
        model.train()
        for features, ratings in tqdm(loader, disable=not self.progress):
            optimizer.zero_grad()
            features = features.to(self.device)
            score, regular = model(x=features, select=True)
            ratings = ratings.unsqueeze(-1).expand(-1, score.shape[1]).to(self.device)
            loss = loss_func(score, ratings, reduction='sum')
            train_loss += loss.item()
            loss += regular
            loss.backward()
            optimizer.step()
            model.clamp()
        return train_loss / num_sample

    # def evaluate_topn(self, model, loader, cuts):
    #     model.eval()
    #     run = dict()
    #     test = dict()
    #     for users, items, features, _ in tqdm(loader, disable=not self.progress):
    #         features = features.to(self.device)
    #         items = items.to(self.device)
    #         score = model.predict(x=features)
    #         score = score.reshape(-1, self.num_candidate)
    #         score, index = torch.topk(score, self.N, dim=1)
    #         users = users.reshape(-1, self.num_candidate)[:, 0].tolist()
    #         test_items = items.reshape(-1, self.num_candidate)[:, 0].tolist()
    #         items = items.reshape(-1, self.num_candidate)
    #         items = torch.gather(items, 1, index)
    #         for idx, user in enumerate(users):
    #             run[str(user)] = {str(items[idx, n].item()): float(score[idx, n].item()) for n in range(self.N)}
    #             test[str(user)] = {str(test_items[idx]): int(1)}
    #     self.evaluator.evaluate(run, test)
    #     result = self.evaluator.show(['{}_{}'.format(metric, cut) for metric in self.metrics for cut in cuts])
    #     return result

    def evaluate_topn(self, model, loader, cuts):
        model.eval()
        run = dict()
        test = dict()
        for users, items, features, ratings in tqdm(loader, disable=not self.progress):
            features = features.to(self.device)
            items = items.to(self.device)
            score = model.predict(x=features)
            score = score.reshape(-1, self.num_candidate)
            score, index = torch.topk(score, self.N, dim=1)
            users = users.reshape(-1, self.num_candidate)[:, 0].tolist()
            ratings = ratings.reshape(-1, self.num_candidate)[:, 0].tolist()
            test_items = items.reshape(-1, self.num_candidate)[:, 0].tolist()
            items = items.reshape(-1, self.num_candidate)
            items = torch.gather(items, 1, index)
            for idx, user in enumerate(users):
                run[str(user)] = {str(items[idx, n].item()): float(score[idx, n].item()) for n in range(self.N)}
                test[str(user)] = {str(test_items[idx]): int(ratings[idx])}
        self.evaluator.evaluate(run, test)
        results = self.evaluator.show(
            [f'{metric_names[metric]}_{cut}' for metric in self.metrics for cut in cuts])
        return {f'{metric}@{cut}': results[f'{metric_names[metric]}_{cut}'] for metric in self.metrics for cut in cuts}
        # return {metric: result for metric, result in zip(metrics, results)}

    def evaluate_ctr(self, model, loader):
        model.eval()
        y_pred = []
        y_label = []
        for features, ratings in tqdm(loader, disable=not self.progress):
            features = features.to(self.device)
            score = model.predict(x=features).detach()
            score = torch.sigmoid(score)
            score = np.clip(score.cpu().numpy(), 1e-8, 1 - 1e-8)
            y_pred.append(score)
            y_label.append(ratings.numpy())
        y_pred = np.concatenate(y_pred)
        y_label = np.concatenate(y_label)
        loss = log_loss(y_label, y_pred)
        auc = roc_auc_score(y_label, y_pred)
        return {'log_loss': loss, 'AUC': auc}
        # return (log_loss, auc), ['log_loss', 'AUC']


class ProgramPersonal(ProgramFM):
    def __init__(self, **inputs):
        super(ProgramPersonal, self).__init__(**inputs)

    def select(self, model, loader, optimizer, num_sample, loss_func):
        train_loss = 0
        model.train()
        for users, _, features, ratings in tqdm(loader, disable=not self.progress):
            optimizer.zero_grad()
            users = users.to(self.device)
            features = features.to(self.device)
            score, regular = model(u=users, x=features, select=True)
            ratings = ratings.unsqueeze(-1).expand(-1, score.shape[1]).to(self.device)
            loss = loss_func(score, ratings, reduction='sum')
            train_loss += loss.item()
            loss += regular
            loss.backward()
            optimizer.step()
            model.clamp()
        return train_loss / num_sample

    def train(self, model, loader, optimizer, num_sample, loss_func):
        train_loss = 0
        model.train()
        for users, _, features, ratings in tqdm(loader, disable=not self.progress):
            optimizer.zero_grad()
            users = users.to(self.device)
            features = features.to(self.device)
            ratings = ratings.to(self.device)
            score, regular = model(u=users, x=features, select=False)
            loss = loss_func(score, ratings, reduction='sum')
            train_loss += loss.item()
            loss += regular
            loss.backward()
            optimizer.step()
        return train_loss / num_sample

    def test(self, model, loader, num_sample, loss_func):
        model.eval()
        test_loss = 0
        for users, _, features, ratings in tqdm(loader, disable=not self.progress):
            users = users.to(self.device)
            features = features.to(self.device)
            ratings = ratings.to(self.device)
            score = model.predict(u=users, x=features)
            loss = loss_func(score.detach(), ratings, reduction='sum')
            test_loss += loss.item()
        return test_loss / num_sample

    def predict(self, model, loader):
        scores = []
        for users, _, features, ratings in tqdm(loader, disable=not self.progress):
            users = users.to(self.device)
            features = features.to(self.device)
            score = model.predict(u=users, x=features).detach()
            score = torch.sigmoid(score)
            scores.append(score.cpu().numpy())
        return np.concatenate(scores)

    def evaluate_ctr(self, model, loader):
        model.eval()
        y_pred = []
        y_label = []
        for users, _, features, ratings in tqdm(loader, disable=not self.progress):
            users = users.to(self.device)
            features = features.to(self.device)
            score = model.predict(u=users, x=features).detach()
            score = torch.sigmoid(score)
            score = score.cpu().numpy()
            score = np.clip(score, 1e-8, 1 - 1e-8)
            y_pred.append(score)
            y_label.append(ratings.numpy())
        y_pred = np.concatenate(y_pred)
        y_label = np.concatenate(y_label)
        loss = log_loss(y_label, y_pred)
        auc = roc_auc_score(y_label, y_pred)
        return {'log_loss': loss, 'AUC': auc}

    # def evaluate_topn(self, model, loader, cuts):
    #     model.eval()
    #     run = dict()
    #     test = dict()
    #     for users, items, features, ratings in tqdm(loader, disable=not self.progress):
    #         users = users.to(self.device)
    #         features = features.to(self.device)
    #         items = items.to(self.device)
    #         score = model.predict(u=users, x=features, select=False)
    #         score = score.reshape(-1, self.num_candidate)
    #         score, index = torch.topk(score, self.N, dim=1)
    #         users = users.reshape(-1, self.num_candidate)[:, 0].tolist()
    #         ratings = ratings.reshape(-1, self.num_candidate)[:, 0].tolist()
    #         test_items = items.reshape(-1, self.num_candidate)[:, 0].tolist()
    #         items = items.reshape(-1, self.num_candidate)
    #         items = torch.gather(items, 1, index)
    #         for idx, user in enumerate(users):
    #             run[str(user)] = {str(items[idx, n].item()): float(score[idx, n].item()) for n in range(self.N)}
    #             test[str(user)] = {str(test_items[idx]): int(ratings[idx])}
    #     self.evaluator.evaluate(run, test)
    #     results = self.evaluator.show(
    #         [f'{metric_names[metric]}_{cut}' for metric in self.metrics for cut in cuts])
    #     return {f'{metric}@{cut}': results[f'{metric_names[metric]}_{cut}'] for metric in self.metrics for cut in cuts}
