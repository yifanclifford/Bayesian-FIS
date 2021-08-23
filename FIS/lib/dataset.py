import pandas as pd
import torch
import yaml
from parse import parse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from lib.utils import read_split


def read_train(path):
    data = []
    for line in open(path, 'r'):
        user, item = parse('{:d}\t{:d}\n', line)
        data.append((user, item))
    return data


def read_test(path):
    data = []
    for line in open(path, 'r'):
        user, items = parse('{:d}\t{:item}', line, dict(item=read_split))
        data.append((user, items))
    return data


def collate_topn(batch):
    users, items, features, ratings = zip(*batch)
    users = torch.tensor(users, dtype=torch.long)
    items = torch.tensor(items, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.long)
    ratings = torch.tensor(ratings, dtype=torch.float32)
    return users, items, features, ratings


def collate_ctr(batch):
    x, y = zip(*batch)
    # x = torch.stack(x)
    # y = torch.stack(y)
    x = torch.LongTensor(x)
    y = torch.FloatTensor(y)
    return x, y


def collate_user(batch):
    user, items, features, ratings = zip(*batch)
    user = user[0]
    items = torch.tensor(items[0], dtype=torch.long)
    ratings = torch.tensor(ratings[0], dtype=torch.float32)
    features = torch.tensor(features[0], dtype=torch.long)
    return user, items, features, ratings


class TopnDataset(Dataset):
    def __init__(self, users, items, features, ratings):
        super(TopnDataset, self).__init__()
        self.users = users
        self.items = items
        self.features = features
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        feature = self.features[idx]
        rating = self.ratings[idx]
        return user, item, feature, rating


class CTRDataset(Dataset):
    def __init__(self, x, y):
        super(CTRDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


class DataCollection:
    def __init__(self, columns, dir_path):
        # self.m = m
        # self.n = n
        self.columns = columns
        self.dir_path = dir_path
        # self.train = pd.read_csv(dir_path + '/train.csv', index_col=False)
        # self.valid = pd.read_csv(dir_path + '/valid.csv', index_col=False)
        # self.test = pd.read_csv(dir_path + '/test.csv', index_col=False)

    def topn_dataset(self, split, batch_size, shuffle=True):
        # data = eval('self.' + split)
        data = pd.read_csv(f'{self.dir_path}/{split}.csv')
        users = data['user'].values.tolist()
        items = data['item'].values.tolist()
        ratings = data['rating'].values.tolist()
        features = data[[col for col in data.columns if col != 'rating']].values.tolist()
        return DataLoader(TopnDataset(users, items, features, ratings), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate_topn), len(users)

    def ctr_dataset(self, split, batch_size, shuffle=True):
        # path = f'{self.dir_path}/{split}.h5'
        # if os.path.isfile(path):
        #     data = pd.read_hdf(pd.HDFStore(path, mode='r'), mode='r')
        # else:
        data = pd.read_csv(f'{self.dir_path}/{split}.csv')
        x = data[[col for col in data.columns if col != 'rating']].values
        y = data['rating'].values
        return DataLoader(CTRDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_ctr), len(y)

    def column_dataset(self, split, columns, batch_size, shuffle=True):
        path = f'{self.dir_path}/{split}.h5'
        if os.path.isfile(path):
            data = pd.read_hdf(pd.HDFStore(path, mode='r'), mode='r')
        else:
            data = pd.read_csv(f'{self.dir_path}/{split}.csv')
        x = data[columns].values
        y = data['rating'].values
        return DataLoader(CTRDataset(x, y), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_ctr)

    # def user_dataset(self, split, shuffle=True):
    #     data = eval('self.' + split)
    #     data = data.groupby('user').agg(lambda x: [i for i in x]).reset_index()
    #     users = []
    #     items = []
    #     ratings = []
    #     features = []
    #     for _, row in data.iterrows():
    #         users.append(row['user'])
    #         items.append(row['item'])
    #         ratings.append(row['rating'])
    #         feature = row[[col for col in data.columns if col not in ['user', 'rating']]].tolist()
    #         feature = [[row['user']] + list(sample) for sample in zip(*feature)]
    #         features.append(feature)
    #     return DataLoader(FMDataset(users, items, features, ratings), batch_size=1, shuffle=shuffle,
    #                       collate_fn=collate_user)

    def calc_M(self):
        num_row = len(self.columns)
        num_col = sum(self.columns)
        M = torch.zeros(num_row, num_col)
        n = 0
        for row, col in enumerate(self.columns):
            M[row, n:(n + col)] = 1
            n += col
        return M
        # n = 1
        # num_single = len(self.single_feature)
        # for idx, fea in enumerate(self.single_feature[1:]):
        #     n += self.num_feature[idx]
        #     data.loc[:, fea] += n
        # for idx, fea in enumerate(self.multi_feature):
        #     n += self.num_feature[num_single - 1 + idx]
        #     data.loc[:, fea] = data[fea].apply(lambda x: [v + n - 1 for v in x])
        # input = [data[self.single_feature].values]
        # num_multi = []
        # for idx, fea in enumerate(self.multi_feature):
        #     max_len = max(data[fea].apply(lambda x: len(x)).values)
        #     num_multi.append(max_len)
        #     input.append(torch.from_numpy(pad_sequences(data[fea], maxlen=max_len, padding='post', dtype='int64')))
        # input = np.concatenate(input, axis=1)
        # for c in self.columns:
        #
        # m = num_single + len(num_multi)
        # n = num_single + sum(num_multi)
        # M = torch.zeros(m, n)
        # for i in range(num_single):
        #     M[i, i] = 1
        # n = num_single
        # for idx, i in enumerate(num_multi):
        #     M[num_single + idx, n:(n + i)] = 1
        #     n += i
        # return FMData(input, data['rating'].values.astype('float32')), M


# class MLHt(FMDataset):
#     def __init__(self, base_path):
#         super(MLHt, self).__init__(base_path)
#
#     def read_train(self, path):


# class ML1m(FMDataset):
#     def __init__(self, dir_path, binary, thres):
#         super(ML1m, self).__init__(dir_path, binary, thres)
#
#     def read_data(self, file_name):
#         data = pickle.load(open(file_name, 'rb'))
#         data['genres'] = data['genres'].apply(lambda x: [int(v) for v in x.split('|')])
#         return data

if __name__ == '__main__':
    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.Loader)
    base_path = '{}/{}'.format(params['data_dir'], params['data_name'])
    dataset = DataCollection(base_path)
    train_loader = dataset.topn_dataset('train')
    for users, items, features, ratings in train_loader:
        print(users)
        print(features.shape)
        print(ratings.shape)
