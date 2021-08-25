import os
import pickle
import random

import pandas as pd
import yaml
from scipy import io
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from lib.utils import get_lib, sample_negative, form_sparse, mkdir_safe


def prepare_user(data):
    data = data.groupby('user')[['item', 'timestamp']].apply(lambda x: x.sort_values('timestamp'))
    user_items = []
    for idx, row in data.groupby(level=0):
        user_items.append((idx, row['item'].tolist()))
    return user_items


def prepare_hist(user_data, hist_len):
    hist_data = []
    for user, items in user_data:
        for i in range(len(items) - hist_len):
            hist_data.append((user, items[i:(i + hist_len)], items[i + hist_len], 1.0))
    return hist_data


def negative(data):
    lib = get_lib(dir_path + '/positive.mtx')
    neg_data = []
    for user, hist_items, item, _ in tqdm(data):
        neg_item = sample_negative(lib, user, 1)[0]
        neg_data.append((user, hist_items, neg_item, 0.0))
    data.extend(neg_data)
    return data


def list2frame(hist_data, hist_len):
    users, hist_items, items, ratings = zip(*hist_data)
    hist_items = list(zip(*hist_items))
    data = dict()
    data['user'] = users
    data['item'] = items
    for h in range(hist_len):
        data['hist_{}'.format(h)] = hist_items[h]
    data['rating'] = ratings
    return pd.DataFrame(data)


def topn(hist_len):
    mkdir_safe(f'{dir_path}/topn')
    data_path = f'{dir_path}/user_data.pkl'
    if not os.path.isfile(data_path):
        path = '{}/raw/ratings.csv'.format(dir_path)
        data = pd.read_csv(path, index_col=False).rename(columns={'userId': 'user', 'movieId': 'item'})
        encoder = LabelEncoder()
        data['user'] = encoder.fit_transform(data['user'])
        data['item'] = encoder.fit_transform(data['item'])
        user_data = prepare_user(data)
        pickle.dump(user_data, open(data_path, 'wb'))
    else:
        user_data = pickle.load(open(data_path, 'rb'))

    prepare_positive(user_data)
    lib = get_lib(f'{dir_path}/positive.mtx')

    hist_data = prepare_hist(user_data, hist_len)
    hist_data = negative(hist_data)
    data = list2frame(hist_data, hist_len)
    num_line = len(data)
    num_train = int(num_line * .8)
    num_valid = int(num_line * .1)
    num_test = num_line - num_train - num_valid
    split = []
    for i in range(num_train):
        split.append(0)
    for i in range(num_valid):
        split.append(1)
    for i in range(num_test):
        split.append(2)
    random.shuffle(split)

    head = ','.join([name for name in data.columns])
    train_writer = open(dir_path + '/topn/train.csv', 'w')
    train_writer.write(head + '\n')
    valid_writer = open(dir_path + '/topn/valid.csv', 'w')
    valid_writer.write(head + '\n')
    test_writer = open(dir_path + '/topn/test.csv', 'w')
    test_writer.write(head + '\n')

    for idx, row in enumerate(data.itertuples(index=False)):
        values = [v for v in row]
        line = ','.join([str(v) for v in values])
        if split[idx] == 0:
            train_writer.write(line + '\n')
        elif split[idx] == 1:
            valid_writer.write(line + '\n')
        else:
            test_writer.write(line + '\n')
            hist_line = ','.join([str(v) for v in values[2:]])
            neg_items = sample_negative(lib, row.user, 100)
            for item in neg_items:
                test_writer.write(f'{row.user},{item},{hist_line},0\n')
    train_writer.close()
    valid_writer.close()
    test_writer.close()


def prepare_positive(data):
    sparse_mat = form_sparse(data, num_user, num_item)
    io.mmwrite('{}/positive'.format(dir_path), sparse_mat)


if __name__ == '__main__':
    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.Loader)
    dir_path = '%s/ML20m' % (params['data_dir'])
    print('Processing the MovieLens 20 million dataset')
    hist_len = int(input('Input the number of historical items: '))

    num_user = 138493
    num_item = 26744

    with open(f'{dir_path}/meta.txt', 'w') as file:
        file.write(f'{num_user}\t{num_item}\t0\n')
        file.write('\t'.join(['1'] * (hist_len + 2)))
    topn(hist_len)
