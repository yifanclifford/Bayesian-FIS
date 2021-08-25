import pandas as pd
import yaml
from parse import parse

from lib.utils import mkdir_safe


def file2frame(filename):
    def read_term(text):
        return int(text.split(':')[0])

    def read_feature(text):
        return [read_term(term) for term in text.split(' ')]

    users = []
    items = []
    features = []
    ratings = []
    for line in open(filename, 'r'):
        result = parse('{:d} {:user} {:item} {:feature}', line,
                       dict(user=read_term, item=read_term, feature=read_feature))
        rating = result[0]
        user = result[1]
        item = result[2]
        feature = result[3]
        users.append(user)
        items.append(item)
        features.append(feature)
        ratings.append(rating)
    data = dict()
    data['user'] = users
    data['item'] = items
    for idx, feature in enumerate(zip(*features)):
        data['feature_{}'.format(idx)] = feature
    data['rating'] = ratings
    return pd.DataFrame(data)


def prepare_topn(split_data, out_path):
    split_data['item'] = split_data['item'] + num_user
    feature_columns = [column for column in split_data.columns if column not in ['user', 'item', 'rating']]
    split_data[feature_columns] = split_data[feature_columns] + num_user + num_item
    split_data.to_csv(out_path, index=False)


def topn(base_path):
    mkdir_safe(f'{base_path}/topn')
    train = file2frame(f'{base_path}/train.rank')
    valid = file2frame(f'{base_path}/valid.rank')
    test = file2frame(f'{base_path}/test.rank')
    prepare_topn(train, f'{base_path}/topn/train.csv')
    prepare_topn(valid, f'{base_path}/topn/valid.csv')
    prepare_topn(test, f'{base_path}/topn/test.csv')


if __name__ == '__main__':
    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.Loader)
    print('Processing the datasets (MLHt, lastFM, Delicious)')
    dataset = input('Name of dataset: ')
    base_path = f"{params['data_dir']}/{dataset}"
    with open('{}/meta.txt'.format(base_path)) as file:
        num_user, num_item, num_feature = parse('{:d}\t{:d}\t{:d}\n', file.readline())
        # columns = file.readline().split('\t')
        # columns = [int(column) for column in columns]
    topn(base_path)
