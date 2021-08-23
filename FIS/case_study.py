import torch
from model.highorder import SFIS, HFIS
from train_fm import read_meta
from lib.dataset import DataCollection
import yaml
from tqdm import tqdm
import numpy as np

column_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
                'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
                'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
features = [1, 242, 250, 257, 3821, 8147, 8171, 13237, 13544, 13575, 291757, 1534649, 1541242, 1541248, 1541252,
            1543736, 1543745, 1543754, 1544186, 1544191, 1544259, 1544428, 1544488]


def get_Z(name):
    index = column_names.index(name)
    return model.Z(torch.arange(features[index], features[index + 1]))


def calculate_probability():
    num_user, num_feature, columns = read_meta('{}/meta.txt'.format(base_path))
    dataset = DataCollection(columns, f'{base_path}/ctr')
    d = 16
    alpha = 0
    beta = 0
    order = 2
    algo = 'HFIS'
    device = torch.device('cuda')
    M = dataset.calc_M().to(device)

    Model = eval(algo)
    model = Model(M, m=num_user, n=num_feature, d=d, alpha=alpha, beta=beta, order=order,
                  policy_layer=params['policy_layer'], num_layer=params['num_layer'], deep_layer=params['deep_layer'],
                  rate=params['selection_rate'], device=device).to(device)
    model_dir = '{}/{}'.format(params['model_dir'], params['data_name'])
    # model_path = '{}/{}_{}.state'.format(model_dir, args.algo, args.d)
    model_path = f"{params['model_dir']}/{params['data_name']}/{algo}_{d}.state"
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    loader = dataset.column_dataset('train', ['age', 'occupation'], params['batch_size'], False)
    probability = []
    for x, y in tqdm(loader):
        Z = model.Z(x)
        _, pro = model.selection(torch.prod(Z, 1))
        probability.append(pro.detach().cpu().numpy())
    probability = np.concatenate(probability)
    path = f"{params['result_dir']}/probability.txt"
    np.savetxt(path, probability, '%.4f')


if __name__ == '__main__':
    params = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    base_path = '{}/{}'.format(params['data_dir'], params['data_name'])
    calculate_probability()
    # path = f"{params['result_dir']}/probability.txt"
    # probability = np.loadtxt(path)
    # print(probability[probability > 0.6422])
