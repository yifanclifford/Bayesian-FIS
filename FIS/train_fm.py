import os

import numpy as np
import torch
from parse import parse
from torch import optim
from torch.nn import functional

from lib.dataset import DataCollection
from lib.utils import EarlyStopping, mkdir_safe
from model.algorithm import ProgramFM
from model.pairwise import FM, PFM, SparseFM, NeurFM
from model.highorder import AFM, HFM


# def evaluate_topn(procedure, model, dataset, valid_loader, test_loader, loss_func, metrics=None, cuts=None):
#     result = procedure.evaluate_topn(model, test_loader, cuts)
#     result = ['{}@{}={:.4f}'.format(metric, cut, result['{}_{}'.format(metric_names[metric], cut)])
#               for metric in metrics for cut in cuts]
#     test_result = ' '.join(result)
#     valid_loss = procedure.test(model, valid_loader, len(dataset.valid), loss_func)
#     return valid_loss, test_result


def evaluate_ctr(procedure, model, dataset, valid_loader, test_loader, loss_func, metrics=None, cuts=None):
    valid_loss = procedure.test(model, valid_loader, len(dataset.valid), loss_func)
    test_log, test_auc = procedure.evaluate_ctr(model, test_loader)
    return valid_loss, 'log={:.4f}, AUC={:.4}'.format(test_log, test_auc)


def read_meta(path):
    with open(path, 'r') as file:
        num_user, num_item, num_feature = parse('{:d}\t{:d}\t{:d}\n', file.readline())
        columns = file.readline().split('\t')
        columns = [int(column) for column in columns]
        num_feature += num_user + num_item
    return num_user, num_feature, columns


def running_fm(args, params):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    base_path = '{}/{}'.format(params['data_dir'], params['data_name'], args.task)
    _, num_feature, columns = read_meta('{}/meta.txt'.format(base_path))
    data_path = '{}/{}'.format(base_path, args.task)
    if not (os.path.exists(data_path) and os.path.isdir(data_path)):
        print('the dataset does not support {} task'.format(args.task))
        return
    dataset = DataCollection(columns, data_path)
    train_loader, num_train = dataset.ctr_dataset('train', params['batch_size'])
    valid_loader, num_valid = dataset.ctr_dataset('valid', params['batch_size'])
    # test_size = params['num_candidate'] * params['test_size'] if args.task == 'topn' else params['batch_size']
    test_loader, num_test = dataset.ctr_dataset('test', params['batch_size'], False)
    M = dataset.calc_M().to(device)

    Model = eval(args.algo)
    model = Model(M, n=num_feature, d=args.d, order=args.order, alpha=args.alpha,
                  deep_layer=params['deep_layer'], num_layer=params['num_layer'],
                  beta=args.beta, device=device, drop=args.drop).to(device)

    loss_func = functional.binary_cross_entropy_with_logits if args.loss == 'log' else functional.mse_loss
    # eval_func = eval('evaluate_{}'.format(args.task))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    procedure = ProgramFM(num_candidate=params['num_candidate'], device=device,
                          progress=args.progress, topn=params['topn'][-1], metrics={'ARHR', 'HR'})
    model_dir = f"{params['model_dir']}/{params['data_name']}"
    mkdir_safe(model_dir)
    model_path = '{}/{}_{}.state'.format(model_dir, args.algo, args.d)
    early = EarlyStopping(model_path, device, args.patience)

    for epoch in range(1, args.epoch + 1):
        train_loss = procedure.train(model, train_loader, optimizer, num_train, loss_func)
        valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
        test_loss = procedure.test(model, test_loader, num_test, loss_func)
        # valid_loss, result = eval_func(procedure, model, dataset, valid_loader, test_loader,
        #                                loss_func, ['HR'], [1, 10])
        print(f'epoch={epoch}, {args.loss} train={train_loss:.4f} valid={valid_loss:.4f}; test {test_loss:.4f}')
        early(valid_loss, model)
        if early.early_stop:
            break

    model.cpu()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # scores = procedure.predict(model, test_loader)
    # np.savetxt(f"{base_path}/pred.txt", scores)

    # valid_loss, result = eval_func(procedure, model, dataset, valid_loader, test_loader, loss_func,
    #                                ['HR', 'ARHR'], [1, 5, 10])
    train_loss = procedure.test(model, train_loader, num_train, loss_func)
    valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
    test_loss = procedure.test(model, test_loader, num_test, loss_func)
    print(f'{args.loss} train={train_loss:.4f} valid={valid_loss:.4f}, test={test_loss:.4f}')

    if args.task == 'ctr':
        results = procedure.evaluate_ctr(model, test_loader)
    else:
        loader, _ = dataset.topn_dataset('test', params['num_candidate'] * params['batch_size'], False)
        results = procedure.evaluate_topn(model, loader, params['topn'])

    print('Test: ' + ' '.join([f'{key}={value:.4f}' for key, value in results.items()]))
    return results
