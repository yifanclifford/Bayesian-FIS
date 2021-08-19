import argparse

import torch
import yaml
from torch import optim
from torch.nn import functional
from lib.dataset import DataCollection
from lib.utils import mkdir_safe, EarlyStopping
from model.pairwise import PFIS, PNFIS
from model.highorder import SFIS, SNFIS, HFIS, SDeepFIS, HDeepFIS
from model.algorithm import ProgramFM, ProgramPersonal
from train_fm import evaluate_ctr, read_meta
import numpy as np


def running_fis(args, params):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    base_path = '{}/{}'.format(params['data_dir'], params['data_name'])
    num_user, num_feature, columns = read_meta('{}/meta.txt'.format(base_path))
    data_path = '{}/{}'.format(base_path, args.task)
    dataset = DataCollection(columns, data_path)
    if args.task == 'topn':
        train_loader, num_train = dataset.topn_dataset('train', params['batch_size'])
        valid_loader, num_valid = dataset.topn_dataset('valid', params['batch_size'], False)
        test_loader, num_test = dataset.topn_dataset('test', params['batch_size'], False)
    else:
        train_loader, num_train = dataset.ctr_dataset('train', params['batch_size'])
        valid_loader, num_valid = dataset.ctr_dataset('valid', params['batch_size'], False)
        test_loader, num_test = dataset.ctr_dataset('test', params['batch_size'], False)
    M = dataset.calc_M().to(device)

    Model = eval(args.algo)
    model = Model(M, m=num_user, n=num_feature, d=args.d, alpha=args.alpha, beta=args.beta, order=args.order,
                  policy_layer=params['policy_layer'], num_layer=params['num_layer'], deep_layer=params['deep_layer'],
                  rate=params['selection_rate'], device=device).to(device)
    loss_func = functional.binary_cross_entropy_with_logits if args.task == 'ctr' else functional.mse_loss

    select_params = [param for name, param in model.named_parameters() if 'probability' in name]
    weight_params = [param for name, param in model.named_parameters() if name not in select_params]

    weight_optimizer = optim.Adam(weight_params, lr=params['weight_lr'])
    select_optimizer = optim.Adam(select_params, lr=params['selection_lr'])

    procedure = ProgramPersonal(num_candidate=params['num_candidate'], device=device,
                                progress=args.progress, topn=10, metrics={'nDCG', 'HR'}) if 'P' in args.algo else \
        ProgramFM(num_candidate=params['num_candidate'], device=device, progress=args.progress, topn=10,
                  metrics={'nDCG', 'HR'})
    model_dir = '{}/{}'.format(params['model_dir'], params['data_name'])
    mkdir_safe(model_dir)
    # tmp_path = '{}/{}_{}.tmp'.format(model_dir, args.algo, args.d)
    model_path = '{}/{}_{}.state'.format(model_dir, args.algo, args.d)
    early = EarlyStopping(model_path, device, args.patience)
    # early = EarlyStopping(model_path, device, args.patience)

    # procedure.evaluate_ctr(model, test_loader)

    for epoch in range(args.epoch):
        early.reset()
        if args.initial:
            model.initial()
        print('iteration {}, training weights ========>'.format(epoch))
        for iter in range(params['weight_epoch']):
            train_loss = procedure.train(model, train_loader, weight_optimizer, num_train, loss_func)
            valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
            test_loss = procedure.test(model, test_loader, num_test, loss_func)
            # valid_loss, result = eval_func(procedure, model, dataset, valid_loader, test_loader,
            #                                loss_func, ['HR'], [1, 10])
            print(f'epoch={epoch}, {args.loss} train={train_loss:.4f} valid={valid_loss:.4f}; test {test_loss:.4f}')
            # valid_loss, test_result = evaluate_topn(procedure, model, dataset, valid_loader, test_loader, loss_func,
            #                                         ['HR'], [1, 10])
            # print('epoch={}, RMSE train={:.4f} valid={:.4f}; test {}'.format(iter, train_loss,
            #                                                                  valid_loss,
            #                                                                  test_result))
            early(valid_loss, model)
            if early.early_stop:
                break

        model.cpu()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
        test_loss = procedure.test(model, test_loader, num_test, loss_func)
        print(f'valid={valid_loss:.4f}; test {test_loss:.4f}')

        if args.task == 'ctr':
            results = procedure.evaluate_ctr(model, test_loader)
        else:
            loader, _ = dataset.topn_dataset('test', params['num_candidate'] * params['test_size'], False)
            results = procedure.evaluate_topn(model, loader, [10])
        print('Test: ' + ' '.join([f'{key}={value:.4f}' for key, value in results.items()]))

        print('iteration {}, training selection ========>'.format(epoch))
        for iter in range(params['selection_epoch']):
            train_loss = procedure.select(model, train_loader, select_optimizer, num_train, loss_func)
            valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
            test_loss = procedure.test(model, test_loader, num_test, loss_func)
            print(f'epoch={iter}, {args.loss} train={train_loss:.4f} valid={valid_loss:.4f}; test {test_loss:.4f}')
            # valid_loss, test_result = evaluate_topn(procedure, model, dataset, valid_loader, test_loader, loss_func,
            #                                         ['HR'], [1, 10])
            # print('RMSE train={:.4f} valid={:.4f}; test {}'.format(train_loss, valid_loss, test_result))


    model.cpu()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    valid_loss = procedure.test(model, valid_loader, num_valid, loss_func)
    test_loss = procedure.test(model, test_loader, num_test, loss_func)
    print(f'valid={valid_loss:.4f}; test {test_loss:.4f}')

    if args.task == 'ctr':
        results = procedure.evaluate_ctr(model, test_loader)
    else:
        loader, _ = dataset.topn_dataset('test', params['num_candidate'] * params['test_size'], False)
        results = procedure.evaluate_topn(model, loader, [10])
    print('Test: ' + ' '.join([f'{key}={value:.4f}' for key, value in results.items()]))
    return results
