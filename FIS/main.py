import argparse
from train_fm import running_fm
from train_fis import running_fis
import yaml
import simplejson as json
from lib.utils import current_timestamp
import faulthandler; faulthandler.enable()

# PFIS --epoch 20 --gpu -d 64 -a 0 -b 0 --initial
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature interaction selection for factorization machines')
    parser.add_argument('algo', help='specify model',
                        choices=['SparseFM', 'FM', 'AFM', 'PFM', 'SFM', 'HFM', 'HNFM', 'SNFM', 'PFIS', 'PNFIS',
                                 'NeurFM', 'PNFM', 'SDeepFM', 'HDeepFM', 'SFIS', 'SNFIS', 'HFIS', 'SDeepFIS', 'HNFIS',
                                 'HDeepFIS'])
    parser.add_argument('--task', help='CTR prediction or top-N recommendation', choices=['ctr', 'topn'], default='ctr')
    parser.add_argument('--loss', help='loss function', choices=['mse', 'log'], default='mse')
    parser.add_argument('--lr', help='learning rate for weights', type=float, default=1e-3)
    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--epoch', help='number of iterations', type=int, default=1)
    parser.add_argument('--progress', help='show progress bar', action='store_true')
    parser.add_argument('--order', help='order of feature interactions (valid only for high-order FMs)', type=int,
                        default=2)
    parser.add_argument('--drop', help='dropout rate', default=.5)
    parser.add_argument('--patience', help='early stopping patience', type=int, default=5)
    parser.add_argument('--initial', help='whether to initialize after iteraction (FIS only)', action='store_true')
    parser.add_argument('-a', '--alpha', help='layer regularization', type=float, default=0.001)
    parser.add_argument('-b', '--beta', help='layer regularization', type=float, default=0.001)
    parser.add_argument('-d', help='dimension of embeddings', default=8, type=int)
    args = parser.parse_args()

    params = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    if 'FM' in args.algo:
        results = running_fm(args, params)
    else:
        results = running_fis(args, params)

    results['data'] = params['data_name']
    results['algo'] = args.algo
    results['timestamp'] = current_timestamp()

    filename = f"{params['result_dir']}/result_{args.task}.txt"
    open(filename, 'a').write(f'{json.dumps(results)}\n')
    # if not os.path.isfile(filename):
    #     open(filename, 'w').write(','.join([col for col in col_names]) + '\n')
    # open(filename, 'a').write(','.join([str(res) for res in results]) + '\n')
