import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import os
import sys
import pdb
import json
import yaml
from datasets import *
import argparse
from models import *
from kgdm import *
from utils import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing KGDM',
        usage='train_kgdm.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--dataset', type=str, default=True)
    parser.add_argument('--pretrain_emb', action='store_true')
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--dataset_neg', action='store_true', default=False)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-len', '--seq_len', default=3, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)

    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)

    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=5, type=int)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=50000, type=int)
    parser.add_argument('--valid_steps', default=20000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    parser.add_argument('--timesteps', type=int, help="the time steps of diffusion model, default:1000", default=1000)
    parser.add_argument('--ddim_sampling_timesteps', type=int, help='prefix of the log path', default=50)
    parser.add_argument('--modelconfig', type=str, default="./model_configs/Tnet.yaml", help='model yaml for exp')
    parser.add_argument('--dataset_onehot', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--exp_info', default=None, type=str, help='prefix of the log path')

    return parser.parse_args(args)


def override_config(args):
    with open(os.path.join('/', *args.init_checkpoint.split('/')[:-1], 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, mode, step=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    if mode == 'normal':
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path, 'checkpoint')
        )
        
    elif mode == 'best':
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path, 'best_checkpoint')
        )
        
    elif mode == 'interval':
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path, 'step' + str(step) + '_checkpoint')
        )
        

def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info(f'{mode} {metric} at step {step}: {metrics[metric]}')


if __name__ == '__main__':
    args = parse_args()
    modelconfig_path = args.modelconfig
    modelconfig = yaml.full_load(open(modelconfig_path)) 
    set_global_seed(args.seed)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    cur_time = parse_time()
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    
    dataset_str = 'dataset_onehot' if args.dataset_onehot else 'dataset_multihot'
    info_str = modelconfig['CEDenoise_Net']['name'] + '-' + dataset_str

    args.save_path = os.path.join(args.save_path, cur_time)
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    train_triples, nentity, nrelation = read_triples(args.data_path, 'train')
    test_triples, _, _ = read_triples(args.data_path, 'test')
    valid_triples, _, _ = read_triples(args.data_path, 'valid')
    
    logging.info('------------------------------'*3)
    logging.info(f'experiment information: {args.exp_info}')
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    logging.info('#train: %d' % len(train_triples))
    logging.info('#valid: %d' % len(valid_triples))
    logging.info('#test: %d' % len(test_triples))


    all_true_triples = train_triples + valid_triples + test_triples
    

    ce_denoiser = CEDenoise_Net(input_dim=args.hidden_dim,
                                with_time_emb=True,
                                max_seq_len=4,
                                modelconfigs=modelconfig)

    kgdm = KGDM(args.dataset,
                ce_denoiser,
                nentity,
                nrelation,
                args.hidden_dim,
                args.gamma,
                args.pretrain_emb,
                timesteps=args.timesteps,
                beta_schedule="cosine",
                linear_start=1e-4,
                linear_end=2e-2,
                max_seq_len=3,
                loss_type="huber",
                lr=1e-5,
                dataset_onehot=args.dataset_onehot,
                ddim_sampling_eta=0.,
                ddim_sampling_timesteps=args.ddim_sampling_timesteps,
                use_ensemble=args.use_ensemble,
                dataset_neg=args.dataset_neg,
                )

    logging.info('Model Parameter Configuration:')
    num_param = 0
    for name, param in kgdm.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_param += np.prod(param.size())
    logging.info(f'Parameter Number:{num_param}')

    if args.cuda:
        kgdm = kgdm.cuda()

    if args.do_train:
        if args.dataset_onehot:
            dataset_class = Train_KGDM_Onehot_Dataset
        if args.dataset_neg:
            dataset_class = Train_KGDM_Neg_Dataset
        else:
            dataset_class = Train_KGDM_Dataset

        train_dataloader_head = DataLoader(
            dataset_class(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=dataset_class.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            dataset_class(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=dataset_class.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kgdm.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(args.init_checkpoint)
        init_step = checkpoint['step']
        kgdm.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        best_mrr = 0.0

        for step in range(init_step, args.max_steps):
            log = kgdm.train_step(kgdm, optimizer, train_iterator, args)
            training_logs.append(log)
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kgdm.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kgdm, optimizer, save_variable_list, args, 'interval', step)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0 and step != 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kgdm.test_step(kgdm, valid_triples, all_true_triples, args, nentity=nentity, nrelation=nrelation)
                log_metrics('Valid', step, metrics)
                if metrics['MRR'] > best_mrr:
                    best_mrr = metrics['MRR']
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kgdm, optimizer, save_variable_list, args, 'best')
                
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kgdm, optimizer, save_variable_list, args, 'normal')

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kgdm.test_step(kgdm, valid_triples, all_true_triples, args, nentity, nrelation)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kgdm.test_step(kgdm, test_triples, all_true_triples, args, nentity, nrelation)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kgdm.test_step(kgdm, train_triples, all_true_triples, args, nentity, nrelation)
        log_metrics('Test', step, metrics)

