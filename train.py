

import argparse
import time
import numpy as np 
import os
import sys

from utils.util import set_random_seed, alg_loss_dict, train_valid_target_eval_names, print_args, save_checkpoint, Tee, img_param_init
from datautil.getdataloader import get_img_dataloader
from JDM import alg
from JDM.opt import *
from JDM import accu


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--net', type= str, default= 'resnet18')
    parser.add_argument('--num_classes', type= int, default=7)
    parser.add_argument('--dis_hidden', type= int, default= 256)
    parser.add_argument('--test_envs', type=int, nargs= '+', default=[0])
    parser.add_argument('--data_file', type=str, default='', help='root_dir')
    parser.add_argument('--dataset', type=str, default= 'PACS')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--split_style', type=str, default='strat')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--seed', type= int, default=3407)
    parser.add_argument('--algorithm', type=str, default='JDM')
    parser.add_argument('--domain_num', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for SGD')
    parser.add_argument('--lr_decay1', type=float, default=1., help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1., help='initial learning rate decay of network')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--max_epoch', type=int, default=100, help='for iterations')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='for the numbers of minbatch in each epoch')
    parser.add_argument('--lr_gamma', type=float, default=3e-4, help= 'for optimizer')
    parser.add_argument('--checkpoint_frep', type=int, default=1, help='checkpoint every N epoch')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--output', type=str, default='train_output', help='output path')
    parser.add_argument('--temperature', type=int, default=0.07, help='the temperature for the predicted logits')
    parser.add_argument('--alpha', type=float, default=0.5, help='the discriminator alpha')

    parser.add_argument('--contrast', action='store_true', help='if contrastive learning')
    parser.add_argument('--pro_dim', type=int, default=512, help='projection dim')
    parser.add_argument('--pre_dim', type=int, default=128, help='hidden dim  of the predictor')
    parser.add_argument('--CON_lambda', type=int, default=0.1, help='the trade off  for contrastive learning')

    parser.add_argument('--AR_lambda', type=float, default=0.01, help='the trade-off for AR')
    args = parser.parse_args()
    args.data_dir = args.data_file + args.data_dir
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    return args

if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loader, eval_loader = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    print('==========')
    algorithm = alg.get_algorithm_class(args.algorithm)
    model = algorithm(args).cuda()
    # if args.algorithm == 'JDM':
    #     model = JDM(args).cuda()
    # elif args.algorithm == 'CONTRA':
    #     model = CONTRA(args).cuda()
    model.train()
    opt = optimizer(model, args)
    sch = scheduler(opt, args)

    s = print_args(args, [])
    print('=================heper-parameter used========')
    print(s)
    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loader) # zip(*) get a source_domian_num*batch size for a minibatch, constituted by a batch from each source domain
    best_valid_acc, target_acc = 0, 0
    print('====================start training==============')
    start_time = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches = [(data) for data in next(train_minibatches_iterator)] #TODO: if the minibatches would be None in the last iter_num?
            step_vals = model.update(minibatches, opt, sch)

        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse): #TODO: change the strategy
            print('manually decrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr'] * 0.1

        if (epoch == (args.max_epoch - 1)) or (epoch % args.checkpoint_frep == 0):
            print('=================epoch %d=============='%(epoch))
            s = ''
            for item in loss_list:
                s += (item + '_loss:%.4f, '%(step_vals[item]))
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array(
                    [accu.accuracy(model, eval_loader[i]) for i in eval_name_dict[item]]
                ))
                s += (item + '_acc:%.4f, ' % acc_record[item])
            print(s[:-1])
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', model, args)
            print('total cost time: %.4f'%(time.time() - start_time))
            model_dict = model.state_dict()

    save_checkpoint('model.pkl', model, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)
    
    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time: %s\n'%(str(time.time() - start_time)))
        f.write('valid acc: %.4f'% best_valid_acc)
        f.write('target acc: %.4f'%target_acc)