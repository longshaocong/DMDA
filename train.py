

import argparse
from utils.util import set_random_seed, alg_loss_dict, train_valid_target_eval_names
from datautil.getdataloader import get_img_dataloader
from JDM.JDM import JDM
from JDM.opt import *


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--net', type= str, default= 'resnet18')
    parser.add_argument('--num_classes', type= int, default=7)
    parser.add_argument('--dis_hidden', type= int, default= 256)
    parser.add_argument('--test_envs', type=list, default=[0])
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
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='for minbatch in each epoch')
    parser.add_argument('--lr_gamma', type=float, default=3e-4, help= 'for optimizer')

if __name__ == 'main':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loader, eval_loader = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    model = JDM.cuda()
    model.train()
    opt = optimizer(model, args)
    sch = scheduler(opt, args)

    
