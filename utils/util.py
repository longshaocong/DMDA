import os
import sys
import random
import numpy as np 
import torch

def set_random_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def alg_loss_dict(args):
    loss_dict = {
                 'DMDA': ['class', 'distri', 'exp', 'extra', 'total']
                }
    return loss_dict[args.algorithm]

def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0    
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict

def print_args(args, print_list):
    s = '==========================================\n'
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += '{}:{}\n'.format(arg, content)
    return s


def save_checkpoint(filename, model, args):
    save_dict = {
        'args': vars(args), 
        'model_dict': model.cpu().state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


class Tee:
    def __init__(self, fname,mode='a'):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'office_home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'terra_incognita':
        domains = ['location_100', 'location_38', 'location_43', 'location_46']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'], 
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'office_home': ['Art', 'Clipart', 'Product', 'Real_World'], 
        'terra_incognita': ['location_100', 'location_38', 'location_43', 'location_46']
    }
    args.input_shape = (3, 32, 32)
    if args.dataset == 'PACS':
        args.num_classes = 7
    elif args.dataset == 'office_home':
        args.num_classes = 65
    elif args.dataset == 'VLCS':
        args.num_classes = 5
    elif args.dataset == 'terra_incognita':
        args.num_classes = 10
    return args