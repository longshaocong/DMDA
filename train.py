

import argparse
from email.policy import default
from this import d


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--net', type= str, default= 'resnet18')
    parser.add_argument('--num_classes', type= int, default=7)
    parser.add_argument('--dis_hidden', type= int, default= 256)
    parser.add_argument('--test_envs', type=list, default=[0])
    parser.add_argument('--dataset', type=str, default= 'PACS')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--split_style', type=str, default='strat')
    parser.add_argument('--seed', type= int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--N_WORKERS', type=int, default=4)