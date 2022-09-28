import os
import torch

# seed setting with seed 3407
def set_random_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str[seed]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def alg_loss_dict(args):
    loss_dict = {'JDM': ['class', 'dis', 'total']
                }
    return loss_dict[args.algorithm]

def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0    
    '''t represent the index of the dataloader in eval_loader, e.g., eval_loader = [0, 1, 2, 0, 1, 2, 3], 
    where 0-4 is the proxy of the domian, the 4-th domain is the target domain. the first three 0 1 2 is the source domain, 
    while the latter 0 1 2 is the valid-set, and the last 3 is the target domain'''
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