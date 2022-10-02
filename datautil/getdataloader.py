'''
to get dataloader
'''

import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

from datautil.imgdata.imgdataload import ImageDataset
import datautil.imgdata.util as imgutil
from datautil.infdataloader import InfiniteDataLoader

def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.data_dir,
                                            names[i], i, transform=imgutil.img_test(args.dataset), test_envs= args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.data_dir, 
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size= rate, train_size=1 - rate, random_state= args.seed
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                raise Exception('the split style is not strat')

            trdatalist.append(ImageDataset(args.dataset, args.data_dir, 
                                            names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.data_dir, 
                                            names[i], i, transform=imgutil.img_test(args.dataset), indices=indexte, test_envs=args.test_envs))
    
    train_loader = [InfiniteDataLoader(
        dataset = env, 
        weights = None, 
        batch_size = args.batch_size, 
        num_workers = args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset = env, 
        batch_size = 64, 
        num_workers = args.N_WORKERS, 
        drop_last = False, 
        shuffle = False)
        for env in trdatalist + tedatalist]

    return train_loader, eval_loaders
