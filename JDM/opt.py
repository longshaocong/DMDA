'''
the optimizer and scheduler
'''
import torch

def get_params(model, args):
    if args.schuse:
        if args.schusech = 'cos':
            initlr = args.lr
        else:
            initlr = 1.
    else:
        initlr = args.lr
    params = [
        {'params': model.featurizer.parameters(), 'lr': agrs.lr_decay1 * initlr}, 
        {'params': model.classifier.parameters(), 'lr': args.lr_decay2 * initlr}, 
        {'params': model.discriminator.parameters(), 'lr': args.lr_decay2 * initlr}, 
        {'params': model.embedding.parameters(), 'lr': args.lr_decay2 * initlr}
    ]
    return params


def optimizer(model, args):
    params = get_params(model, args)
    optimizer = torch.optim.SGD(
        params, lr= args.lr, momentum= args.momentum, weight_decay= args.weight_decay, nesterov= True
    )
    return optimizer


def scheduler(optimizer, args):
    if not args.schuse:
        return None
    # TODO: the T_max in cosineAnneallingLR is problematic, should be smaller
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
        )
    return scheduler