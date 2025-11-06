'''
the optimizer and scheduler
'''
import torch


def get_params(model, args):
    if args.schuse:
        if args.schusech == 'cos':
            initlr = args.lr
        else:
            initlr = 1.
    else:
        initlr = args.lr
    params = []
    encode_params = [
        {'params': model.featurizer.parameters(), 'lr': args.lr_decay1 * initlr}, 
        {'params': model.classifier.parameters(), 'lr': args.lr_decay2 * initlr}
    ]
    
    if args.algorithm in ['DMDA']:
        encode_params.append(
            {'params': model.expert_classifier.parameters(), 'lr': args.lr_decay2 * initlr}
        )
        encode_params.append(
            {'params': model.Distribution.parameters(), 'lr': args.lr_decay2 * initlr})
        encode_params.append(
            {'params': model.embedding.parameters(), 'lr': args.lr_decay2 * initlr}
        )
        encode_params.append(
            {'params': model.classifier_1.parameters(), 'lr': args.lr_decay2 * initlr}
        )


    return encode_params


def optimizer(model, args):
    encode_params = get_params(model, args)
    optimizer = torch.optim.SGD(
        encode_params, lr= args.lr, momentum= args.momentum, weight_decay= args.weight_decay, nesterov= True
    )
    return optimizer

def scheduler(optimizer, args):
    if not args.schuse:
        return None
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
        )
    return scheduler