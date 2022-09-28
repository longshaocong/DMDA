import torch

def accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = model.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).ep(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    model.train()
    return correct / total