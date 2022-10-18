'''
This is the script for joint distribution matching in domain generalization
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

from JDM.RESNET import get_fea
from JDM.common_network import feat_classifier, projector, predictor

class CONTRA(nn.Module):
    
    def __init__(self,args):
        super(CONTRA, self).__init__()

        self.featurizer = get_fea(args)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.projector = projector(self.featurizer.in_features, args.pro_dim)
        self.predictor =  predictor(args.pro_dim, args.pre_dim)
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.args = args

    def update(self, minibatches, opt, sch):
        x1 = torch.cat([data[0].cuda().float() for data in minibatches])
        x2 = torch.cat([data[3].cuda().float() for data in minibatches])
        y = torch.cat([data[1].cuda().long() for data in minibatches])
        z = self.featurizer(x)

        predict_y = self.classifier(z)
        classifier_loss = F.cross_entropy(predict_y, y)

        z1 = self.projector(x1)
        z2 = self.projector(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        dez1 = z1.detach()
        dez2 = z2.detach()

        contrastive_loss = -(self.criterion(p1, dez2).mean() + self.criterion(p2, dez1).mean()) * 0.5

        loss = classifier_loss + contrastive_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'con': contrastive_loss.item()}
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))