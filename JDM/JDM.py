'''
This is the script for joint distribution matching in domain generalization
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

from JDM.RESNET import get_fea
from JDM.common_network import feat_classifier, class_embedding
from JDM import adver_network

class JDM(nn.Module):
    
    def __init__(self,args):
        super(JDM, self).__init__()

        self.featurizer = get_fea(args)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.discriminator = adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.embedding = class_embedding(args.num_classes, self.featurizer.in_features)
        self.args = args

    def update(self, minibatches, opt, sch, temperature):
        x = torch.cat([data[0].cuda().float() for data in minibatches])
        y = torch.cat([data[1].cuda().long() for data in minibatches])
        z = self.featurizer(x)

        predict_y = self.classifier(z)
        classifier_loss = F.cross_entropy(predict_y, y)

        # not using sample method
        index_pre = F.softmax(predict_y / temperature, dim=1)
        # using gumbel to sample from the distribution of the preidct logit
        # index_pre = F.gumbel_softmax(predict_y, tau=temperature, hard=False)
        
        disc_input = z + self.embedding(index_pre)
        disc_input = adver_network.ReverseLayer.apply(
            disc_input, self.args.alpha
        )
        disc_out = self.discriminator(disc_input)
        disc_label = torch.cat([
            torch.full((data[0].shape[0], ), i, dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])
        disc_loss = F.cross_entropy(disc_out, disc_label)

        loss = classifier_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # if sch:
        #     sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))