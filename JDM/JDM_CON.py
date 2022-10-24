'''
This is the script for joint distribution matching in domain generalization
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

from JDM.RESNET import get_fea
from JDM.common_network import feat_classifier, class_embedding, projector, predictor
from JDM import adver_network

class JDM_con(nn.Module):
    
    def __init__(self,args):
        super(JDM_con, self).__init__()

        self.featurizer = get_fea(args)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.discriminator = adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.embedding = class_embedding(args.num_classes, self.featurizer.in_features)

        # for contrastive learning
        self.projector = projector(self.featurizer.in_features, args.pro_dim)
        self.predictor =  predictor(args.pro_dim, args.pre_dim)
        self.criterion = nn.CosineSimilarity(dim=1).cuda()

        self.args = args

    def update(self, minibatches, opt, sch, temperature):
        x1 = torch.cat([data[0].cuda(non_blocking=True).float() for data in minibatches])
        x2 = torch.cat([data[3].cuda(non_blocking=True).float() for data in minibatches])
        y = torch.cat([data[1].cuda(non_blocking=True).long() for data in minibatches])

        Re1 = self.featurizer(x1)
        Re2 = self.featurizer(x2)

        # predict_y1 = self.classifier(Re1)
        # predict_y2 = self.classifier(Re2)
        # classifier_loss = (F.cross_entropy(predict_y1, y) + F.cross_entropy(predict_y2, y)) * 0.5

        z1 = self.projector(Re1)
        z2 = self.projector(Re2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        dez1 = z1.detach()
        dez2 = z2.detach()

        contrastive_loss = -(self.criterion(p1, dez2).mean() + self.criterion(p2, dez1).mean()) * 0.5

        z = self.featurizer(x1)
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

        loss = classifier_loss + disc_loss + self.args.CON_lambda * contrastive_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item(), 
                'con':contrastive_loss.item()}
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))