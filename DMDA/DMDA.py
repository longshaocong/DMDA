'''
This is the script for DMDA in domain generalization
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

from DMDA.RESNET import get_fea
from DMDA.common_network import feat_classifier, expert_classifier, semantic_embedding
from DMDA import Dis_App
from DMDA.Dis_App import WarmStartGradientReverseLayer

class DMDA(nn.Module):
    
    def __init__(self,args):
        super(DMDA, self).__init__()

        self.featurizer = get_fea(args)
        self.classifier_1 = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.expert_classifier = expert_classifier(args.num_classes, self.featurizer.in_features)
        self.Distribution = Dis_App.Distribution(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.embedding = semantic_embedding(args.num_classes, self.featurizer.in_features)
        self.args = args

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

    def update(self, minibatches, optimizer, lr_scheduler, temperature):
        x = torch.cat([data[0].cuda().float() for data in minibatches])
        y = torch.cat([data[1].cuda().long() for data in minibatches])
        
        z, map = self.featurizer(x)
        predict_y1 = self.classifier_1(z)
        extra_loss = F.cross_entropy(predict_y1, y)
        
        mask = self.classifier_1.fc.weight[y, :]
        a, idx = torch.sort(mask, descending=True)
        idx = idx[:,:int(self.featurizer.in_features * self.args.ratio)]

        bi_mask = torch.zeros(z.shape).cuda()
        bi_mask[:,idx] = 1
        z_new = z * bi_mask

        predict_y2 = self.classifier(z_new)
        classifier_loss = F.cross_entropy(predict_y2, y)
        

        cp_y = self.expert_classifier(z_new)
        exp_loss = F.cross_entropy(cp_y, y)
       
        index_pre = F.softmax(cp_y / temperature, dim=1)

        
        disc_input = z_new + self.embedding(index_pre) * bi_mask

        disc_input = self.grl(disc_input)
        disc_out = self.Distribution(disc_input)
        disc_label = torch.cat([
            torch.full((data[0].shape[0], ), i, dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])
        distri_loss = F.cross_entropy(disc_out, disc_label)

        
        loss = classifier_loss + 1 * (distri_loss + exp_loss) + self.args.beta * extra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'distri': distri_loss.item(), 'exp':exp_loss.item(), 
                'extra': extra_loss.item()}
    
    def predict(self, x):
        z, _ = self.featurizer(x)

        predict_y1 = self.classifier_1(z)
        pred_label = torch.max(predict_y1, dim=1)[1]
        mask = self.classifier_1.fc.weight[pred_label, :]

        a, idx = torch.sort(mask, descending=True)
        idx = idx[:,:int(self.featurizer.in_features * self.args.ratio)]

        bi_mask = torch.zeros(z.shape).cuda()
        bi_mask[:,idx] = 1
        z_new = z * bi_mask

        predict_y2 = self.classifier(z_new)
        return predict_y2