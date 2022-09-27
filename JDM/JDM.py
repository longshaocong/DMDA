'''
This is the script for joint distribution matching in domain generalization
'''
import torch
import torch.nn as nn

from JDM.RESNET import get_fea
from JDM.common_network import feat_classifier, class_embedding
from JDM import adver_network

class JDM():
    
    def __init__(self,args):
        super(JDM, self).__init__(args)

        self.featurizer = get_fea(args)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.discriminator = adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.embedding = class_embedding(args.num_classes, self.featurizer.in_features)
        self.args = args