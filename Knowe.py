import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from util import count_acc, Averager, MultiClassCrossEntropy, load_model
from resnet12 import resnet12

model_pool = {
    "resnet50": lambda num_classes=2: resnet50(num_classes=num_classes),
    "resnet12": lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=num_classes),
    "resnet12forcifar": lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2,
                                                       num_classes=num_classes)
}


class Knowe(nn.Module):

    def __init__(self, args, norm=None):
        super().__init__()

        self.norm = norm
        self.args = args
        if self.args.dataset in ['cifar100']:
            model = model_pool['resnet12forcifar'](0)
            self.num_features = 640
        if self.args.dataset == 'tiered':
            model = model_pool['resnet12'](0)
            self.num_features = 640
        if self.args.dataset in ['living17', 'entity30', 'entity13', 'nonliving26']:
            model = model_pool['resnet50'](0)
            self.num_features = 2048
        if self.args.pretrained is not None:
            load_model(model, self.args.pretrained, False)
        self.encoder = model
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def encode(self, x):
        if self.args.dataset in ['cifar100', 'tiered']:
            x = self.encoder(x)
        else:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = self.encoder.avgpool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        x = self.encode(input)
        if self.norm is True:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            output = x / self.args.lambd

        elif self.norm is False:
            output = self.fc(x)
        return output

    def get_logits(self, x, fc):
        if self.args.norm is False:
            return F.linear(x, fc)
        elif self.args.norm is True:
            return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) / self.args.lambd

    def pre_update(self, dataloader, class_list, session):  # class_list new classes num
        if self.args.dataset == 'cifar100':
            self.args.new_class_num = len(class_list)
        if self.args.new_class_num * self.args.shot <= self.args.batch_size_new and self.args.decoupled is True:
            for batch in dataloader:
                data, label = [_.cuda() for _ in batch]
                label = label.to(torch.int64)
                if len(label.shape) == 2:
                    label = label.reshape(-1)
                if self.args.decoupled is False:
                    self.encoder.train()
                data = self.encode(data).detach()
        else:
            data, label = None, None

        if self.args.memory is True:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
        elif self.args.memory is False:
            new_fc = nn.Parameter(
                torch.rand(self.args.base_class + (session - 1) * self.args.way + len(class_list), self.num_features,
                           device="cuda"),
                requires_grad=True)  # FT 20+session*10
        nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        if self.args.method == 'subspace':
            self.args.new_class_num = len(class_list)
            self.subspace(new_fc, data, label, session)
        else:
            self.update(new_fc, data, label, session, dataloader)

    def update(self, new_fc, data, label, session, dataloader):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True

        if self.args.decoupled is False:
            optimized_parameters = [
                {'params': new_fc},
                {'params': filter(lambda p: p.requires_grad, self.encoder.parameters())},
            ]
            self.fc.requires_grad_(requires_grad=True)
        else:
            optimized_parameters = [{'params': new_fc}]

        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, weight_decay=0.0005)
        for epoch in range(self.args.epochs_new):

            if self.args.upperbound is False:
                if self.args.memory is True:
                    old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                elif self.args.memory is False:
                    old_fc = self.fc.weight[:0, :].detach()
            else:
                old_fc = self.fc.weight[:self.args.base_class, :].detach()
            if self.args.new_class_num * self.args.shot > self.args.batch_size_new or self.args.decoupled is False:
                true, total_num, total_loss = 0, 0, 0
                for batch in dataloader:
                    data, label = [_.cuda() for _ in batch]
                    label = label.to(torch.int64)
                    if len(label.shape) == 3:
                        label = label[0].reshape(-1)
                    if len(label.shape) == 2:
                        label = label.reshape(-1)
                    if self.args.decoupled is False:
                        self.encoder.train()
                    data = self.encode(data).detach()
                    fc = torch.cat([old_fc, new_fc], dim=0)
                    logits = self.get_logits(data, fc)

                    loss = F.cross_entropy(logits, label)
                    acc, total = count_acc(logits, label)

                    total_loss = total_loss + loss
                    true = true + acc
                    total_num = total_num + total
                loss = total_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if self.args.upperbound is False:
            if self.args.memory is False:
                self.fc.weight.data[:new_fc.shape[0], :].copy_(new_fc.data)
            else:
                self.fc.weight.data[
                self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session,
                :].copy_(
                    new_fc.data)
        else:
            self.fc.weight.data[self.args.base_class:self.args.base_class + self.args.way * session, :].copy_(
                new_fc.data)

    def train_LwF(self, optimizer, session, pre_model, trainloader):
        theta = 0.1
        tl = Averager()
        ta = Averager()
        tkl = Averager()
        tc = Averager()
        self.train()
        for i, batch in enumerate(trainloader, 1):
            data, label = [_.cuda() for _ in batch]
            label = label.to(torch.int64)
            if len(data.shape) == 5:
                data = data[0]
            if len(label.shape) == 3:
                label = label[0].reshape(-1)
            if len(label.shape) == 2:
                label = label.reshape(-1)
            logits_before = pre_model(data)
            logits = self(data)
            kl_loss = MultiClassCrossEntropy(logits[:, :self.args.base_class + (session - 1) * self.args.way],
                                             logits_before[:, :self.args.base_class + (session - 1) * self.args.way], 2)
            loss = F.cross_entropy(logits, label)
            total_loss = theta * loss + (1 - theta) * kl_loss
            true, num = count_acc(logits, label)
            tl.add(total_loss.item(), 1)
            ta.add(true, num)
            tkl.add(kl_loss.item(), 1)
            tc.add(loss.item(), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_tl = tl.item()
        acc = ta.item()
        avg_kl = tkl.item()
        avg_c = tc.item()
        return avg_tl, acc, avg_kl, avg_c

    def train_ScaIL(self, session):
        new = torch.load(os.path.join(self.args.save_path, f'fc_parameters_{session}.pth'))
        new_weights = new[0].detach().numpy()
        new_min_label = self.args.base_class + (session - 1) * self.args.way
        new_max_label = self.args.base_class + session * self.args.way
        scaled_weights = copy.deepcopy(new_weights)
        mea_dim_new = np.mean(-np.sort(-np.abs(new_weights[new_min_label:new_max_label, :])), axis=0)
        for b2 in range(0, session):  # b2 [0,session)
            old = torch.load(os.path.join(self.args.save_path, f'fc_parameters_{b2}.pth'))
            old_weights = old[0].detach().numpy()
            if b2 != 0:
                old_min_label = self.args.base_class + (b2 - 1) * self.args.way
                old_max_label = self.args.base_class + b2 * self.args.way
            else:
                old_min_label = 0
                old_max_label = self.args.base_class
            mean_dim_old = np.mean(-np.sort(-np.abs(old_weights[old_min_label:old_max_label, :])), axis=0)
            weights_factor = mea_dim_new / mean_dim_old
            for label in range(old_min_label, old_max_label):
                argsrt = np.argsort(-old_weights[label, :])
                for dim in range(old_weights.shape[1]):
                    scaled_weights[label][dim] = old_weights[label][dim] * weights_factor[argsrt[dim]]
        self.fc.weight.data.copy_(torch.tensor(scaled_weights))

    def subspace(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, weight_decay=0.0005)
        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].clone().detach()
                base_fc = self.fc.weight[:self.args.base_class, :].clone().detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                Q, R = torch.qr(base_fc.permute(1, 0), some=False)
                loss_reg2 = 0
                loss_reg1 = torch.norm(new_fc)
                for i in range(self.args.new_class_num):
                    fc_class = new_fc[i]
                    fc_class = torch.unsqueeze(fc_class, dim=-1)
                    mc = torch.mm(Q, fc_class)
                    new = fc_class.T - mc
                    loss_reg2 += torch.norm(new)
                if self.args.dataset == 'tiered':
                    loss = F.cross_entropy(logits,
                                           label) + 0.0001 * loss_reg1 + 0.0001 * loss_reg2
                elif self.args.dataset in ['living17', 'nonliving26', 'cifar100']:
                    loss = F.cross_entropy(logits, label) + 0.1 * loss_reg1 + 0.001 * loss_reg2
                else:  # entity13 entity30
                    loss = F.cross_entropy(logits, label) + 0.01 * loss_reg1 + 0.001 * loss_reg2
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            self.fc.weight.data[
            self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session,
            :].copy_(new_fc.data)

    def fc_align(self, session):
        old_weight = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
        new_weight = self.fc.weight.data[self.args.base_class + self.args.way * (
                session - 1):self.args.base_class + self.args.way * session, :]
        Norm_of_new = torch.norm(new_weight, dim=1)
        Norm_of_old = torch.norm(old_weight, dim=1)
        gamma = torch.mean(Norm_of_old) / torch.mean(Norm_of_new)
        updated_new_weight = gamma * new_weight
        self.fc.weight.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            updated_new_weight.data)
