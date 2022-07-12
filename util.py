import random

import torch.nn.functional as F
import torch
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable


def set_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def load_model(model, pretrained, with_mlp=False):
    for name, param in model.named_parameters():
        if name == 'fc.weight' or name == 'fc.bias':
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        load_encoder_q_state_dict(model, state_dict, with_mlp)
    else:
        raise NotImplementedError

def load_encoder_q_state_dict(model, state_dict, with_mlp=False):
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q.'):
            # remove prefix
            if not with_mlp and k.startswith('module.encoder_q.fc'):
                continue
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if not with_mlp:
        # assume vanilla resnet model
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        msg = model.load_state_dict(state_dict)

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x, num):
        self.v = self.v + x
        self.n = self.n + num

    def item(self):
        if self.v == 0 or self.n == 0:
            return 0
        return self.v / self.n


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return sum(pred == label).item(), len(label)


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        print('create folder:', path)


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def base_train(model, trainloader, optimizer, args):
    tl = Averager()
    ta = Averager()
    ta_coarse = Averager()
    model = model.train()
    for i, batch in enumerate(trainloader, 1):
        data, train_label = [_.cuda() for _ in batch]
        train_label = train_label.to(torch.int64)
        logits = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc_total, acc_len = count_acc(logits, train_label)
        total_loss = loss
        tl.add(total_loss.item(), acc_len)
        ta.add(acc_total, acc_len)
        ta_coarse.add(acc_total, acc_len)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    ta_coarse = ta_coarse.item()
    ta_fine = 0
    return tl, ta, ta_coarse, ta_fine


def test(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_coarse = Averager()
    va_fine = Averager()
    va_now = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            acc_now, now_len = 0, 0
            acc_coarse, coarse_len = 0, 0
            acc_fine, fine_len = 0, 0
            val_label_coarse = []
            val_label_fine = []
            val_label_now = []
            data, test_label = [_.cuda() for _ in batch]
            test_label = test_label.to(torch.int64)
            if len(test_label.shape) != 1:
                test_label = test_label[0].reshape(-1)
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)

            acc, acc_len = count_acc(logits, test_label)
            for j in range(len(test_label)):
                if test_label[j] <= args.base_class - 1:
                    val_label_coarse.append(j)
                else:
                    val_label_fine.append(j)
            if session >= 1:
                for k in range(len(test_label)):
                    if args.base_class + (session - 1) * args.way <= test_label[k] \
                            < args.base_class + session * args.way:
                        val_label_now.append(k)
                if len(val_label_now) != 0:
                    acc_now, now_len = count_acc(logits[val_label_now], test_label[val_label_now])
            else:
                acc_now, now_len = acc, acc_len
            if len(val_label_coarse) != 0:
                acc_coarse, coarse_len = count_acc(logits[val_label_coarse], test_label[val_label_coarse])
            if len(val_label_fine) != 0:
                acc_fine, fine_len = count_acc(logits[val_label_fine], test_label[val_label_fine])

            vl.add(loss.item(), acc_len)
            va.add(acc, acc_len)
            va_coarse.add(acc_coarse, coarse_len)
            va_fine.add(acc_fine, fine_len)
            va_now.add(acc_now, now_len)

        vl = vl.item()
        va = va.item()
        va_coarse = va_coarse.item()
        va_fine = va_fine.item()
        va_now = va_now.item()
    return vl, va, va_coarse, va_fine, va_now


def plot_fig(trlog, args):
    x = [i for i in range(args.sessions)]
    x_coarse = x[:args.sessions - 1]
    x_fine = x[1:args.sessions]

    y_max_acc = trlog['max_acc']
    y_coarse_acc = trlog['coarse_acc'][:args.sessions - 1]
    y_fine_acc = trlog['fine_acc'][1:]
    y_now_acc = trlog['now_acc']

    plt.plot(x, y_max_acc, 'ro-', color='blue', label='total_acc')
    plt.plot(x_coarse, y_coarse_acc, 'rs-', color='red', label='coarse_acc')
    plt.plot(x_fine, y_fine_acc, 'r^-', color='green', label='fine_acc')
    plt.scatter(x, y_now_acc, c='orange', label='now_acc')

    plt.xlabel('Sessions')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper right")

    plt.grid()
    plt.savefig(f'{args.save_path}/{args.dataset}.jpg')
    plt.show()
