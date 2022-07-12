import abc
import os

import numpy as np
import torch

from Knowe import Knowe
import torch.nn as nn
from copy import deepcopy

from data import get_base_dataloader, get_new_dataloader
from util import mkdir, save_list_to_txt, base_train, test, plot_fig


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.trlog = {'train_loss': [], 'train_acc': [], 'max_acc_epoch': 0, 'max_acc': [0.0] * args.sessions,
                      'coarse_acc': [0.0] * args.sessions, 'fine_acc': [0.0] * args.sessions,
                      'now_acc': [0.0] * args.sessions}

    @abc.abstractmethod
    def train(self):
        pass


class C2FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.model = Knowe(self.args, norm=self.args.norm)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu))).cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=0.0005)
        return optimizer

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        result_list = [self.args]
        coarse_acc = []
        fine_acc = []
        now_acc = []
        for session in range(0, self.args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)
            if session == 0:  # load base class train img label
                if self.args.dataset == 'cifar100':
                    print('new classes for this session:\n', np.unique(train_set.targets))
                if self.args.dataset == 'tiered':
                    print('new classes for this session:\n', np.unique(train_set.labels))
                if self.args.dataset in ['living17', 'entity13', 'entity30', 'nonliving26']:
                    print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer = self.get_optimizer_base()

                for epoch in range(self.args.epochs_base):
                    if self.args.model_dir is None:
                        base_train(self.model, trainloader, optimizer, self.args)
                    if self.args.saveweights:
                        parameters = [e.cpu() for e in self.model.module.fc.parameters()]
                        torch.save(parameters, os.path.join(self.args.save_path, f'fc_parameters_{session}.pth'))

                    tsl, tsa, tsa_coarse, tsa_fine, tsa_now = test(self.model, testloader, self.args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        self.trlog['coarse_acc'][session] = float('%.3f' % (tsa_coarse * 100))
                        self.trlog['now_acc'][session] = float('%.3f' % (tsa_now * 100))
                        save_model_dir = os.path.join(self.args.save_path,
                                                      'session' + str(session) + f'_epoch{epoch}' + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                coarse_acc.append(self.trlog['max_acc'][session])
                now_acc.append(self.trlog['max_acc'][session])

            else:  # incremental  sessions
                print("training session: [%d]" % session)
                self.model.module.norm = self.args.norm
                self.model.eval()
                if self.args.dataset == 'cifar100':
                    class_list = np.unique(train_set.targets)
                    trainloader.dataset.transform = testloader.dataset.transform
                else:
                    class_list = [x for x in range(train_set.new_class_num)]

                if self.args.decoupled is False:
                    for name, param in self.model.module.named_parameters():
                        if name == 'fc.weight':
                            if self.args.method != 'LwF':
                                param.requires_grad = False
                            else:
                                param.requires_grad = True
                        else:
                            param.requires_grad = True

                if self.model.module.norm is False:
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                                lr=self.args.lr_new,
                                                momentum=0.9, dampening=0,
                                                weight_decay=0.0005)

                    if self.args.method in ['Knowe', 'ANCOR']:
                        self.model.module.pre_update(trainloader, class_list, session)
                    elif self.args.method == 'LwF':
                        pre_model = Knowe(self.args, norm=self.args.norm)
                        pre_model = nn.DataParallel(pre_model, list(range(self.args.num_gpu)))
                        pre_model = pre_model.cuda()
                        pre_model.load_state_dict(self.best_model_dict)
                        for epoch in range(self.args.epochs_new):
                            self.model.module.train_LwF(optimizer, session,
                                                        pre_model,
                                                        trainloader)
                    elif self.args.method == 'ScaIL':
                        self.model.module.train_ScaIL(session)
                    elif self.args.method == 'align':
                        self.model.module.pre_update(trainloader, class_list, session)
                        self.model.module.fc_align(session)
                    elif self.args.method == 'subspace':
                        self.model.module.pre_update(trainloader, class_list, session)
                    else:
                        raise NotImplementedError

                elif self.model.module.norm is True:
                    self.model.module.pre_update(trainloader, class_list, session)

                if self.args.saveweights:
                    parameters = [e.cpu() for e in self.model.module.fc.parameters()]
                    torch.save(parameters, os.path.join(self.args.save_path, f'fc_parameters_{session}.pth'))
                tsl, tsa, ta_coarse, ta_fine, ta_now = test(self.model, testloader, self.args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['coarse_acc'][session] = float('%.3f' % (ta_coarse * 100))
                self.trlog['fine_acc'][session] = float('%.3f' % (ta_fine * 100))
                self.trlog['now_acc'][session] = float('%.3f' % (ta_now * 100))
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                result_list.append(
                    'Session {}, test Acc {:.3f}, loss {}, acc total {}, acc coarse {}, acc fine {}, acc now {}\n'.format(
                        session, self.trlog['max_acc'][session], tsl, tsa, ta_coarse, ta_fine, ta_now))
                coarse_acc.append(float('%.3f' % (ta_coarse * 100)))
                fine_acc.append(float('%.3f' % (ta_fine * 100)))
                now_acc.append(float('%.3f' % (ta_now * 100)))

        result_list.append(f"Base Session Best Epoch {self.trlog['max_acc_epoch']}\n")
        result_list.append(f"total_acc {self.trlog['max_acc']}\n")
        result_list.append(f"coarse_acc {self.trlog['coarse_acc']}\n")
        result_list.append(f"fine_acc {self.trlog['fine_acc']}\n")
        result_list.append(f"now_acc {self.trlog['now_acc']}\n")

        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print(f"total_acc:{self.trlog['max_acc']}")
        print(f"coarse_acc:{self.trlog['coarse_acc']}")
        print(f"fine_acc:{self.trlog['fine_acc']}")
        print(f"now_acc:{self.trlog['now_acc']}")

        plot_fig(self.trlog, self.args)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)

    def set_save_path(self):
        norm = self.args.norm
        self.args.save_path = 'norm_' + f'{norm}'
        self.args.save_path = self.args.save_path + f'/{self.args.dataset}_{self.args.way}way_{self.args.shot}shot_{self.args.query}query_{self.args.sessions}session'
        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        mkdir(self.args.save_path)
