# encoding=utf-8
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

class Optimizer:
    def __init__(self, model_parameters, steps):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        self.decay = 0.75
        self.decay_step = 10000 # 20000 # 50000 # 100000 # 1000
        self.learning_rate = 2e-4
        self.bert_lr = 1e-10 # 1e-5

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=self.learning_rate)
                self.optims.append(optim)

                l = lambda step: self.decay ** (step // self.decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)
            elif name.startswith("bert"):
                optim_bert = AdamW(parameters, self.bert_lr, eps=1e-8)
                self.optims.append(optim_bert)

                scheduler_bert = get_linear_schedule_with_warmup(optim_bert, 0, steps)
                self.schedulers.append(scheduler_bert)

                for group in parameters:
                    for p in group['params']:
                        self.all_params.append(p)
            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res