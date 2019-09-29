import torch
import torch.nn as nn


class Transformer(object):
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.transformer_parameters(),
                                          lr=args.pruner_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.pruner_weight_decay)
        self.baseline = 0
        self.gamma = args.gamma

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        loss, reward, pruned_accuracy, normal_ent, reduce_ent = self.model._loss_pruner(input_valid, target_valid, self.baseline)
        loss.backward()
        nn.utils.clip_grad_norm(self.model.transformer_parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.update_baseline(reward)
        return reward, pruned_accuracy, normal_ent, reduce_ent




