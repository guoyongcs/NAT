import torch


class Transformer(object):
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.transformer_parameters(),
                                          lr=args.transformer_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.transformer_weight_decay)
        self.baseline = 0
        self.gamma = args.gamma

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        loss, reward, optim_accuracy, normal_ent, reduce_ent = self.model._loss_transformer(input_valid, target_valid, self.baseline)
        loss.backward()
        self.optimizer.step()
        self.update_baseline(reward)
        return optim_accuracy, normal_ent, reduce_ent




