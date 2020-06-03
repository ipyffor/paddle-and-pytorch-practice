import torch
from torch import optim
from net import Resnet18
import os
# import numpy as np

class Trainer():
    def __init__(self, lr, device=torch.device('cpu'), model_path = None):
        self.model_path = model_path
        self.model = Resnet18().to(device)
        self.optim = optim.SGD(self.model.parameters(),lr, momentum=0.9, weight_decay=5e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        pass
    def pred_process(self):
        pass
    def loss_acc_fn(self, images, labels):
        self.logits = self.model(images)
        self.loss = self.loss_fn(self.logits, labels)
        pass
    def forward(self, images, labels, is_eval = False):
        if is_eval:
            self.model.eval()
            with torch.no_grad():
                self.loss_acc_fn(images, labels)
            self.model.train()
            return self.loss, self.logits
        else:
            self.loss_acc_fn(images, labels)
            return self.loss, self.logits
        pass
    def train_update(self):
        self.loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        pass
    def save_model(self, fname = None):
        assert self.model_path is not None
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        fname = 'resnet18.pth' if fname is None else fname
        torch.save(self.model.state_dict(), '{}/{}'.format(self.model_path, fname))
        pass
    def load_model(self, fname = None):
        assert self.model_path is not None
        fname = 'resnet18.pth' if fname is None else fname
        dict = torch.load('{}/{}'.format(self.model_path, fname))
        self.model.load_state_dict(dict)
        pass
    def set_model_path(self, path):
        self.model_path = path


if __name__ == '__main__':
    torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))