# Python 3.8.5


import UTKFace
import AttackDataset
import Target
import Attacker
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy
import argparse
import random
import torch.nn as nn
import torch.autograd as autograd
import pickle


class AttributeInferenceAttack:

    def __init__(self, target, load, device, ds_root, params):
        self.target = target.to(device)
        self.load = load
        self.device = device
        self.ds_root = ds_root
        self.params = params

    def _load_dataset(self):
        transform = transforms.Compose(
                            [ transforms.Resize(size=256),
                              transforms.CenterCrop(size=224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.dataset = AttackDataset.AttackDataset(root='AttackDataset', test=True, transform=transform)
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.params['batch_size'], num_workers=1)
        return dataloader

    def _process_raw_data(self, loader):
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        self.target.eval()

        with torch.no_grad():
            for l, (x, y) in enumerate(loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                logits = self.target(x)

                for i, logit in enumerate(logits):
                    idx = random.randint(0, 100)
                    d = (list(logit.numpy()), int(y[i]))
                    if idx < 70:
                        self.train_data.append(d)
                    elif idx < 80:
                        self.eval_data.append(d)
                    else:
                        self.test_data.append(d)

    def save_data(self):
        with open('Attacker/train_data.txt', 'wb') as trd:
            pickle.dump(self.train_data, trd)

        with open('Attacker/eval_data.txt', 'wb') as evd:
            pickle.dump(self.eval_data, evd)

        with open('Attacker/test_data.txt', 'wb') as ted:
            pickle.dump(self.test_data, ted)

        self.load_data()

    def load_data(self):
        with open('Attacker/train_data.txt', 'rb') as trd:
            self.train_data = pickle.load(trd)

        with open('Attacker/eval_data.txt', 'rb') as evd:
            self.eval_data = pickle.load(evd)

        with open('Attacker/test_data.txt', 'rb') as ted:
            self.test_data = pickle.load(ted)

        self.train_dl = DataLoader(self.train_data, batch_size=64)
        self.eval_dl = DataLoader(self.train_data, batch_size=64)
        self.test_dl = DataLoader(self.train_data, batch_size=64)

    def _train_model(self):
        loss_fn = F.cross_entropy
        for epoch in range(self.params['epochs']):
            self.model.train()

            for inputs, labels in self.train_dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # forward
                output = self.model(inputs)
                output = output.view(-1, 2)
                loss = loss_fn(output, labels)

                # backward + optimization
                loss.backward()
                self.optimizer.step()

            acc = self.evaluate(self.eval_dl)
            print(f'\tEpoch {epoch + 1}/{self.params["epochs"]} : {acc}')

    def evaluate(self, data):
        num_correct = 0
        num_samples = 0

        model.eval()
        with torch.no_grad():
            for x, y in data:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                logits = self.model(x)
                _, preds = torch.max(logits, dim=1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        return float(num_correct) / float(num_samples)

    def run(self):
        if self.load:
            # load saved dataset
            print(f' [+] Load AttackDataset')
            self.load_data()
        else:
            # process raw dataset
            print(f' [+] Process AttackDataset')
            dataloader = self._load_dataset()
            self._process_raw_data(dataloader)
            self.save_data()

        # create attacker model
        print(f' [+] Create Attack Model (MLP)')
        self.model = Attacker.MLP(self.params['feat_amount'],       # feature amount (White, Black, Asian, Indian, Others)
                                  self.params['num_hnodes'],        # hidden nodes
                                  self.params['num_classes'],       # num classes
                                  self.params['activation_fn'],     # activation function
                                  self.params['dropout'])           # dropout
        self.model.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # train attacker model
        print(f' [+] Train Attack Model for {self.params["epochs"]} epochs')
        self._train_model()

        return self.evaluate(self.test_dl)


# cmd args
parser = argparse.ArgumentParser(description='Link-Stealing Attack')

parser.add_argument("--train",
                    action="store_true",
                    help="Train Target Model")

parser.add_argument("--device",
                    default="cpu",
                    help="Device for calculations")

parser.add_argument("--load",
                    action="store_true",
                    help="Load saved Attacker Dataset")

args = parser.parse_args()

# load target model
if args.train:
    target = Target.Target(device=args.device, train=True, ds_root='UTKFace')
else:
    target = Target.Target(device=args.device)
# run attack
parames = {'epochs': 200,
          'lr': 0.001,
          'batch_size': 1,
          'feat_amount': 5,
          'num_hnodes': 128,
          'num_classes': 2,
          'activation_fn': nn.ReLU(),
          'dropout': 0.5}

attack = AttributeInferenceAttack(target.model,
                                  args.load,
                                  device=args.device,
                                  ds_root='AttackerDataset',
                                  params=parames)
acc = attack.run()
print(f'Attribute Inference Attack: {acc}')
