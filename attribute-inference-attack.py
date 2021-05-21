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
        self.loss_fn = params['loss_fn']


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
                logits = self.target.get_last_hidden_layer(x)

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
        with open(f'Attacker/train_data-{self.params["feat_amount"]}.txt', 'wb') as trd:
            pickle.dump(self.train_data, trd)

        with open(f'Attacker/eval_data-{self.params["feat_amount"]}.txt', 'wb') as evd:
            pickle.dump(self.eval_data, evd)

        with open(f'Attacker/test_data-{self.params["feat_amount"]}.txt', 'wb') as ted:
            pickle.dump(self.test_data, ted)

        self.load_data()

    def load_data(self):
        with open(f'Attacker/train_data-{self.params["feat_amount"]}.txt', 'rb') as trd:
            self.train_data = self.get_data(pickle.load(trd))

        with open(f'Attacker/eval_data-{self.params["feat_amount"]}.txt', 'rb') as evd:
            self.eval_data = self.get_data(pickle.load(evd))

        with open(f'Attacker/test_data-{self.params["feat_amount"]}.txt', 'rb') as ted:
            self.test_data = self.get_data(pickle.load(ted))

        self.train_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])
        self.eval_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])
        self.test_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])

    def get_data(self, list):
        data = []
        for input, label in list:
            input = torch.FloatTensor(input)
            data.append([input, label])
        return data

    def _train_model(self):
        for epoch in range(1, self.params['epochs'] + 1):
            self.model.train()

            for inputs, labels in self.train_dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                # backward + optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                acc = self.evaluate(self.eval_dl)
                print(f'\tEpoch {epoch}/{self.params["epochs"]} : {acc}')

    def evaluate(self, data):
        num_correct = 0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in data:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                logits = self.model(x)
                _, preds = torch.max(logits, dim=1)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        return float(num_correct) / float(num_samples)

    def _save_model(self):
        torch.save(self.model.state_dict(), f'Attacker/attack-{1 if self.params["feat_amount"] == 2 else 2}.pt')

    def run(self):
        if self.load:
            # load saved dataset
            print(f'\n [+] Process AttackDataset')
            self.load_data()
        else:
            # process raw dataset
            print(f'\n [+] Load AttackDataset')
            dataloader = self._load_dataset()
            self._process_raw_data(dataloader)
            self.save_data()

        # create attacker model
        print(f' [+] Create Attack Model (MLP)')
        self.model = Attacker.MLP(self.params['feat_amount'],       # feature amount
                                  self.params['num_hnodes'],        # hidden nodes
                                  self.params['num_classes'],       # num classes
                                  self.params['activation_fn'],     # activation function
                                  self.params['dropout'])           # dropout
        self.model.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # train attacker model
        print(f' [+] Run Attack {1 if self.params["feat_amount"] == 2 else 2}')
        self._train_model()
        self._save_model()

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

# Attack 1
parames1 = {'epochs': 1500,
            'lr': 0.01,
            'batch_size': 64,
            'feat_amount': 2,
            'num_hnodes': 16,
            'num_classes': 5,
            'activation_fn': nn.Sigmoid(),
            'loss_fn': F.cross_entropy,
            'dropout': 0}

attack1 = AttributeInferenceAttack(target.model,
                                  args.load,
                                  device=args.device,
                                  ds_root='AttackerDataset',
                                  params=parames1)
acc1 = attack1.run()

# Attack 2
parames2 = {'epochs': 1500,
            'lr': 0.01,
            'batch_size': 64,
            'feat_amount': 256,
            'num_hnodes': 16,
            'num_classes': 5,
            'activation_fn': nn.Sigmoid(),
            'loss_fn': F.cross_entropy,
            'dropout': 0}

attack2 = AttributeInferenceAttack(target.model,
                                  args.load,
                                  device=args.device,
                                  ds_root='AttackerDataset',
                                  params=parames2)
acc2 = attack2.run()
print(f'\n [  Target  ] Gender Prediction: 0.8863 acc.\n')
print(f' [ Baseline ] Guessing: 0.20 acc.')
print(f' [ Attack 1 ] Attribute Inference Attack - Race: {acc1:0.4f} acc.')
print(f' [ Attack 2 ] Attribute Inference Attack - Race: {acc2:0.4f} acc.\n')
