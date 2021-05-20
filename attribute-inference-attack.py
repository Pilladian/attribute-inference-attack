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
                logits = self.target(x)

                for i, logit in enumerate(logits):
                    while torch.max(logit) < 1:
                        logit += 10
                    idx = random.randint(0, 100)
                    d = (list(logit.numpy()), int(y[i]))
                    print(d)
                    if random.randint(0, 100) > 80:
                        exit()

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
            self.train_data = self.get_data(pickle.load(trd))

        with open('Attacker/eval_data.txt', 'rb') as evd:
            self.eval_data = self.get_data(pickle.load(evd))

        with open('Attacker/test_data.txt', 'rb') as ted:
            self.test_data = self.get_data(pickle.load(ted))

        self.train_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])
        self.eval_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])
        self.test_dl = DataLoader(self.train_data, batch_size=self.params['batch_size'])

    def get_data(self, list):
        data = []
        for input, label in list:
            input = torch.FloatTensor(input)
            # l = torch.zeros(2)
            # if label == 1:
            #     l[1] = 1
            # else:
            #     l[0] = 1
            data.append([input, label])
        return data

    def _train_model(self):
        for epoch in range(self.params['epochs']):

            for inputs, labels in self.test_data:
                self.model.train(inputs, labels)

            acc = self.evaluate(self.eval_data)
            print(f'\tEpoch {epoch + 1}/{self.params["epochs"]} : {acc}')

    def evaluate(self, data):
        num_correct = 0
        num_samples = 0

        for x, y in data:

            logits = [i[0] for i in self.model.query(x)]
            pred = 1 if logits[0] < logits[1] else 0

            num_correct += 1 if pred == y else 0
            num_samples += 1

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
        self.model = Attacker.NeuralNetwork(5, 16, 2, 0.01)

        # train attacker model
        print(f' [+] Train Attack Model for {self.params["epochs"]} epochs')
        self._train_model()

        return self.evaluate(self.test_data)


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
parames = {'epochs': 50,
          'lr': 0.001,
          'batch_size': 1,
          'feat_amount': 5,
          'num_hnodes': 16,
          'num_classes': 2,
          'activation_fn': nn.Sigmoid(),
          'loss_fn': F.cross_entropy,
          'dropout': 0.1}

attack = AttributeInferenceAttack(target.model,
                                  args.load,
                                  device=args.device,
                                  ds_root='AttackerDataset',
                                  params=parames)
acc = attack.run()
print(f'Attribute Inference Attack: {acc}')
