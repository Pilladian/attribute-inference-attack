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


class AttributeInferenceAttack:

    def __init__(self, target, device, ds_root, epochs, batch_size):
        self.target = target
        self.device = device.to(device)
        self.ds_root = ds_root
        self.epochs = epochs
        self.batch_size = batch_size

    def _load_dataset(self):
        transform = transforms.Compose(
                            [ transforms.Resize(size=256),
                              transforms.CenterCrop(size=224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

        self.dataset = AttackDataset.AttackDataset(root='AttackDataset', test=True, transform=transform)
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=1)

        return dataloader

    def _process_raw_data(self, loader):
        size = self.dataset.__len__()
        train_val = 0.7
        test_val = 0.1
        self.features = torch.zeros((self.dataset.__len__() + 1, 2), dtype=torch.float)
        self.labels = torch.zeros((self.dataset.__len__() + 1), dtype=torch.long)
        with torch.no_grad():
            for l, (x, y) in enumerate(loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                logits = target.model(x)

                for i, logit in enumerate(logits):
                    self.features[(l * self.batch_size) + i] = logit
                    label = torch.ones(1, 1)
                    label[0][0] = int(y[i])
                    self.labels[(l * self.batch_size) + i] = label

        # train, eval, test
        train_mask = torch.zeros(size, dtype=torch.bool)
        val_mask = torch.zeros(size, dtype=torch.bool)
        test_mask = torch.zeros(size, dtype=torch.bool)

        train_val_split = int(size * train_val)
        val_test_split  = train_val_split + int(size * test_val)

        # train masks
        for a in range(size):
            train_mask[a] = a < train_val_split
            val_mask[a] = a >= train_val_split and a < val_test_split
            test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = train_mask.nonzero().squeeze()
        self.val_nid = val_mask.nonzero().squeeze()
        self.test_nid = test_mask.nonzero().squeeze()

    def _train_model(self):
        for epoch in range(self.epochs):
            self.model.train()

            # forward
            logits = self.model(self.features)
            loss = F.cross_entropy(logits[self.train_nid], self.labels[self.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # evaluate
            acc = self.evaluate(self.val_nid)
            print(f'\tEpoch {epoch + 1}/{self.epochs} : {acc}')

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(self.features)
            logits = logits[nid]
            labels = self.labels[nid]
            _, pred = torch.max(logits, dim=1)

            return torch.sum(pred == labels).item() * 1.0 / len(labels)

    def run(self):
        # load dataset
        print(f' [+] Load AttackDataset')
        dataloader = self._load_dataset()
        # process raw dataset
        print(f' [+] Process AttackDataset')
        self._process_raw_data(dataloader)

        # create attacker model
        print(f' [+] Create Attack Model (MLP)')
        self.model = Attacker.MLP(2,         # feature amount
                                  16,        # hidden nodes
                                  2,         # num classes (White, Black, Asian, Indian, Others)
                                  2,         # hidden layer
                                  F.relu,    # activation function
                                  0.5)       # dropout
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # train attacker model
        print(f' [+] Train Attack Model for {self.epochs} epochs')
        self._train_model()

        return self.evaluate(self.test_nid)


# cmd args
parser = argparse.ArgumentParser(description='Link-Stealing Attack')

parser.add_argument("--train",
                    action="store_true",
                    help="Train Target Model")

parser.add_argument("--device",
                    default="cpu",
                    help="Device for calculations")

args = parser.parse_args()

# load target model
if args.train:
    target = Target.Target(device=args.device, train=True, ds_root='UTKFace')
else:
    target = Target.Target(device=args.device)
# run attack
attack = AttributeInferenceAttack(target.model, device=args.device, ds_root='AttackerDataset', epochs=200, batch_size=64)
acc = attack.run()
print(f'Attribute Inference Attack: {acc}')
