import argparse
from model.model import LitResnet
import lightning as L
from torchvision.datasets import CIFAR10
import os
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split

parser = argparse.ArgumentParser(description='Fine-tuning on CIFAR10 using the model of resnet34 with pytorch-lightning')

parser.add_argument('--batch',  '-b',
                    dest="limit_batch",
                    type=int,
                    help = 'batch limit on training',
                    default=100)

parser.add_argument('--check_val_e',  '-ch',
                    dest="check_val_every_n_epoch",
                    type=int,
                    help = 'val on training',
                    default=5)

parser.add_argument('--max_e',  '-e',
                    dest="max_epochs",
                    type=int,
                    help = 'number of epochs',
                    default=100)

parser.add_argument('--output_ckp',  '-o',
                    dest="root_dir",
                    type=str,
                    help = 'saving checkpoint',
                    default="./checkpoints/")

args = parser.parse_args()


model = LitResnet()

trainer = L.Trainer(limit_train_batches=args.limit_batch, max_epochs=args.max_epochs, check_val_every_n_epoch=args.max_epochs, logger=True, default_root_dir=args.root_dir) 

train_dataset = CIFAR10(os.getcwd(),  train=True, download=True)
DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD),
    ]
)

dataset_train = CIFAR10(root=os.getcwd(), train=True, transform=train_transform, download=False)
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(DATA_MEANS,DATA_STD)]) ##Normalize data
dataset_val = CIFAR10(os.getcwd(), train=True, transform=test_transform)

L.seed_everything(42)
train_set, _ = random_split(dataset_train, [45000, 5000])
L.seed_everything(42)
_, val_set = random_split(dataset_val, [45000, 5000])


train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)


trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader) ## Needed of DataLoaders here
