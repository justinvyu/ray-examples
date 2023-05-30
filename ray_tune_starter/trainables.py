
from typing import Dict
from filelock import FileLock
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


from ray.tune import Trainable


def build_dataloaders(batch_size):
    with FileLock("/mnt/cluster_storage/data.lock"):
        data_dir = "/mnt/cluster_storage/fashion_mnist"

        # Download training data from open datasets.
        training_data = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    train_data, valid_data = random_split(training_data, [0.8, 0.2])
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


def train_epoch(dataloader, model, loss_fn, optimizer, device="cpu"):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_epoch(dataloader, model, loss_fn, device="cpu"):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    return test_loss


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 32, 10)  # Output size: 10 (number of classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out



class TrainableA(Trainable):
    def setup(self, config):
        self.model = ModelA()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.batch_size = config["batch_size"]
        self.lr = config["lr"]

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_data, self.test_data = build_dataloaders(batch_size=self.batch_size)

    def reset_config(self, new_config: dict) -> bool:
        self.batch_size = new_config["batch_size"]
        self.lr = new_config["lr"]

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        return True

    def step(self) -> dict:
        train_epoch(self.train_data, self.model, self.loss_fn, self.optimizer, device=self.device)
        valid_loss = validate_epoch(self.train_data, self.model, self.loss_fn, device=self.device)
        return {"validation_loss": valid_loss}

    def save_checkpoint(self, tmp_checkpoint_dir: str) -> str:
        torch.save(self.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(tmp_checkpoint_dir, "optim.pt"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str):
        model_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pt"))
        optim_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "optim.pt"))

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)


class TrainableB(Trainable):
    def setup(self, config):
        self.model = ModelB()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.batch_size = config["batch_size"]
        self.lr = config["lr"]

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_data, self.test_data = build_dataloaders(batch_size=self.batch_size)

    def reset_config(self, new_config: dict) -> bool:
        self.batch_size = new_config["batch_size"]
        self.lr = new_config["lr"]

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        return True

    def step(self) -> dict:
        train_epoch(self.train_data, self.model, self.loss_fn, self.optimizer, device=self.device)
        valid_loss = validate_epoch(self.train_data, self.model, self.loss_fn, device=self.device)
        return {"validation_loss": valid_loss}

    def save_checkpoint(self, tmp_checkpoint_dir: str) -> str:
        torch.save(self.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(tmp_checkpoint_dir, "optim.pt"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str):
        model_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pt"))
        optim_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "optim.pt"))

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)

