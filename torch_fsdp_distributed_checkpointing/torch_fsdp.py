# Adapted from ray:
# https://docs.ray.io/en/latest/train/examples/pytorch/torch_fashion_mnist_example.html
# The main modifications are wrapped in comments.

# See here for helpful FSDP references: https://pytorch.org/docs/stable/fsdp.html

from typing import Dict
from ray import air, train
from ray.air import session
import tempfile
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.air.config import ScalingConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="/tmp/fashion_mnist_data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="/tmp/fashion_mnist_data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // session.get_world_size()
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=worker_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=worker_batch_size)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model.
    model = NeuralNetwork()
    ## MODIFICATION 1: PARALLEL_STRATEGY -> FSDP
    model = train.torch.prepare_model(model, parallel_strategy="fsdp")
    ## ----------------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start_epoch = 1

    checkpoint = session.get_checkpoint()
    if checkpoint:
        # TODO: restoring doesn't really work at the moment.
        with checkpoint.as_directory() as ckpt_dir:
            print(
                "\nRank {session.get_world_rank()} checkpoint dir:",
                os.listdir(ckpt_dir),
            )
            state_dict = torch.load(
                os.path.join(ckpt_dir, f"model-{session.get_world_rank()}.pt")
            )
            optim_state_dict = torch.load(
                os.path.join(ckpt_dir, f"optim-{session.get_world_rank()}.pt")
            )
            state = torch.load(os.path.join(ckpt_dir, f"extra.pt"))

            start_epoch = state["epoch"] + 1

            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
            ):
                model.load_state_dict(state_dict)
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    optim_state_dict, model, optimizer
                )
                optimizer.load_state_dict(optim_state_dict)

            state_dict, optim_state_dict = None, None

    for epoch in range(start_epoch, epochs + 1):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        loss = validate_epoch(test_dataloader, model, loss_fn)

        ## MODIFICATION 2: Save a sharded state dict instead of gathering then saving.
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        ):
            state_dict = model.state_dict()
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            ## MAKE SURE TO SAVE EACH SHARD UNDER A UNIQUE PATH.
            ## E.g. Use the worker rank to construct the file paths.
            torch.save(
                state_dict, os.path.join(tmpdir, f"model-{session.get_world_rank()}.pt")
            )
            torch.save(
                optim_state_dict,
                os.path.join(tmpdir, f"optim-{session.get_world_rank()}.pt"),
            )
            torch.save({"epoch": epoch}, "extra.pt")

            state_dict, optim_state_dict = None, None

            checkpoint = TorchCheckpoint.from_directory(tmpdir)
            session.report(dict(loss=loss), checkpoint=checkpoint)
        ## ----------------------------------------------------------------


def train_fashion_mnist():
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 3e-4, "batch_size": 64, "epochs": 10},
        scaling_config=ScalingConfig(
            num_workers=2, use_gpu=True, resources_per_worker={"CPU": 15.0, "GPU": 1.0}
        ),
        run_config=air.RunConfig(
            ## MODIFICATION 3: Use an s3 bucket as the "shared filesystem"
            storage_path="s3://your-s3-bucket",
            name="TorchTrainer_fsdp_distributed_checkpointing",
            ## MODIFICATION 4: Configure AIR to upload from workers directly.
            checkpoint_config=air.CheckpointConfig(
                _checkpoint_keep_all_ranks=True,
                _checkpoint_upload_from_workers=True,
            ),
            # failure_config=air.FailureConfig(max_failures=1),
        ),
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")


if __name__ == "__main__":
    train_fashion_mnist()
