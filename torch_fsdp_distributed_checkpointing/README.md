# Torch FSDP Example with Distributed Checkpointing

This example walks through how to run model training with `torch.distributed.fsdp`,
while saving sharded checkpoints on each worker rather than gathering model/optimizer
state on the rank 0 worker. This reduces the communication overhead needed to checkpoint,
and allows workers to upload their sharded state independently to some
persistent storage (cloud, NFS).

## Dependencies

Install the requirements in `requirements.txt` first.

## Compute

This example was run on a Ray cluster with 2 `g4dn.4xlarge` nodes.

## TODO

Restoration on failure retry doesn't work as of Ray 2.5.0.
