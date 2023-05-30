import argparse
import datetime
import os
from re import search
from typing import Type

from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler


## Step 1: Import your custom trainables
from trainables import TrainableA, TrainableB


def get_current_timestamp() -> str:
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestamp


def launch_tuning_run(trainable_type: str):
    ## Step 2: Set the search space
    param_space = {
        "lr": tune.uniform(5e-5, 1e-3),
        "batch_size": tune.choice([64, 128, 256, 512]),
    }

    ## Step 3: Assign resources to your Trainable
    trainable_resources_map = {
        # TrainableA doesn't require GPU
        "TrainableA": tune.with_resources(TrainableA, resources={"CPU": 2.0}),
        # TrainableB reuqires GPU
        "TrainableB": tune.with_resources(TrainableB, resources={"CPU": 2.0, "GPU": 0.5}),
    }
    trainable = trainable_resources_map[args.trainable_type]

    ## Step 4: Set up a shared storage location for the experiment
    storage_path = "/mnt/user_storage/"
    exp_dir_name = f"{trainable_type}_tuning_run_{get_current_timestamp()}"

    ## Step 6: Integrate with Weights & Biases
    os.environ["WANDB_API_KEY"] = "<your-wandb-api-key>"

    ## Create a Tuner
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        run_config=air.RunConfig(
            # Set a `checkpoint_frequency` if you want to periodically checkpoint the model.
            # Save the `num_to_keep` latest checkpoints. Comment this out to keep all checkpoints.
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1, num_to_keep=5),

            ## Step 3: Set up a shared storage location for the experiment
            local_dir=storage_path,
            name=exp_dir_name,
            # Disable Tune's syncing logic, since the network filesystem already handles syncing between nodes.
            sync_config=tune.SyncConfig(syncer=None),

            ## Step 4: Set a stopping condition for each trial
            stop=lambda trial_id, result: result["validation_loss"] < 0.4 or result["training_iteration"] >= 50,

            ## Step 6: Integrate with Weights & Biases
            callbacks=[WandbLoggerCallback(project="dreamfold_test", group="tune_multiple_trainables")],
        ),
        tune_config=tune.TuneConfig(
            ## Step 2: Set up the search space + number of samples
            # Each trial will randomly sample the Tune search spaces defined earlier!
            num_samples=12,
            
            ## (Optional) Step 5: Set a custom search algorithm
            # Set the metric / mode that the search will use
            metric="validation_loss",  # custom metric -- see `step` return value in trainables.py
            mode="min",
            # Use Optuna to suggest new trials based on prior trials' performance.
            search_alg=OptunaSearch(),
            # Limit the concurrency to 8 trials at a time, so that the search algorithm gets
            # a chance to fit on some data, so that it can suggest better hyperparameter configs to try next.
            max_concurrent_trials=8,
        )
    )

    results: tune.ResultGrid = tuner.fit()

    # Analyze results!
    print(f"Run with {trainable_type} finished! Find the results at:\n", os.path.join(storage_path, exp_dir_name))

    best_result = results.get_best_result(metric="validation_loss", mode="min")
    print("Best result hyperparameter config:\n", best_result.config)
    print("Best result metrics:\n", best_result.metrics)
    print("Best result checkpoint path:\n", best_result.checkpoint.path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainable_type", type=str)
    args = parser.parse_args()
    
    launch_tuning_run(args.trainable_type)

