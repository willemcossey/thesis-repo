import pandas as pd
import numpy as np
from helper.Dataset import Dataset
import torch
from helper.NeuralNet import NeuralNet
from os import path
from itertools import chain
from time import time
import subprocess


# after having chosen hyper parameters for certain low data quality and low data quantity -> low variance, high bias regime
# Goal: Check how adding more datapoints and better data quality influences the bias of the neural network.

# Having a simple architecture makes sure that we do not overfit by adding more data and better quality data.


## Specify datasets to be used
dataset_name_list = [
    "4f0d31c5c84e0988848b9fb476927585.json",
    "2d3d0ea6f837d4536fb4742d66cb5de4.json",
]
dataset_list = [
    Dataset.from_json(path.join("src", "datasets", d)) for d in dataset_name_list
]
dataset_resolutions = [
    ds.meta["experiment_assumptions"]["n_samples"] for ds in dataset_list
]
dataset_sizes = [ds.meta["size"] for ds in dataset_list]

dataset_df = pd.DataFrame(
    data=list(zip(dataset_name_list, dataset_list, dataset_resolutions, dataset_sizes)),
    columns=["name", "objects", "resolution", "size"],
)

min_size = np.min(dataset_df["size"])
# Specify number of training samples to be investigated
n_training_list = list(
    np.array(
        64 * np.power(2, np.arange(-3, min(5, np.log2(min_size) - 6) + 1)), dtype=int
    )
)

# Start training loop
train_err_conf = list()
val_err_conf = list()
test_err_conf = list()

## Specify network architecture to be used.

nn_arch = {
    "hidden_layers": 2,
    "neurons": 100,
    "regularization_exp": 2,
    "regularization_param": 0,
    "batch_size": None,
    "epochs": 2,
    "optimizer": "ADAM",
    "init_weight_seed": 42,
    "activation": "tanh",
    "add_sftmax_layer": False,
}


col = [
    list(nn_arch.keys()),
    list(dataset_df.columns),
    ["n train samples", "train err", "valid err", "test err"],
]
col = np.array(list(chain(*col)))
result_df = pd.DataFrame(data=None, columns=col)

# for each dataset
for m in range(len(dataset_df)):
    # for diff amount of training samples
    data = dataset_df.loc[m, "objects"]
    for n in n_training_list:
        print(n)
        x = torch.from_numpy(data.get_inputs(end=n, silent=False)).to(torch.float)
        x = (x - torch.Tensor([6, 0])) / torch.Tensor([12, 2])
        y = torch.from_numpy(
            data.get_outputs(end=n, otype="aggregated", silent=False)
        ).to(torch.float)
        # train specified architecture
        nn_arch["batch_size"] = int(n)
        setup_properties = {
            "hidden_layers": nn_arch["hidden_layers"],
            "neurons": nn_arch["neurons"],
            "regularization_exp": nn_arch["regularization_exp"],
            "regularization_param": nn_arch["regularization_param"],
            "batch_size": nn_arch["batch_size"],
            "epochs": nn_arch["epochs"],
            "optimizer": nn_arch["optimizer"],
            "init_weight_seed": nn_arch["init_weight_seed"],
            "activation": nn_arch["activation"],
            "add_sftmax_layer": nn_arch["add_sftmax_layer"],
        }

        (
            relative_error_train_,
            relative_error_val_,
            relative_error_test_,
        ) = NeuralNet.train_single_configuration(setup_properties, x, y)

        dat = [
            list(nn_arch.values()),
            list(dataset_df.loc[m, :]),
            [n],
            [relative_error_train_],
            [relative_error_val_],
            [relative_error_test_],
        ]
        dat = [np.array(list(chain(*dat)))]

        result_line = pd.DataFrame(data=dat, columns=col)
        result_df = pd.concat([result_df, result_line], axis=0)


print(result_df.columns)

result_df.drop("objects", axis=1, inplace=True)
result_df.reset_index(drop=True, inplace=True)

git_label = subprocess.check_output(["git", "describe"]).strip().decode("utf-8")
max_size = max(dataset_sizes)
min_resolution = min(dataset_resolutions)
max_resolution = max(dataset_resolutions)
name_string = f"experiment-15-size--{min_size}-{max_size}--resolution--{min_resolution}-{max_resolution}--git-{git_label}-time-{time()}.csv"
result_df.to_csv(path.join("src", "experiment-data", name_string))