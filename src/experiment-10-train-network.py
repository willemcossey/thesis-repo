from src.helper.NeuralNet import NeuralNet
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from src.helper.Dataset import Dataset

# import dataset

data = Dataset.from_json("src/datasets/7d0ca7a38db3c6cf84efa7bfa36e8a7e.json")

n_samples = 1000

x = torch.from_numpy(data.get_inputs(end=n_samples)).to(torch.float)
y = torch.from_numpy(data.get_outputs(end=n_samples, type="aggregated")).to(torch.float)

#%%

hyperparameters_configurations = {
    "hidden_layers": [1],
    "neurons": [10, 20],
    "regularization_exp": [0],
    "regularization_param": [0],
    "batch_size": [100],
    "epochs": [100, 1000, 5000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567, 134, 124],
    "activation": ["tanh"],
}
#
# hyperparameters_configurations = {
#         "hidden_layers": [2],
#         "neurons": [30],
#         "regularization_exp": [0],
#         "regularization_param": [0],
#         "batch_size": [100],
#         "epochs": [500],
#         "optimizer": ["LBFGS"],
#         "init_weight_seed": [567],
#         "activation": ["tanh"]
#     }

settings = list(itertools.product(*hyperparameters_configurations.values()))
print(len(settings))

#%%

i = 0

train_err_conf = list()
val_err_conf = list()
test_err_conf = list()
for set_num, setup in enumerate(settings):
    print(
        "###################################",
        set_num,
        "###################################",
    )
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "regularization_exp": setup[2],
        "regularization_param": setup[3],
        "batch_size": setup[4],
        "epochs": setup[5],
        "optimizer": setup[6],
        "init_weight_seed": setup[7],
        "activation": setup[8],
    }

    (
        relative_error_train_,
        relative_error_val_,
        relative_error_test_,
    ) = NeuralNet.train_single_configuration(setup_properties, x, y)
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)
    test_err_conf.append(relative_error_test_)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
test_err_conf = np.array(test_err_conf)

#     plt.figure(figsize=(16, 8))
#     plt.grid(True, which="both", ls=":")
#     plt.scatter(train_err_conf, test_err_conf, marker="*", label="Training Error")
#     plt.scatter(val_err_conf, test_err_conf, label="Validation Error")
#     plt.scatter(test_err_conf, test_err_conf, marker="v", label="Generalization Error")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel(r'Selection Criterion - Log Scale')
#     plt.ylabel(r'Generalization Error - Log Scale')
#     # plt.title(r'Validation - Training Error VS Generalization error ($\sigma=$' + str(sigma) + r')')
#     plt.legend()
#     # plt.savefig("./sigma_" + str(sigma) + ".png", dpi=4001)
#
#
#  #%%
# plt.show()


# Exercise: for each hyper-parameter (e.g hidden_layers, neurons, etc.),
# plot the conditional distribution of the generalization error given all possible realizations of the chosen hyperparameter.
