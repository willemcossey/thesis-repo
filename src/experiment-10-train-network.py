from helper.NeuralNet import NeuralNet
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from helper.Dataset import Dataset
import winsound
from os import path

# import dataset

# dataset_name = "7d0ca7a38db3c6cf84efa7bfa36e8a7e.json"
dataset_name = "e3513ee46f1829cd90d7e07973b5e4f1.json"
data = Dataset.from_json(path.join("src", "datasets", dataset_name), lazy=True)

# data.compute_aggregated_output(20)

n_samples = 64
#%%
x = torch.from_numpy(data.get_inputs(end=n_samples, silent=False)).to(torch.float)
y = torch.from_numpy(
    data.get_outputs(end=n_samples, otype="aggregated", silent=False)
).to(torch.float)
print(y[1, :])


#%%

x = (x - torch.Tensor([6, 0])) / torch.Tensor([12, 2])

#%%
visual = None

# hyperparameters_configurations = {
#     "hidden_layers": [1],
#     "neurons": [10, 20, 40],
#     "regularization_exp": [2],
#     "regularization_param": [1e-4],
#     "batch_size": [100],
#     "epochs": [50, 100, 150, 200, 500],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567, 134, 124],
#     "activation": ["tanh"],
#     "add_softmax":["False"],
# }

# linear output
# #1 1000samples 10 000 particles - max dimensions

# hyperparameters_configurations = {
#     "hidden_layers": [1],
#     "neurons": [40, 60, 100],
#     "regularization_exp": [2],
#     "regularization_param": [0, 1e-4],
#     "batch_size": [100],
#     "epochs": [200, 500, 1000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567, 134, 124],
#     "activation": ["tanh"],
#     "add_softmax": ["False"],
# }
# #2 1000samples 10 000 particles - max dimensions, no regularization

# hyperparameters_configurations = {
#     "hidden_layers": [1],
#     "neurons": [100],
#     "regularization_exp": [2],
#     "regularization_param": [0, 1e-4],
#     "batch_size": [100],
#     "epochs": [1000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567, 134, 124],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [True],
# }
# visual = True
# #3 No 'learning' happening for the moment

# hyperparameters_configurations = {
#     "hidden_layers": [1],
#     "neurons": [100, 200, 400],
#     "regularization_exp": [2],
#     "regularization_param": [0],
#     "batch_size": [n_samples],
#     "epochs": [2000, 4000, 8000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_softmax": [False],
# }
# #4 1000samples 10 000 particles- n 100, e 8000 - 19.72% test error

# hyperparameters_configurations = {
#     "hidden_layers": [1,2,3],
#     "neurons": [50, 100, 150],
#     "regularization_exp": [2],
#     "regularization_param": [0],
#     "batch_size": [n_samples],
#     "epochs": [2000, 4000, 8000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False
# 5 1000samples 10 000 particles - 1 layers, 100 neurons, 8000 epochs - 19.72% test error

# hyperparameters_configurations = {
#     "hidden_layers": [1,2,3],
#     "neurons": [50, 100, 150],
#     "regularization_exp": [2],
#     "regularization_param": [0],
#     "batch_size": [n_samples],
#     "epochs": [2000, 4000, 8000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False
# 1 64 samples 512 particles - 1l 100n 4000e 28.838% (runner up:  3l 150 n 2000 ep 29.1%)

# hyperparameters_configurations = {
#     "hidden_layers": [1,2,3],
#     "neurons": [50, 100, 150],
#     "regularization_exp": [2],
#     "regularization_param": [1e-4],
#     "batch_size": [n_samples],
#     "epochs": [2000, 4000, 8000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False
# 2 64 samples 512 particles - 3l 150n 8000 e

hyperparameters_configurations = {
    "hidden_layers": [1, 2, 3],
    "neurons": [50, 100, 150],
    "regularization_exp": [2],
    "regularization_param": [1e-4],
    "batch_size": [n_samples],
    "epochs": [2000, 4000, 8000],
    "optimizer": ["ADAM"],
    "init_weight_seed": [567],
    "activation": ["tanh"],
    "add_sftmax_layer": [False],
}
visual = False
# 2 64 samples 512 particles -

# hyperparameters_configurations = {
#     "hidden_layers": [2, 3, 4],
#     "neurons": [200],
#     "regularization_exp": [2],
#     "regularization_param": [0],
#     "batch_size": [n_samples],
#     "epochs": [4000, 6000, 8000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False
# 6 - 3 layers, 8000 epochs - 12.6% test error
# 6 - analytical test - 2l, 4000 - 21.8% t, 13 tr, 14 v

# hyperparameters_configurations = {
#     "hidden_layers": [2],
#     "neurons": [200],
#     "regularization_exp": [0],
#     "regularization_param": [0],
#     "batch_size": [n_samples],
#     "epochs": [4000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [568],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = True

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
        "add_sftmax_layer": setup[9],
    }

    (
        relative_error_train_,
        relative_error_val_,
        relative_error_test_,
    ) = NeuralNet.train_single_configuration(
        setup_properties, x, y, visual_check=visual
    )
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)
    test_err_conf.append(relative_error_test_)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
test_err_conf = np.array(test_err_conf)

frequency = 440  # Set Frequency To 2500 Hertz
duration = 2000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

#%%

fig = plt.figure()
horiz_var_ind = 1
for i in range(len(settings)):
    plt.scatter(i, train_err_conf[i], marker="*", color="blue", label="Training Error")
    plt.scatter(i, val_err_conf[i], color="green", label="Validation Error")
    plt.scatter(
        i, test_err_conf[i], marker="v", color="red", label="Generalization Error"
    )
plt.yscale("log")
plt.xlabel(r"Configuration Index")
plt.ylabel(r"Errors - Log Scale")
# plt.title(r'Validation - Training Error VS Generalization error ($\sigma=$' + str(sigma) + r')')
plt.legend(["Training", "Validation", "Generalization"])
plt.grid(visible=True, which="both", axis="both")
# plt.savefig("./sigma_" + str(sigma) + ".png", dpi=4001)
plt.show(block=True)

#%% Train and save best model
