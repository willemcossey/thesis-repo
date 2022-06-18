from helper.NeuralNet import NeuralNet
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from helper.Dataset import Dataset
import winsound
from os import path
from torch.utils.data import DataLoader
import time
import matplotlib as mpl

# import dataset

mpl.style.use(path.join("src", "grayscale_adjusted.mplstyle"))

# dataset_name = "8aef15d64f53b52a4735756a4fe868bf.json" #resolution: 16384
dataset_name = "e3513ee46f1829cd90d7e07973b5e4f1.json"  # resolution: 512
data = Dataset.from_json(path.join("src", "datasets", dataset_name), lazy=True)

# data.compute_aggregated_output(20)

n_samples = 64
#%%
x = torch.from_numpy(data.get_inputs(end=n_samples, silent=False)).to(torch.float)
y = torch.from_numpy(
    data.get_outputs(end=n_samples, otype="aggregated", silent=False)
).to(torch.float)
print(y[1, :])


input_dim = x.shape[1]
output_dim = y.shape[1]
#%%

transform_input = lambda x: (x - torch.Tensor([6, 0])) / torch.Tensor([12, 2])

x = transform_input(x)

#%%
visual = None

hyperparameters_configurations = {
    "hidden_layers": [1, 2, 3],
    "neurons": [100, 200, 300],
    "regularization_exp": [2],
    "regularization_param": [0, 1e-4],
    "batch_size": [n_samples],
    "epochs": [1000, 2000, 4000],
    "optimizer": ["ADAM"],
    "init_weight_seed": [567, 134, 124],
    "activation": ["tanh"],
    "add_sftmax_layer": [False],
}
visual = False
# n tr 64 r tr 512 - 27.29%

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

# hyperparameters_configurations = {
#     "hidden_layers": [1, 2, 3],
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
# 1 1000 samples 16384 particles - l 3 200 n 23.0% analyt test

# hyperparameters_configurations = {
#     "hidden_layers": [3, 4],
#     "neurons": [200,300,400],
#     "regularization_exp": [2],
#     "regularization_param": [0,1e-4],
#     "batch_size": [n_samples],
#     "epochs": [6000, 8000, 12000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [567],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False
# 2 1000 samples 16384 particles - 21.35% l4 n 200 e 6000 reg 1e-4-2

# hyperparameters_configurations = {
#     "hidden_layers": [2],
#     "neurons": [200],
#     "regularization_exp": [2],
#     "regularization_param": [0, 1e-4],
#     "batch_size": [n_samples],
#     "epochs": [4000],
#     "optimizer": ["ADAM"],
#     "init_weight_seed": [568],
#     "activation": ["tanh"],
#     "add_sftmax_layer": [False],
# }
# visual = False

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

best_model_ind = np.argmin(test_err_conf)
best_model = settings[best_model_ind]
print(best_model)


model = NeuralNet(
    input_dimension=input_dim,
    output_dimension=output_dim,
    n_hidden_layers=best_model[0],
    neurons=best_model[1],
    regularization_param=best_model[3],
    regularization_exp=best_model[2],
    retrain_seed=best_model[7],
    activation_name=best_model[8],
    add_sftmax_layer=best_model[9],
)

optimizer_ = model.get_optimizer(best_model[6])

n_epochs = best_model[5]

batch_size = best_model[4]

start_time = time.time()

training_set = DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=best_model[4], shuffle=True
)
history = NeuralNet.fit(model, training_set, n_epochs, optimizer_, p=2, verbose=False)

training_time = time.time() - start_time

plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, n_epochs + 1), np.log10(history[0]), label="Train Loss")
plt.legend()

model_filename = f"nn-in-{input_dim}-out-{output_dim}-hid-{best_model[0]}-n-{best_model[1]}-activ-{best_model[8]}-regul-{best_model[3]}-{best_model[2]}-soft-{best_model[9]}-rng-{best_model[7]}-data-{dataset_name}-resol-{data.meta['experiment_assumptions']['n_samples']}-n_tr-{n_samples}-opt-{best_model[6]}-ep-{best_model[5]}-batch-{best_model[4]}-tr_time-{int(training_time)}.pt"

#%%
surrogate_model = model


# Load testing data

#%%
x_test, y_test = NeuralNet.get_test_samples(10, y.shape[1])

# Make prediction
# test_pred = surrogate_model.forward(
#     transform_input(
#         torch.tensor(
#             [
#                 [1, 0],
#             ]
#         )
#     )
# ).detach()

test_pred = surrogate_model.forward(transform_input(x_test)).detach()

h = 2 / output_dim
centers = [-1 + h / 2 + i * h for i in range(output_dim)]
for i in range(x_test.shape[0]):
    f = plt.figure()
    plt.plot(centers, test_pred[i, :], label="neural network prediction")
    plt.plot(centers, y_test[i, :], label="simulation result")
    plt.legend()
    plt.xlabel("Opinion $w$")
    plt.ylabel("$P_T(w)$")
    plt.title(f"$\lambda$ = {x_test[i,0]:.2f} $m$ = {x_test[i,1]:.2f}")
    plt.show(block=False)
    prediction_name_str = f"experiment-10-pred-result-{model_filename}-input-{x_test[i,0]:.2f}-{x_test[i,1]:.2f}.eps"
    f.savefig(path.join("src", "experiment-data", prediction_name_str))


#%% save and load model and make prediction again

# print(surrogate_model.state_dict())
models_folder = path.join("src", "models")
#%%
torch.save(surrogate_model, path.join(models_folder, model_filename))

#%%

# model_filename = "nn-in-2-out-20-hid-2-n-300-activ-tanh-regul-0.0001-2-soft-False-rng-124-data-e3513ee46f1829cd90d7e07973b5e4f1.json-resol-512-n_tr-64-opt-ADAM-ep-4000-batch-64-tr_time-15.pt"
model_filename = "nn-in-2-out-20-hid-4-n-200-activ-tanh-regul-0.0001-2-soft-False-rng-567-data-8aef15d64f53b52a4735756a4fe868bf.json-resol-16384-n_tr-1000-opt-ADAM-ep-6000-batch-1000-tr_time-148.pt"

loaded_model = torch.load(
    path.join(
        models_folder,
        model_filename,
    )
)
loaded_model = torch.load(path.join(models_folder, model_filename))
#%%
loaded_inference = loaded_model.forward(transform_input(x_test))
print(torch.all(loaded_inference == test_pred))
