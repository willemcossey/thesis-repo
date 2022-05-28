import torch
from torch import nn, optim, utils
from torch.utils.data import DataLoader
import numpy as np
from math import floor, ceil, exp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import markers
from helper.Distribution import Uniform


# Based on code provided for the course "Deep Learning and Scientific Computing" at ETH Spring Semester 2022.
# Thanks to Roberto Molinaro for this part of the code.
# https://gitlab.ethz.ch/mroberto
# https://math.ethz.ch/sam/the-institute/people.html?u=mroberto
# https://www.linkedin.com/in/roberto-molinaro-16806b145


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        regularization_param,
        regularization_exp,
        retrain_seed,
        activation_name,
        add_sftmax_layer,
    ):

        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation_name = activation_name
        self.activation = self.get_activation(activation_name)
        # Parameter denoting whether to add Sftmax regularization
        self.add_sftmax_layer = add_sftmax_layer
        # Regularization parameter
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed

        if self.n_hidden_layers != 0:
            self.input_layer = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(self.neurons, self.neurons)
                    for _ in range(n_hidden_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(self.neurons, self.output_dimension)
            if self.add_sftmax_layer:
                print("Using Softmax")
                self.sftmax_layer = nn.Softmax(0)
                # self.sftmax_layer = nn.Linear(self.neurons,self.output_dimension)

        else:
            print("Simple linear regression")
            self.linear_regression_layer = nn.Linear(
                self.input_dimension, self.output_dimension
            )

        self.init_xavier()

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                if self.activation_name in ["tanh", "relu"]:
                    gain = nn.init.calculate_gain(self.activation_name)
                else:
                    gain = 1
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return reg_loss

    @staticmethod
    def get_activation(activation_name):
        if activation_name in ["tanh"]:
            return nn.Tanh()
        elif activation_name in ["relu"]:
            return nn.ReLU(inplace=True)
        elif activation_name in ["lrelu"]:
            return nn.LeakyReLU(inplace=True)
        elif activation_name in ["sigmoid"]:
            return nn.Sigmoid()
        elif activation_name in ["softplus"]:
            return nn.Softplus(beta=4)
        elif activation_name in ["celu"]:
            return nn.CELU()
        else:
            raise ValueError("Unknown activation function")

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        if self.n_hidden_layers != 0:
            x = self.activation(self.input_layer(x))
            for k, l in enumerate(self.hidden_layers):
                x = self.activation(l(x))
            if self.add_sftmax_layer:
                x = self.output_layer(x)
                return self.sftmax_layer(x)
            else:
                return self.output_layer(x)
        else:
            return self.linear_regression_layer(x)

    @staticmethod
    def fit(
        model,
        training_set,
        x_validation_,
        y_validation_,
        num_epochs,
        optimizer,
        p,
        verbose=True,
    ):
        history = [[], []]
        regularization_param = model.regularization_param

        # Loop over epochs
        for epoch in tqdm(range(num_epochs)):
            if verbose:
                print(
                    "################################ ",
                    epoch,
                    " ################################",
                )

            running_loss = list([0])

            # Loop over batches
            for j, (x_train_, u_train_) in enumerate(training_set):

                def closure():
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    u_pred_ = model(x_train_)
                    loss_u = torch.mean(
                        (
                            u_pred_.reshape(
                                -1,
                            )
                            - u_train_.reshape(
                                -1,
                            )
                        )
                        ** p
                    )
                    loss_reg = model.regularization()
                    loss = loss_u + regularization_param * loss_reg
                    loss.backward()
                    # Compute average training loss over batches for the current epoch
                    running_loss[0] += loss.item() / len(training_set)
                    return loss

                optimizer.step(closure=closure)

            y_validation_pred_ = model(x_validation_)
            validation_loss = torch.mean(
                (
                    y_validation_pred_.reshape(
                        -1,
                    )
                    - y_validation_.reshape(
                        -1,
                    )
                )
                ** p
            ).item()
            history[0].append(running_loss[0])
            history[1].append(validation_loss)

            if verbose:
                print("Training Loss: ", np.round(running_loss[0], 8))
                print("Validation Loss: ", np.round(validation_loss, 8))

        print("Final Training Loss: ", np.round(history[0][-1], 8))
        print("Final Validation Loss: ", np.round(history[1][-1], 8))
        return history

    @staticmethod
    def train_single_configuration(conf_dict, x_, y_, visual_check=None):

        if visual_check is None:
            visual_check = False
        print(conf_dict)

        # Get the configuration to test
        opt_type = conf_dict["optimizer"]
        n_epochs = conf_dict["epochs"]
        n_hidden_layers = conf_dict["hidden_layers"]
        neurons = conf_dict["neurons"]
        regularization_param = conf_dict["regularization_param"]
        regularization_exp = conf_dict["regularization_exp"]
        retrain_seed = conf_dict["init_weight_seed"]
        batch_size = conf_dict["batch_size"]
        activation = conf_dict["activation"]
        add_sftmax_layer = conf_dict["add_sftmax_layer"]

        # Set random seed for reproducibility
        torch.manual_seed(retrain_seed)
        np.random.seed(retrain_seed)

        # define size of test,training and validation data
        test_ratio = 0
        val_ratio = 0.2
        train_ratio = 0.8
        assert np.equal(test_ratio + val_ratio + train_ratio, 1.0)

        validation_size = ceil(val_ratio * x_.shape[0])
        training_size = floor(train_ratio * x_.shape[0])
        # test_size = x_.shape[0] - validation_size - training_size

        x_train = x_[:training_size, :]
        y_train = y_[:training_size, :]

        x_val = x_[training_size + 1 : training_size + validation_size, :]
        y_val = y_[training_size + 1 : training_size + validation_size, :]

        # x_test = x_[training_size + validation_size :, :]
        n_test_samples = 10
        x_test = torch.ones(n_test_samples, 2)
        x_test[:, 0] = torch.tensor(Uniform(0, 12).sample(n_test_samples))
        x_test[:, 1] = torch.tensor(Uniform(-1, 1).sample(n_test_samples))
        n_buckets = len(y_[0])
        h = 2 / n_buckets
        centers = [-1 + h / 2 + i * h for i in range(n_buckets)]

        def inv_dist(w, m, lam):
            if abs(w) == 1:
                return 0
            else:

                res1 = np.log(1 + w) * (-2 + (m / (2 * lam)))
                res2 = np.log(1 - w) * (-2 - (m / (2 * lam)))
                res3 = -((1 - m * w) / (lam * (1 - w**2)))
                if exp(res1 + res2 + res3) == 0:
                    print(w, m, lam)
                return np.exp(res1 + res2 + res3)

        def inv_dist_norm(c, m, lam):
            y = [inv_dist(s, m, lam) for s in c]
            sum = np.array(y).sum()
            if sum is None or sum == 0 or sum == torch.nan:
                print(sum)
            return y / sum

        y_test = torch.tensor(
            np.array([inv_dist_norm(centers, p[1], p[0]) for p in x_test])
        )
        # y_test = y_[training_size + validation_size :, :]

        training_set = utils.data.DataLoader(
            utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        my_network = NeuralNet(
            input_dimension=x_train.shape[1],
            output_dimension=y_train.shape[1],
            n_hidden_layers=n_hidden_layers,
            neurons=neurons,
            regularization_param=regularization_param,
            regularization_exp=regularization_exp,
            retrain_seed=retrain_seed,
            activation_name=activation,
            add_sftmax_layer=add_sftmax_layer,
        )

        if opt_type == "ADAM":
            optimizer_ = torch.optim.Adam(my_network.parameters(), lr=0.001)
        elif opt_type == "LBFGS":
            optimizer_ = optim.LBFGS(
                my_network.parameters(),
                lr=0.1,
                max_iter=1,
                max_eval=50000,
                tolerance_change=1.0 * np.finfo(float).eps,
            )
        else:
            raise ValueError("Optimizer not recognized")

        history = NeuralNet.fit(
            my_network,
            training_set,
            x_val,
            y_val,
            n_epochs,
            optimizer_,
            p=2,
            verbose=False,
        )

        y_test = y_test.reshape(
            -1,
        )

        y_val = y_val.reshape(
            -1,
        )
        y_train = y_train.reshape(
            -1,
        )
        # normalize inputs before feeding to nn
        x_test = (x_test - torch.Tensor([6, 0])) / torch.Tensor([12, 2])
        y_test_pred = my_network(x_test)
        y_val_pred = my_network(x_val)
        y_train_pred = my_network(x_train)
        # raise ValueError
        y_train_pred = y_train_pred.reshape(
            -1,
        )
        y_test_pred = y_test_pred.reshape(
            -1,
        )
        y_val_pred = y_val_pred.reshape(
            -1,
        )

        if visual_check:
            plt.figure()
            n_examples = 3
            output_size = 20
            for i in np.random.choice(
                range(1, (len(y_test) // output_size) + 1),
                size=min(n_examples, len(y_test) // output_size),
            ):
                color = tuple(np.random.choice(range(256), size=3) / 256)
                plt.scatter(
                    np.linspace(-1, 1, output_size),
                    y_test[(i - 1) * output_size : i * output_size],
                    color=color,
                    marker=markers.MarkerStyle(marker="o", fillstyle="none"),
                )
                plt.scatter(
                    np.linspace(-1, 1, output_size),
                    y_test_pred.detach().numpy()[
                        (i - 1) * output_size : i * output_size
                    ],
                    color=color,
                    marker="x",
                )
                # print(y_test[(i - 1) * output_size : i * output_size])
            plt.show(block=True)

        # Compute the relative validation error
        relative_error_train = torch.mean(abs(y_train_pred - y_train)) / torch.mean(
            abs(y_train)
        )
        print(
            "Relative Training Error: ",
            relative_error_train.detach().numpy() * 100,
            "%",
        )

        # Compute the relative validation error
        relative_error_val = torch.mean(abs(y_val_pred - y_val)) / torch.mean(
            abs(y_val)
        )
        print(
            "Relative Validation Error: ",
            relative_error_val.detach().numpy() * 100,
            "%",
        )

        # Compute the relative L2 error norm (generalization error)
        relative_error_test = torch.mean(torch.abs(y_test_pred - y_test)) / torch.mean(
            torch.abs(y_test)
        )
        print(
            "Relative Testing Error: ",
            relative_error_test.detach().numpy() * 100,
            "%",
        )

        return (
            relative_error_train.item(),
            relative_error_val.item(),
            relative_error_test.item(),
        )
