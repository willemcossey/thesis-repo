import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from helper.Dataset import Dataset
from sklearn.linear_model import LinearRegression
from os import path

# import dataset

data = Dataset.from_json(
    path.join("src", "datasets", "7d0ca7a38db3c6cf84efa7bfa36e8a7e.json"), lazy=True
)

n_samples = 1000
#%%
x = torch.from_numpy(data.get_inputs(end=n_samples, silent=False)).to(torch.float)
y = torch.from_numpy(
    data.get_outputs(end=n_samples, otype="aggregated", silent=False)
).to(torch.float)
print(y[1, :])

regr = LinearRegression()
regr.fit(x, y)  # LS optimum

#%%
y_pred = regr.predict(x)

RMSE_train = torch.sqrt(torch.mean((y - y_pred) ** 2))
mean_rel_abs_error = torch.mean(abs(y - y_pred)) / torch.mean(y)

print(f"""RMSE train: {RMSE_train}""")
print(f"""mean rel abs error train: {mean_rel_abs_error}""")

#%%
n_test_samples = 1000


x_test = torch.from_numpy(
    data.get_inputs(start=n_samples, end=n_samples + n_test_samples)
).to(torch.float)
y_test = torch.from_numpy(
    data.get_outputs(
        start=n_samples, end=n_samples + n_test_samples, otype="aggregated"
    )
).to(torch.float)

y_test_pred = regr.predict(x_test)


RMSE_test = torch.sqrt(torch.mean((y_test - y_test_pred) ** 2))
mean_rel_abs_error_test = torch.mean(abs(y_test - y_test_pred)) / torch.mean(y_test)
print(f"""RMSE test set: {RMSE_test}""")
print(f"""mean rel abs error test set: {mean_rel_abs_error_test}""")

#%%
experiment_name_str = (
    f"experiment-11-{data.name}-n_samples-{n_samples}-n_test_samples-{n_test_samples}"
)

#%%

fig = plt.figure()
width = 5
length = 4

for l in range(0, length):
    for w in range(0, width):
        ax = fig.add_subplot(length, width, width * l + w + 1, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], y[:, width * l + w], marker="o")
        ax.scatter(x[:, 0], x[:, 1], y_pred[:, width * l + w], marker="x")

n_bucket = 2
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x[:, 0], x[:, 1], y[:, width * l + w], "o", color="r")
ax.scatter(x[:, 0], x[:, 1], y_pred[:, width * l + w], "x", color="b")
plt.xlabel("lambda")
plt.ylabel("mean opininion")
plt.legend(
    [f"Histogram height bucket #{width*l+w+1}", "Linear Regression approximation"]
)

error = y - y_pred
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x[:, 0], x[:, 1], error[:, width * l + w], "o", color="b")
ax.scatter(np.zeros(x[:, 0].shape), x[:, 1], error[:, width * l + w], ".")
ax.scatter(x[:, 0], np.ones(x[:, 1].shape), error[:, width * l + w], ".")
plt.xlabel("lambda")
plt.ylabel("mean opininion")
ax.set_zlabel("$y - \hat{y}$)")
fig.savefig(path.join("src", "experiment-data", f"{experiment_name_str}-error.eps"))
