import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from helper.Dataset import Dataset
from sklearn.linear_model import LinearRegression

# import dataset

data = Dataset.from_json(
    "src\\datasets\\7d0ca7a38db3c6cf84efa7bfa36e8a7e.json", lazy=True
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

print(f"""RMSE: {RMSE_train}""")

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

print(f"""RMSE: {RMSE_test}""")

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
ax.scatter(x[:, 0], x[:, 1], y[:, width * l + w], marker="o")
ax.scatter(x[:, 0], x[:, 1], y_pred[:, width * l + w], marker="x")
plt.xlabel("lambda")
plt.ylabel("mean opininion")
plt.legend(
    [f"Histogram height bucket #{width*l+w+1}", "Linear Regression approximation"]
)
