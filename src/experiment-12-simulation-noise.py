import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from helper.Dataset import Dataset
from sklearn.linear_model import LinearRegression
from os import path
import subprocess
import matplotlib as mpl

mpl.style.use(path.join("src", "grayscale_adjusted.mplstyle"))


# import dataset
dataset_name = "19dd290bf11a4ff6ff8a74793d7f16ae.json"
data = Dataset.from_json(path.join("src", "datasets", dataset_name), lazy=True)

print(data.meta)

n_buckets = 20

# data.compute_aggregated_output(n_buckets)

n_samples = 64
#%%


def inv_dist(w, m, lam):
    if abs(w) == 1:
        return 0
    else:
        res = (1 + w) ** (-2 + (m / (2 * lam)))
        res = res * (1 - w) ** (-2 - (m / (2 * lam)))
        res = res * np.exp(-((1 - m * w) / (lam * (1 - w**2))))
        return res


#%%
x = torch.from_numpy(data.get_inputs(end=n_samples, silent=False)).to(torch.float)
y = torch.from_numpy(
    data.get_outputs(end=n_samples, otype="aggregated", silent=False)
).to(torch.float)


h = 2 / n_buckets
centers = [-1 + h / 2 + i * h for i in range(n_buckets)]
p = centers + [-1, 1]
p.sort()
y_exact = [inv_dist(s, x[0, 1], x[0, 0]) for s in p]
print(np.array(y_exact).sum())
y_exact = y_exact / (np.array(y_exact).sum())

#%%

y_mean = torch.mean(y, dim=0)

plt.figure()
ax = plt.plot(centers, y_mean, label=f"Average simulation result (n={n_samples})")
plt.plot(p, y_exact, color="red", label="Analytical solution")


plt.violinplot(torch.transpose(y, 0, 1), positions=centers, vert=True, widths=0.05)
# plt.boxplot(torch.transpose(y,0,1),vert=True,showmeans=True)

# for i in range(0,n_samples):
#     plt.scatter(range(0,n_buckets),y[i,:],color='blue')

plt.legend(loc="lower center")
# plt.title(
#     f"Avg simulation result vs. analytical result\n{n_samples} simulations with {data.meta['experiment_assumptions']['n_samples']} agents for {data.meta['experiment_assumptions']['t_end']} time units"
# )
#%%
git_label = subprocess.check_output(["git", "describe"]).strip().decode("utf-8")
filename_str = f"experiment-12-n-{n_samples}-nagents-{data.meta['experiment_assumptions']['n_samples']}-t_horiz-{data.meta['experiment_assumptions']['t_end']}-data-{dataset_name}-git-{git_label}.png"
experiment_data_dir = path.join("src", "experiment-data")
plt.savefig(path.join(experiment_data_dir, filename_str))
print(path.join(experiment_data_dir, filename_str))
#%%

RMSE = torch.sqrt(torch.mean((y - y_mean) ** 2))
MAE = torch.mean(abs(y - y_mean))
MRAE = torch.mean(abs(y - y_mean)) / torch.mean(abs(y))
y_std = torch.median(torch.std(y, dim=0))
RAE = torch.mean(abs(y - y_mean) / (y_mean)) * 100


# print(f"mean: {y_mean}")
print(f"median std: {y_std**2}")
print(f"MAE: {MAE}")
print(f"MRAE: {MRAE}")
print(f"RMSE: {RMSE}")
print(f"mean relative error: {RAE:.2f}%")
