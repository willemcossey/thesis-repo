from helper.Datapoint import Datapoint
import random
import numpy as np
from math import sqrt
from os import path
import subprocess

# from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)

n_parconfig = 5
n_instances_per_parconfig = 10
low_n_agents = 2
high_n_agents = 4
len_n_agents = 8
n_agents_for_instance = (
    np.round(
        np.power(
            10, np.linspace(low_n_agents, high_n_agents, len_n_agents, endpoint=True)
        )
    )
    .astype(int)
    .tolist()
)
t_horiz_for_instance = 100
n_buckets = 20

lmbs = np.linspace(0.1, 5, n_parconfig, endpoint=False)
ms = np.linspace(-0.9, 1, n_parconfig, endpoint=False)

#%%

dp_list = []

for a in range(len(n_agents_for_instance)):
    dp_list.append([])
    for p in range(n_parconfig):
        dp_list[a].append([])
        for i in tqdm(range(n_instances_per_parconfig)):
            n_agents = n_agents_for_instance[a]
            seed = random.randint(1, 2**32 - 1)
            lmb = lmbs[p]
            m = ms[p]
            gamma = 0.005
            theta_std = sqrt(gamma * lmb)
            assert gamma > 0
            experiment_assumptions = dict(
                gamma=gamma,
                theta_bound="lambda g, w: (1 - g) / (1 + abs(w))",
                p="lambda w: 1",
                d="lambda w: (1 - w ** 2)",
                t_end=t_horiz_for_instance,
                n_samples=n_agents,
                uniform_theta=True,
                lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
                seed=seed,
                lmb=lmb,
                mean_opinion=m,
                theta_std=theta_std,
            )
            # print(n_agents,lmb,m,seed)
            dp = []
            dp = Datapoint({"lmb": lmb, "m": m}, experiment_assumptions)
            dp.compute_output(silent=True)
            # print(a,p,i)
            # print(id(dp))
            dp_list[a][p].append(dp)

#%%

errors = np.ones([len_n_agents, n_parconfig])
outputs = np.ones([len_n_agents, n_parconfig, n_instances_per_parconfig, n_buckets])

for a in range(len_n_agents):
    for p in range(n_parconfig):
        [dp.compute_aggregated_output(n_buckets) for dp in dp_list[a][p]]
        outputs[a, p, :, :] = np.array(
            [dp.output["aggregated"] for dp in dp_list[a][p]]
        )
        print(a, p)
        mean_solution = np.mean(outputs[a, p, :, :], axis=0)
        res = outputs[a, p, :, :] - mean_solution
        abs_errors = np.abs(res)
        mean_abs_err = np.mean(abs_errors)
        RMAE = np.divide(mean_abs_err, np.mean(np.abs(mean_solution)))
        errors[a, p] = RMAE

#%%
git_label = subprocess.check_output(["git", "describe"]).strip().decode("utf-8")
filename_str = f"experiment-13-p-{n_parconfig}-i-{n_instances_per_parconfig}-a-{low_n_agents}-{high_n_agents}-{len_n_agents}-t_horiz-{t_horiz_for_instance}-n-{n_buckets}-seed-{seed}-git-{git_label}"
experiment_data_dir = path.join("src", "experiment-data")
np.savez(path.join(experiment_data_dir, f"{filename_str}.npz"), errors=errors)

#%%

# f = np.load(path.join(experiment_data_dir, f"{filename_str}.npz"))
f = np.load(
    path.join(
        experiment_data_dir,
        f"experiment-14-p-3-i-100-a-2-2-1-t-10-ts-60-n-20-seed-3540192740-git-exp-4-working-98-g4fb7ad0.npz",
    )
)
errors = f["errors"]

#%%

mean_error = [
    np.mean([errors[a, p] for p in range(n_parconfig)])
    for a in range(len(n_agents_for_instance))
]

print(mean_error)

plt.figure()
plt.loglog(n_agents_for_instance, mean_error, label="mean RMAE")
plt.loglog(
    np.linspace(10**2, 10**4, 100),
    0.1 * np.power(np.linspace(10**2, 10**4, 100), -0.5),
    linestyle="--",
    label="$x^{-1/2}$ reference",
)
plt.xlabel("#particles simulated")
plt.ylabel("Relative Mean Absolute error")
# plt.title(
#     f"#configurations = {n_parconfig}, #repetitions = {n_instances_per_parconfig}, #buckets = {n_buckets}"
# )
plt.legend()
# plt.show(block=True)
plt.savefig(path.join(experiment_data_dir, f"{filename_str}-figure-1.png"))
