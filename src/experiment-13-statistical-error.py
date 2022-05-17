from helper.Datapoint import Datapoint
import random
import numpy as np
from math import sqrt

# from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt


random.seed(42)
np.random.seed(42)

n_parconfig = 5
n_instances_per_parconfig = 1
n_agents_for_instance = (
    np.round(np.power(10, np.linspace(2, 6, 16))).astype(int).tolist()
)
t_horiz_for_instance = 10
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

std = []

for a in range(len(n_agents_for_instance)):
    std.append([])
    for p in range(n_parconfig):
        [dp.compute_aggregated_output(n_buckets) for dp in dp_list[a][p]]
        outputs = [dp.output["aggregated"] for dp in dp_list[a][p]]
        print(a, p)
        std[a].append(np.std(outputs, axis=0))

#%%

mean_std = [
    np.mean([std[a][i] for i in range(n_parconfig)])
    for a in range(len(n_agents_for_instance))
]

print(mean_std)

plt.figure()
plt.loglog(n_agents_for_instance, mean_std, label="mean std")
plt.loglog(
    np.linspace(10**2, 10**4, 100),
    0.1 * np.power(np.linspace(10**2, 10**4, 100), -0.5),
    linestyle="--",
    label="$x^{-1/2}$ reference",
)
plt.xlabel("#particles simulated")
plt.ylabel("mean standard deviation")
plt.title(
    f"#configurations = {n_parconfig}, #repetitions = {n_instances_per_parconfig}, #buckets = {n_buckets}"
)
plt.legend()
plt.show(block=True)
