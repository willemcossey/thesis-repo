from helper.SimulationJob import SimulationJob
import random
import numpy as np
from math import sqrt

# from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

n_parconfig = 3
n_instances_per_parconfig = 10
n_agents_for_instance = (
    np.round(np.power(10, np.linspace(2, 4, 8))).astype(int).tolist()
)
# n_agents_for_instance = [100,1000,10000]
t_step = 10
n_timesteps = 10
n_buckets = 20

lmbs = np.linspace(0.1, 5, n_parconfig, endpoint=False)
ms = np.linspace(-0.9, 1, n_parconfig, endpoint=False)

# %%
h = 2 / n_buckets
result_list = np.ones(
    [
        len(n_agents_for_instance),
        n_parconfig,
        n_instances_per_parconfig,
        n_timesteps,
        n_buckets,
    ]
)

for a in range(len(n_agents_for_instance)):
    for p in range(n_parconfig):
        for i in tqdm(range(n_instances_per_parconfig)):
            latest_result = None
            for t in range(n_timesteps):
                n_agents = n_agents_for_instance[a]
                seed = random.randint(1, 2**32 - 1)
                lmb = lmbs[p]
                m = ms[p]
                gamma = 0.005
                theta_std = sqrt(gamma * lmb)
                assert gamma > 0
                sim = SimulationJob(
                    gamma,
                    theta_std,
                    lambda g, w: (1 - g) / (1 + abs(w)),
                    lambda w: 1,
                    lambda w: (1 - w**2),  # P&T p. 241
                    m,
                    t_step,
                    n_agents,
                    uniform_theta=True,
                )
                sim.hide_progress = True
                sim.run(init_samples=latest_result)
                latest_result = sim.result.copy()
                # print(n_agents,lmb,m,seed)
                # print(a,p,i)
                # print(id(dp))
                result_list[a, p, i, t, :] = (
                    np.histogram(sim.result, n_buckets, range=[-1, 1], density=True)[0]
                    * h
                )

#%%


def inv_dist(w, m, lam):
    if abs(w) == 1:
        return 0
    else:

        res = np.power((1 + w), (-2 + (m / (2 * lam))))
        res = res * (1 - w) ** (-2 - (m / (2 * lam)))
        res = res * np.exp(-((1 - m * w) / (lam * (1 - w**2))))
        return res


def inv_dist_norm(c, m, lam):
    y = [inv_dist(s, m, lam) for s in c]
    return y / (np.array(y).sum())


#%%
bias = []

h = 2 / n_buckets
centers = [-1 + h / 2 + i * h for i in range(n_buckets)]
x = centers + [-1, 1]
x.sort()

#%% First visualization: behaviour over time for one a and p.

a = 2
p = 1

data = result_list[a, p, :, :]

avg_data = np.mean(data, axis=0)

y_exact = [inv_dist(s, ms[p], lmbs[p]) for s in x]
y_exact = y_exact / (np.array(y_exact).sum())

plt.figure()
for t in range(0, n_timesteps, 2):
    plt.plot(centers, avg_data[t], label=f"t = {(t+1)*t_step}")
plt.plot(x, y_exact, label=f"analytical solution", color="red", marker="o")
plt.legend()
plt.show()

#%% mean(mean solution wrt i - analytical solution) wrt p as a function of a for fixed t

data = result_list[:, :, :, :]

exact_solutions = [inv_dist_norm(centers, ms[p], lmbs[p]) for p in range(n_parconfig)]

errors = np.ones(data.shape[:-1])

for p in range(n_parconfig):
    for i in range(n_instances_per_parconfig):
        for a in range(len(n_agents_for_instance)):
            for t in range(n_timesteps):
                errors[a, p, i, t] = np.linalg.norm(
                    data[a, p, i, t] - exact_solutions[p], 2
                )

err_agg = np.mean(errors, axis=(1, 2))

plt.figure()
for t in range(0, n_timesteps, 2):
    plt.loglog(n_agents_for_instance, err_agg[:, t], label=f"t = {(t+1)*t_step}")
plt.legend()
plt.xlabel("#particles simulated")
plt.ylabel(f"MSE (n={n_instances_per_parconfig*n_parconfig})")
plt.show()


#%% mean wrt i as a function of t for fixed a

data = result_list[:, :, :, :]

exact_solutions = [inv_dist_norm(centers, ms[p], lmbs[p]) for p in range(n_parconfig)]

errors = np.ones(data.shape[:-1])

for p in range(n_parconfig):
    for i in range(n_instances_per_parconfig):
        for a in range(len(n_agents_for_instance)):
            for t in range(n_timesteps):
                errors[a, p, i, t] = np.linalg.norm(
                    data[a, p, i, t] - exact_solutions[p], 2
                )

err_agg = np.mean(errors, axis=(1, 2))

plt.figure()
for a in range(len(n_agents_for_instance)):
    plt.loglog(
        [(t + 1) * t_step for t in range(n_timesteps)],
        err_agg[a, :],
        label=f"{n_agents_for_instance[a]} particles",
    )
plt.loglog(
    np.power(10, np.linspace(1, 2, 8)),
    0.6 * np.power(np.power(10, np.linspace(1, 2, 8)), -0.5),
    label="$t^{-1/2}$ reference",
    linestyle="--",
)
plt.legend()
plt.xlabel("simulated time")
plt.ylabel(f"MSE (n={n_instances_per_parconfig*n_parconfig})")
plt.show()


#%%

data = result_list[:, :, :, :]

exact_solutions = [inv_dist_norm(centers, ms[p], lmbs[p]) for p in range(n_parconfig)]

errors = np.ones(data.shape[:-1])

for p in range(n_parconfig):
    for i in range(n_instances_per_parconfig):
        for a in range(len(n_agents_for_instance)):
            for t in range(n_timesteps):
                errors[a, p, i, t] = np.linalg.norm(
                    data[a, p, i, t] - exact_solutions[p], 2
                )

p = 2
a = 7

err_mean = np.mean(errors[a, p, :, :], axis=0)
err_lower_conf = np.percentile(errors[a, p, :, :], 95, axis=0)
err_upper_conf = np.percentile(errors[a, p, :, :], 5, axis=0)

plt.figure()
plt.fill_between(
    [(t + 1) * t_step for t in range(n_timesteps)],
    err_lower_conf,
    err_upper_conf,
    facecolor="orange",  # The fill color
    color="black",  # The outline color
    alpha=0.2,
)
plt.plot([(t + 1) * t_step for t in range(n_timesteps)], err_mean, color="red")
plt.xlabel("simulated time")
plt.ylabel("$|avg solution - analytical solution|_{L^2}$")
plt.show()


#%%

data = result_list[:, :, :, :]

exact_solutions = [inv_dist_norm(centers, ms[p], lmbs[p]) for p in range(n_parconfig)]

errors = np.ones(data.shape[:-1])

for p in range(n_parconfig):
    for i in range(n_instances_per_parconfig):
        for a in range(len(n_agents_for_instance)):
            for t in range(n_timesteps):
                errors[a, p, i, t] = np.linalg.norm(
                    data[a, p, i, t] - exact_solutions[p], 2
                )

a = 7


plt.figure()
for p in range(n_parconfig):
    err_mean = np.mean(errors[a, p, :, :], axis=0)
    err_lower_conf = np.percentile(errors[a, p, :, :], 95, axis=0)
    err_upper_conf = np.percentile(errors[a, p, :, :], 5, axis=0)
    plt.fill_between(
        [(t + 1) * t_step for t in range(n_timesteps)],
        err_lower_conf,
        err_upper_conf,  # The outline color
        alpha=0.2,
        label=f"m={ms[p]:.2f}, lmb={lmbs[p]:.2f}",
    )
    plt.plot([(t + 1) * t_step for t in range(n_timesteps)], err_mean, color="red")
plt.xlabel("simulated time")
plt.ylabel("$|avg solution - analytical solution|_{L^2}$")
plt.legend()
plt.show()
