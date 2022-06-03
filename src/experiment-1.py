from helper.SimulationJob import SimulationJob
from math import exp, sqrt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import path

# case: P = 1, D = 1-w^2

# Parameter initialization
lamb = 0.1
mean_opinion = 0.5
nagents = 10000
t_horiz = 200

# theta_std = 0.1
# gamma = (theta_std ** 2) / lamb
gamma = 0.005
theta_std = sqrt(gamma * lamb)
print(f"gamma= {gamma}")
print(f"theta_std = {theta_std}")


def inv_dist(w, m, lam):
    res = (1 + w) ** (-2 + (m / (2 * lam)))
    res = res * (1 - w) ** (-2 - (m / (2 * lam)))
    res = res * exp(-((1 - m * w) / (lam * (1 - w ** 2))))
    return res


# compute reference solution as described p. 241
resolution = 1000
x = np.linspace(-0.99, 1, num=resolution, endpoint=False)
y = [inv_dist(s, mean_opinion, lamb) for s in x]
data = [x, y]
data = np.transpose(data)
reference = pd.DataFrame(data=data, columns=["w", "g_inf(w)"])

# run MC simulation
sim = SimulationJob(
    gamma,
    theta_std,
    lambda g, w: (1 - g) / (1 + abs(w)),
    lambda w: 1,
    lambda w: (1 - w ** 2),  # P&T p. 241
    mean_opinion,
    t_horiz,
    nagents,
    uniform_theta=True,
)
sim.run()
result_df = pd.Series(sim.result, name="opinion")

# Generate histogram data
counts, bins = np.histogram(result_df, bins=np.linspace(-1, 1, 200))
bins = 0.5 * (bins[:-1] + bins[1:])

# Scaling of reference solution
mean_sim_result = np.mean(counts)
mean_ref_result = np.mean(reference["g_inf(w)"])
reference["g_inf(w)"] = (mean_sim_result / mean_ref_result) * reference["g_inf(w)"]

np.save(
    path.join("experiment-data",f"experiment-1-lambda-{lamb}-nagents-{nagents}-t-horiz-{t_horiz}"),
    sim.result,
)

# Generating figure
plt.ion()
fig = plt.figure()
plt.bar(x=bins, height=counts, width=2 / len(bins))
plt.plot(reference["w"], reference["g_inf(w)"], "r")
plt.suptitle(f"Steady Opinion Profile for P = 1 and D = 1- w^2")
plt.title(f"lambda= {lamb}, n= {nagents} and {t_horiz} simulated time units")
plt.xlabel("Opinion []")
plt.ylabel("Count []")
plt.show(block=True)
cfm = plt.get_current_fig_manager()
cfm.window.raise_()
cfm.window.activateWindow()
