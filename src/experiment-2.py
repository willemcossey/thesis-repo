from helper.SimulationJob import SimulationJob
from math import exp, sqrt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import path

matplotlib.use("qt5agg")

# case: P = 1, D = 1-abs(w)

# Parameter initialization
lamb = 1.2
nagents = 10000
t_horiz = 10000

# theta_std = 0.08
# gamma = (theta_std ** 2) / lamb
gamma = 0.001
theta_std = sqrt(gamma * lamb)
print(f"gamma= {gamma}")
print(f"theta_std = {theta_std}")


def inv_dist(w, m, lam):
    res = (1 - abs(w)) ** (-2 - (2 / lam))
    res = res * exp(-((1 - (m * w / abs(w))) / (2 * lam * (1 - abs(w)))))
    return res


# compute reference solution as described p. 241
resolution = 1000
x = np.linspace(-0.99, 1, num=resolution, endpoint=False)
y = [inv_dist(s, 0, lamb) for s in x]
data = [x, y]
data = np.transpose(data)
reference = pd.DataFrame(data=data, columns=["w", "g_inf(w)"])

# run MC simulation
sim = SimulationJob(
    gamma,
    theta_std,
    lambda g, w: 1 - g,
    lambda w: 1,
    lambda w: 1 - abs(w),  # P&T p. 241-242
    t_end=t_horiz,
    n_samples=nagents,
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
    path.join("experiment-data",f"experiment-2-lambda-{lamb}-nagents-{nagents}-t-horiz-{t_horiz}"),
    sim.result,
)

# Generating figure
plt.ion()
fig = plt.figure()
plt.bar(x=bins, height=counts, width=2 / len(bins))
plt.plot(reference["w"], reference["g_inf(w)"], "r")
plt.suptitle(f"Steady Opinion Profile for P = 1 and D = 1- |w|")
plt.title(f"lambda= {lamb}, n= {nagents} and {t_horiz} simulated time units")
plt.xlabel("Opinion []")
plt.ylabel("Count []")
cfm = plt.get_current_fig_manager()
cfm.window.activateWindow()
cfm.window.raise_()
plt.show(block=True)
