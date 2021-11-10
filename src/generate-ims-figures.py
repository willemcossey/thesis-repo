from helper.SimulationJob import SimulationJob
from math import exp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")

# case: P = 1, D = 1-w^2

lamb = 0.1
theta_std = 0.1
gamma = (theta_std ** 2) / lamb
print(f"gamma= {gamma}")
print(f"theta_std = {theta_std}")

w_ = [-0.3, 0.7]
rho = 0.5

exponent = 1

# compute reference solution as described p. 241


def inv_dist(w, m, lam):
    res = (1 + w) ** (-2 + (m / (2 * lam)))
    res = res * (1 - w) ** (-2 - (m / (2 * lam)))
    res = res * exp(-((1 - m * w) / (lam * (1 - w ** 2))))
    return res


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
    lambda g, w: (1 - g) / (1 + abs(w)),
    lambda w: 1,
    lambda w: (1 - w ** 2),  # P&T p. 241
    100,
    10000,
)

sim.run()

result_df = pd.Series(sim.result, name="opinion")

# computing histogram data

counts, bins = np.histogram(result_df, bins=np.linspace(-1, 1, 200))
bins = 0.5 * (bins[:-1] + bins[1:])

# Scaling of reference solution
mean_sim_result = np.mean(counts)
mean_ref_result = np.mean(reference["g_inf(w)"])
reference["g_inf(w)"] = (mean_sim_result / mean_ref_result) * reference["g_inf(w)"]

# Generating figure
plt.ion()

fig = plt.figure()
plt.bar(x=bins, height=counts, width=2 / len(bins))
plt.plot(reference["w"], reference["g_inf(w)"], "r")
cfm = plt.get_current_fig_manager()
cfm.window.activateWindow()
cfm.window.raise_()
plt.show(block=True)
