import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from os import path
import subprocess

mpl.style.use(path.join("src", "grayscale_adjusted.mplstyle"))
mpl.rcParams["axes.labelsize"] = 12
git_label = subprocess.check_output(["git", "describe"]).strip().decode("utf-8")
#%% ABC-MH costs

t_fixed = 0
t_variable = 9600 / 7000

#%% S-ABC-MH


nn_model_name_str = "nn-in-2-out-20-hid-2-n-200-activ-tanh-regul-0.0001-2-soft-False-rng-568-data-e3513ee46f1829cd90d7e07973b5e4f1.json-resol-512-n_tr-64-opt-ADAM-ep-4000-batch-64-tr_time-12.pt"

s_variable = 8 / 7000

s_fixed_data_gen_light = 247
s_fixed_train_light = 15

s_fixed_data_gen_heavy = 113312
s_fixed_train_heavy = 148

#%% Plot diagram

xs = np.linspace(0, 100000, 1000)
t_cost = t_fixed + t_variable * xs
s_cost_light = s_fixed_train_light + s_fixed_data_gen_light + s_variable * xs
s_cost_heavy = s_fixed_train_heavy + s_fixed_data_gen_heavy + s_variable * xs

f = plt.figure()
# plt.loglog(xs, t_cost, label="ABC-MH")
# plt.loglog(xs, s_cost_light, label="Surrogate ABC-MH - lightweight")
# plt.loglog(xs, s_cost_heavy, label="Surrogate ABC-MH - heavy")
plt.plot(xs, t_cost, label="ABC-MH")
plt.plot(xs, s_cost_light, label="Surrogate ABC-MH - lightweight")
plt.plot(xs, s_cost_heavy, label="Surrogate ABC-MH - heavy")

# plt.plot(xs,s_cost - s_fixed_data_gen, label="Surrogate ABC-MH - data reused")
# plt.plot(xs,np.fmin(t_cost,s_cost),"r-",label="optimal choice")
plt.xlabel("# samples generated []")
plt.ylabel("Computational cost [cpu seconds]")
plt.legend()
plt.show(block=True)

f.savefig(
    path.join("src", "experiment-data", f"accounting-{git_label}-nn-light-heavy.png"),
    bbox_inches="tight",
)
