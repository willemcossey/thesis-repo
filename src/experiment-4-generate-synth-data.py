from src.helper.SimulationJob import SimulationJob
from math import sqrt
import numpy as np
from os.path import join
from helper.ExperimentVisualizer import ExperimentVisualizer

# initialize lambda and m
lmb = 0.5
m = -0.5

gamma = 0.01
theta_std = sqrt(gamma * lmb)
assert gamma > 0
experiment_assumptions = dict(
    free_parameters={"lmb", "m"},
    theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
    gamma=gamma,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    p=lambda w: 1,
    d=lambda w: (1 - w ** 2),
    t_horiz=200,
    nagents=10000,
)

synth_t_horiz = experiment_assumptions["t_horiz"]
synth_nagents = experiment_assumptions["nagents"] * 10

# create synthetic data
synth_job = SimulationJob(
    gamma,
    theta_std,
    experiment_assumptions["theta_bound"],
    experiment_assumptions["p"],
    experiment_assumptions["d"],
    m,
    synth_t_horiz,
    synth_nagents,
    True,
)
synth_job.run()
synth = synth_job.result


np.save(
    join(
        "experiment-data",
        f"synth-data-lmb-{lmb}-m-{m}-t_horiz-{synth_t_horiz}-nagents-{synth_nagents}",
    ),
    synth,
)

print(f"synth-data-lmb-{lmb}-m-{m}-t_horiz-{synth_t_horiz}-nagents-{synth_nagents}")

ExperimentVisualizer.from_file(
    f"synth-data-lmb-{lmb}-m-{m}-t_horiz-{synth_t_horiz}-nagents-{synth_nagents}"
    + ".npy"
)
