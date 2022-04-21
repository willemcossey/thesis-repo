from helper.SimulationJob import SimulationJob
from math import sqrt
import numpy as np
from os.path import join
from helper.ExperimentVisualizer import ExperimentVisualizer
import random

import argparse

parser = argparse.ArgumentParser(description="Generate synthetic data.")

parser.add_argument(
    "-l", "--lambda", dest="lmb", type=float, default=1, help="lambda value"
)
parser.add_argument(
    "-m", "--mean", dest="m", type=float, default=0, help="mean opinion value"
)
parser.add_argument(
    "-t", "--t_horiz", dest="t_horiz", type=int, default=50, help="time horizon"
)
parser.add_argument(
    "-n", "--nagents", dest="nagents", type=int, default=100, help="number of agents"
)
parser.add_argument(
    "-r",
    "--rng_seed",
    dest="seed",
    type=int,
    default=None,
    help="random number generator seed to be used",
)
parser.add_argument("--show", action="store_true")

args = parser.parse_args()

if args.seed is None:
    seed = random.randint(1, 2**32 - 1)
else:
    seed = args.seed

random.seed(seed)
np.random.seed(seed)

lmb = args.lmb
m = args.m

print(lmb, type(lmb), m, type(m))

gamma = 0.01
theta_std = sqrt(gamma * lmb)
assert gamma > 0
experiment_assumptions = dict(
    free_parameters={"lmb", "m"},
    theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
    gamma=gamma,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    p=lambda w: 1,
    d=lambda w: (1 - w**2),
    t_horiz=args.t_horiz,
    nagents=args.nagents,
)

synth_t_horiz = experiment_assumptions["t_horiz"]
synth_nagents = experiment_assumptions["nagents"]

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

out_filename = f"synth-data-lmb-{lmb}-m-{m}-t_horiz-{synth_t_horiz}-nagents-{synth_nagents}-seed-{seed}"

np.save(
    join(
        "experiment-data",
        out_filename,
    ),
    synth,
)

print(out_filename)

print(args.show)

if args.show:
    ExperimentVisualizer.from_file(out_filename + ".npy")
