import random
import numpy as np
from math import sqrt
from helper.Datapoint import Datapoint
import argparse

parser = argparse.ArgumentParser(description="Generate datapoints")

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
    "-b",
    "-nbins",
    dest="nbins",
    type=int,
    default=20,
    help="number of bins for computing density",
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
    seed = random.randint(1, 2 ** 32 - 1)
else:
    seed = args.seed

random.seed(seed)
np.random.seed(seed)


# generate datapoint settings and value

lmb = args.lmb
m = args.m
t_horiz = args.t_horiz
nagents = args.nagents

# Example parameters

# lmb = 0.2
# m = 0
# t_horiz=200
# nagents=100
# seed = 1867453612


gamma = 0.005
theta_std = sqrt(gamma * lmb)
assert gamma > 0
experiment_assumptions = dict(
    gamma=gamma,
    theta_bound="lambda g, w: (1 - g) / (1 + abs(w))",
    p="lambda w: 1",
    d="lambda w: (1 - w ** 2)",
    t_end=t_horiz,
    n_samples=nagents,
    uniform_theta=True,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    seed=seed,
    lmb=lmb,
    mean_opinion=m,
    theta_std=theta_std,
)

dp = Datapoint({"lmb": lmb, "m": m}, experiment_assumptions)
dp.compute_output()
dp.compute_aggregated_output(100)
print(f"""{dp.name}.json""")
