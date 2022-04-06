from helper.Dataset import Dataset
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Generate datapoints")
parser.add_argument(
    "-s", "--size", dest="size", type=int, default=20, help="dataset size"
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
    seed = random.randint(1, 2 ** 32 - 1)
else:
    seed = args.seed

random.seed(seed)
np.random.seed(seed)

# Example parameters

gamma = 0.005
# t_horiz = 200
# nagents = 100
# size = 200

dset = Dataset(
    "random",
    {"lmb": [0, 12], "m": [-1, 1]},
    args.size,
    None,
    dict(
        gamma=gamma,
        theta_bound="lambda g, w: (1 - g) / (1 + abs(w))",
        p="lambda w: 1",
        d="lambda w: (1 - w ** 2)",
        t_end=args.t_horiz,
        n_samples=args.nagents,
        uniform_theta=True,
        lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
        seed=seed,
    ),
)

dset.compute_output()
dataset_name = dset.name

print("reconstructing from dataset file")

d = Dataset.from_json(f"src\\datasets\\{dset.name}.json")
d.compute_output()
d.compute_aggregated_output(20)

print(d.datapoints)
