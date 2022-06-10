from helper.ExperimentVisualizer import ExperimentVisualizer
from os import path

pathdict = dict(
    paths=[
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.01-0.01--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-50-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-1.0-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-1.0-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.01-0.01--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
    ],
    m=[
        0,
    ]
    * 9,
    l=[
        1,
    ]
    * 9,
    b=[1000] * 9,
)


print(
    len(pathdict["paths"]), len(pathdict["m"]), len(pathdict["l"]), len(pathdict["b"])
)

in_path = path.join("experiment-data")

for i in range(len(pathdict["paths"])):
    out_path = path.join("experiment-data", "surrogate-inverse-validation")
    f = ExperimentVisualizer.from_samples_file(
        path.join(in_path, pathdict["paths"][i]),
        pathdict["b"][i],
        pathdict["l"][i],
        pathdict["m"][i],
        block=False,
        title=True,
    )
    # f.savefig(f"hst--{pathdict['paths'][i]}-.png")
    f.savefig(path.join(out_path, f"hst--{pathdict['paths'][i]}-.png"))

    g = ExperimentVisualizer.from_samples_file(
        path.join(in_path, pathdict["paths"][i]),
        pathdict["b"][i],
        pathdict["l"][i],
        pathdict["m"][i],
        mode="series",
        block=False,
        title=True,
    )
    # g.savefig(f"srs--{pathdict['paths'][i]}-.png")
    g.savefig(path.join(out_path, f"srs--{pathdict['paths'][i]}-.png"))
