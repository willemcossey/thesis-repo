from helper.ExperimentVisualizer import ExperimentVisualizer
from os import path

pathdict = dict(
    paths=[
        "experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-600-burn_in-300-proposal_std--0.01-0.01--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646410274.npz",
        "experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-600-burn_in-300-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646410258.npz",
        "experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-600-burn_in-300-proposal_std--0.1-0.1--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646410235.npz",
    ],
    l=[0.1] * 3,
    m=[0.5] * 3,
    b=[300] * 3,
)


print(
    len(pathdict["paths"]), len(pathdict["m"]), len(pathdict["l"]), len(pathdict["b"])
)

in_path = path.join(
    "src", "experiment-data", "inverse-validation-exp3~proposal", "data"
)

for i in range(len(pathdict["paths"])):
    out_path = path.join(
        "src",
        "experiment-data",
        "inverse-validation-exp3~proposal",
        "selected",
    )
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
