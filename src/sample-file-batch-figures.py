from helper.ExperimentVisualizer import ExperimentVisualizer

pathdict = dict(paths = ["experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-50-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.1-0.1--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.01-0.01--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697523.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-50-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697523.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-1.0-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-1.0-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.01-0.01--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-200-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.01-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.01-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-50-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-0.1-m-0.0-t_horiz-200-nagents-10000.npy--noise_std-1.0-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697523.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-1.0-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.05-0.05--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
"experiment-5-inverse-problem-from--synth-data-lmb-2.0-m-0.5-t_horiz-200-nagents-10000.npy--noise_std-0.1-n_observations-100-num_rounds-6000-burn_in-1000-proposal_std--0.01-0.01--initial_sample--0.5--0.5--t_horiz-100-nagents-1000_1646697525.npz",
],
    m = [0.5,0,0,0.5,0,0.5,0,0,0.5,0,0.5,0.5,0.5,0.5,0,0,0,0.5,0.5],
    l=[0.1,2,0.1,2,2,2,0.1,2,2,2,0.1,0.1,2,2,2,2,0.1,2,2],
    b=[1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
)


print(len(pathdict['paths']),len(pathdict['m']),len(pathdict['l']),len(pathdict['b']))

for i in range(len(pathdict['paths'])):
    f = ExperimentVisualizer.from_samples_file("experiment-data\\"+pathdict["paths"][i],pathdict["b"][i],pathdict["l"][i],pathdict["m"][i])
    f.savefig(f"hist-from--{pathdict['paths'][i]}-.png")
    f = ExperimentVisualizer.from_samples_file("experiment-data\\" + pathdict["paths"][i], pathdict["b"][i],
                                                 pathdict["l"][i], pathdict["m"][i], mode='series')
    f.savefig(f"series-from--{pathdict['paths'][i]}-.png")

