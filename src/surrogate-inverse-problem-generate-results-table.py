#!/usr/bin/env python
# coding: utf-8

# ## Analyse the results of a Posterior Sampling Routine
#
# - Load data
#
# ### Symptoms
#
# - zoom in
# - Are there any lambda in lumps ?
# - Color samples wrt lump
# - Does running average look different ?
#
# ### Diagnosis
#
# ### Solution

# In[1]:


import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
from helper.SimulationJob import SimulationJob
from math import sqrt
from helper.InverseProblem import InverseProblem
from scipy.stats import ks_2samp
import seaborn as sns


# In[2]:

expdir = os.path.abspath("src\\experiment-data\surrogate-inverse-validation")
expname = "various"
dirname = os.path.join(expdir, expname, "data")
datafiles = np.array(
    [
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.01-0.01--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.01-n_observations-50-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.01-0.01--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-0.1-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-1.0-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.1-0.1--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
        "C:/Users/wille/thesis-repo/src/experiment-data/surrogate-inverse-validation/various/experiment-16--synth-data-lmb-1.0-m-0.0-t_horiz-200-nagents-10000.npy--noise-1.0-n_observations-25-num_rounds-6000-burn_in-1000-proposal--0.05-0.05--initial_sample--1.5--0.2-nn-nn-start-27576812-seed-64.npz",
    ]
)
files_to_load = range(0, len(datafiles))
# files_to_load = range(0, 20)
paths = [os.path.join(datafiles[i]) for i in files_to_load]
paths

all_data = [np.load(p) for p in paths]
# ["$\lambda^*$ "," $m^*$ "," $N_{obs}$ "," $\sigma_n$ "," $N_{burn}$ "," $N_s$ "," $\lambda_0$ "," $m_0$ "," $\sigma_p$ "," $N_{it}$ "," $T_{it}$ "," $\bar{\lambda}$ "," $\bar{m}$ ","$\lambda_{MAP}$","$m_{MAP}$"," $p\!-\!value_{synth}$ "," $p\!-\!value_{MAP}$"]
summary_table = pd.DataFrame(
    data=[],
    columns=[
        "n_obs",
        "noise",
        "n_burn",
        "n_samp",
        "init_lmb",
        "init_m",
        "prop",
        "nn",
        "true_lmb",
        "true_m",
        "avg_lmb",
        "avg_m",
        "MAP_lmb",
        "MAP_m",
        "ks_synth",
        "ks_map",
    ],
)
# summary_table = pd.DataFrame(data = [],columns = ["n_obs","noise","n_burn","n_samp","init_lmb","init_m","prop","n_it","t_it","true_lmb","true_m","avg_lmb","avg_m","MAP_lmb","MAP_m"])

#%%

for j in range(len(datafiles)):

    in_research = j
    name = datafiles[in_research]
    data = all_data[in_research]
    name

    x = re.search("(?<=--)(.*?)(?=--)", name)
    synth_data_name = x.group()
    print(synth_data_name)
    true_lmb = float(re.search("(?<=lmb-)(.*?)(?=-m)", synth_data_name).group())
    true_m = float(re.search("(?<=m-)(.*?)(?=-t)", synth_data_name).group())
    sp = re.split("npy--", name)
    # print(sp)
    exp_meta = sp[1]
    print(exp_meta)
    x = re.search("(?<=noise-)(.*?)(?=-n_obs)", exp_meta)
    noise = float(x.group())
    print(noise)
    x = re.search("(?<=-n_observations-)(.*?)(?=-num)", exp_meta)
    n_observations = int(x.group())
    print(n_observations)
    x = re.search("(?<=-num_rounds-)(.*?)(?=-burn)", exp_meta)
    num_rounds = int(x.group())
    print(num_rounds)
    x = re.search("(?<=-burn_in-)(.*?)(?=-proposal)", exp_meta)
    burn_in = int(x.group())
    print(burn_in)
    x = re.search("(?<=-proposal--)(.*?)(?=-)", exp_meta)
    proposal = float(x.group())
    print(proposal)
    x = re.search("(?<=-initial_sample)(.*?)(?<=nn)", exp_meta)
    s = x.group()
    print(s)
    x = re.search("(?<=--)(.*?)(?=-)", s)
    init_lmb = float(x.group())
    print(init_lmb)
    y = re.search("(?<=5-)(.*?)(?=-nn)", s)
    init_m = float(y.group())
    print(init_m)
    y = re.search("(?<=-nn-)(.*?)(?=-s)", exp_meta)
    nn = str(y.group())
    print(nn)
    x = re.search("(?<=-seed-)(.*?)(?=.npz)", exp_meta)
    seed = int(x.group())
    print(seed)

    df = pd.DataFrame.from_dict(dict(data), orient="columns")
    burnin = burn_in
    df_postburn = df.loc[burnin:, :]

    #

    # plt.figure(1)
    # plt.subplot(211)
    # # plt.scatter(df.index,df['lmb'],marker='x')
    # plt.plot(df.index, df["lmb"])
    # plt.subplot(212)
    # plt.plot(df.index, df["m"])
    # plt.show()

    # #

    # # from IPython.display import Image

    # # Image(
    # #     filename=f"""experiment-data\\inverse-validation-exp5~multiple\\hist-from--{name}-.png"""
    # # )

    #

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    lmb_count, lmb_x = np.histogram(df_postburn.loc[:, "lmb"], bins=200)
    m_count, m_x = np.histogram(df_postburn.loc[:, "m"], bins=200)
    # axs[0].set_ylabel("count")
    # axs[0].set_xlabel("lambda")
    # axs[1].set_xlabel("mean opinion")
    # plt.show()

    mean_lmb = np.mean(df_postburn.loc[:, "lmb"])
    mean_m = np.mean(df_postburn.loc[:, "m"])

    m_MAP = float(m_x[np.argmax(m_count)])
    lmb_MAP = float(lmb_x[np.argmax(lmb_count)])
    print(m_MAP, lmb_MAP)
    MAP = dict(m=m_MAP, lmb=lmb_MAP)

    # plt.figure()
    # plt.hist2d(df_postburn.loc[:, "lmb"], df_postburn.loc[:, "m"], density=True)
    # plt.show()

    # sns.pairplot(df_postburn, kind="hist")
    # plt.show()

    synth_data_path = f"""src\\experiment-data\\{synth_data_name}"""
    synth = np.load(synth_data_path)
    s_df = pd.DataFrame(synth, columns=["w"])

    # plt.figure(1)
    # plt.hist(s_df["w"], bins=200)
    # plt.show()

    def generate_invariant(lmb, m):
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
            t_horiz=200,
            nagents=10000,
        )

        t_horiz = experiment_assumptions["t_horiz"]
        nagents = experiment_assumptions["nagents"]

        # create synthetic data
        job = SimulationJob(
            gamma,
            theta_std,
            experiment_assumptions["theta_bound"],
            experiment_assumptions["p"],
            experiment_assumptions["d"],
            m,
            t_horiz,
            nagents,
            True,
        )

        job.run()
        return job.result

    for combo in [
        MAP,
    ]:
        combo["w"] = generate_invariant(combo["lmb"], combo["m"])

    # plt.figure(4)
    # plt.hist(MAP["w"], bins=200, histtype="step", density=True, label="MAP estimate")
    # plt.hist(s_df["w"], bins=200, histtype="step", density=True, label="synthetic data")
    # plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
    # plt.show(block=True)

    # plt.savefig()

    # ### Zoom in

    # start_index = 1600
    # end_index = 1700

    # df_zoom = df.loc[start_index:end_index, :]
    # plt.figure(2)
    # plt.subplot(211)
    # plt.scatter(df_zoom.index, df_zoom.loc[:, "lmb"])
    # plt.subplot(212)
    # plt.scatter(df_zoom.index, df_zoom.loc[:, "m"])

    # ### Comparison synthetic population vs synthetic sample

    def generate_synth_sample(synth_population, n_observations, noise_std, seed=None):
        if seed is None:
            seed = random.randint(1, 2**32 - 1)
        else:
            seed = seed

        random.seed(seed)
        np.random.seed(seed)
        observed_data = synth_population[0:n_observations]
        # add noise
        noise_std = noise_std
        noisy_observed_data = InverseProblem.add_noise(observed_data, noise_std)
        return noisy_observed_data

    # seed = int(re.search("(?<=seed-)(.*?)(?=[.])", name).group())
    # noise = float(re.search("(?<=noise-)(.*?)(?=[-])", name).group())
    # n_observations = int(re.search("(?<=n_observations-)(.*?)(?=[-])", name).group())

    # print(seed, noise, n_observations)

    synth_sample = generate_synth_sample(synth, n_observations, noise, seed)
    np.mean(synth)

    # plt.figure(5)
    # plt.hist(synth_sample, 200)
    # plt.show()

    # plt.figure(6)
    # plt.hist(
    #     s_df["w"],
    #     bins=200,
    #     label="synthetic population",
    #     cumulative=True,
    #     density=True,
    #     color="paleturquoise",
    #     range=(-1, 1),
    # )
    # plt.hist(
    #     synth_sample,
    #     histtype="step",
    #     bins=200,
    #     cumulative=True,
    #     density=True,
    #     label="synthetic sample",
    #     color="black",
    #     range=(-1, 1),
    # )
    # plt.hist(
    #     MAP["w"],
    #     bins=200,
    #     histtype="step",
    #     label="MAP population estimate",
    #     cumulative=True,
    #     density=True,
    #     color="dodgerblue",
    #     range=(-1, 1),
    # )
    # plt.legend()
    # plt.show()

    ks_dist_MAP = ks_2samp(MAP["w"], synth_sample)
    ks_dist_synth = ks_2samp(s_df["w"], synth_sample)
    print(
        pd.DataFrame(
            index=["MAP", "synth data"],
            data=[ks_dist_MAP.pvalue, ks_dist_synth.pvalue],
            columns=["p-value"],
        )
    )
    print(summary_table.columns)
    # print([true_lmb,true_m,n_observations,noise,burn_in,num_rounds,init_lmb,init_m,proposal,nagents,t_horiz,mean_lmb,mean_m,lmb_MAP,m_MAP,ks_dist_synth.pvalue,ks_dist_MAP.pvalue])
    summary_table.loc[in_research, :] = [
        n_observations,
        noise,
        burn_in,
        num_rounds,
        init_lmb,
        init_m,
        proposal,
        nn,
        true_lmb,
        true_m,
        mean_lmb,
        mean_m,
        lmb_MAP,
        m_MAP,
        ks_dist_synth.pvalue,
        ks_dist_MAP.pvalue,
    ]
    # summary_table.loc[in_research,:] = [n_observations,noise,burn_in,num_rounds,init_lmb,init_m,proposal,nagents,t_horiz,true_lmb,true_m,mean_lmb,mean_m,lmb_MAP,m_MAP]

    summary_table.to_excel(os.path.join(dirname, f"surrogate_{expname}_summary.xlsx"))
