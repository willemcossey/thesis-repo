## About
Masterproef-Willem-Cossey

This repo contains the code for my Master's thesis

Experiment contents:
* experiment 1-3: reproducing results from the book [Interacting Multiagent Systems](https://global.oup.com/academic/product/interacting-multiagent-systems-9780199655465?cc=be&lang=enn&#) by Pareschi & Toscani (referred to in the code as _P&T_).
	* 1: case: P = 1, D = 1-w^2
	* 2: case: P = 1, D = 1-abs(w)
	* 3: case: P = 1, D = 1-w^2 for different values of lambda and mean opinion
* experiment 4-5,16: validating MCMC parameter estimation routine
	* 4: generate synthetic data and store locally
	* 5: load synthetic data and estimate parameters used. Generate plots and store posterior samples.
	* experiment 16: Perform the inverse problem with a neural network surrogate
* experiment 6-9: Construct training datapoints and datasets
	* 6: validating the addition of random noise to synthetic population data
	* 7: Generate one datapoint
	* 8: Generate one dataset
	* 9: Construct a dataset from a list of child datasets
* experiment 10: Train a neural network from a dataset
* experiment 11: Perform an OLS regression on a dataset and report the error
* experiment 12-15: Simulation routine performance
	* 12: Check the noise present on the simulation results
	* 13: Check the statistical error on the simulation results
	* 14: Check the total error vs. the analytical solution on the simulation results
* experiment 15: Check the performance of NN after training for different quality and quantity of data

Other scripts:
* computational accounting: generate plot of computational cost different MCMC methods
* inv-dist-stability-test: Check the numerical stability of the implementation of the analytical solution for the stationary opinion density
* inverse-problem-generate-results-table: Generate an excel file with the hyperparameters and results of a list of inverse problem experiments
* surrogata-inverse-problem-generate-results-table: Same as above for the surrogate inverse problems
* plot-style-test: Plotting an example of the current version of the matplotlib .mplstyle file
* sample-file-batch-figures: Generate figures for a list of result files of  inverse problem experiments

Jupyter notebooks: 
* inverse-problem-solution-analysis: Visualize and analyze results of inverse problem experiment
* surrogate-inverse-problem-solution-analysis: Same as above for the surrogate inverse problem
* TruncatedNormal_moments_experiment: Notebook investigating the properties of truncated normal distributions
	

## How to install

### 1) add requirements

`pip install -r requirements.txt`

### 2) add pre commit hooks
to make sure commits contain nicely formatted code and no jupyter notebooks with output included.

`pre-commit install`

## How to use

The experiments are inside the `src` file and are numbered. Inside is a description of what they do.

Inide the `src/helper` file helper classes are contained to generate the results.
