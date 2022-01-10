## About
Masterproef-Willem-Cossey

This repo contains the code for my Master's thesis. \

Experiment contents:
*experiment 1-3: reproducing results from the book [Interacting Multiagent Systems](https://global.oup.com/academic/product/interacting-multiagent-systems-9780199655465?cc=be&lang=enn&#) by Pareschi & Toscani (referred to in the code as _P&T_).
	1: case: P = 1, D = 1-w^2
	2: case: P = 1, D = 1-abs(w)
	3: case: P = 1, D = 1-w^2 for different values of lambda and mean opinion
*experiment 4-5: validating MCMC parameter estimation routine
	4: generate synthetic data and store locally
	5: load synthetic data and estimate parameters used. Generate plots and store posterior samples.


## How to install

### 1) add requirements

`pip install -r requirements.txt`

### 2) add pre commit hooks
to make sure commits contain nicely formatted code and no jupyter notebooks with output included.

`pre-commit install`

## How to use

The experiments are inside the `src` file and are numbered. Inside is a description of what they do.

Inide the `src/helper` file helper classes are contained to generate the results.
