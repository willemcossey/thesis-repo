# thesis-repo

## About

This repo contains the code for my Master's thesis. \
Currently the experiments are about reproducing results from the book [Interacting Multiagent Systems](https://global.oup.com/academic/product/interacting-multiagent-systems-9780199655465?cc=be&lang=enn&#) by Pareschi & Toscani (referred to in the code as _P&T_).

## How to install

### 1) add requirements

`pip install -r requirements.txt`

### 2) add pre commit hooks
to make sure commits contain nicely formatted code and no jupyter notebooks with output included.

`pre-commit install`

## How to use

The experiments are inside the `src` file and are numbered. Inside is a description of what they do.

Inide the `src/helper` file helper classes are contained to generate the results.
