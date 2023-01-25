# Vertical Federated Learning Quickstart

This repo contains a (growing) collection of scripts to get started with vertical FL.

## Setup

Create a virtual environemnt and install the packages from `requirements.txt`.
The code has been tested with Python 3.10.

## Use Cases

### PyTorch MLP Trained on Adult

This use case is a reimplementation of a pipeline defined by [Wei et al.](https://arxiv.org/abs/2202.04309)
and consequently modified by [@BalticBytes](https://github.com/BalticBytes/vertical-federated-learning-kang).
It uses the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) to train a small MLP with PyTorch.
The main entrypoint is `scripts/standalone_train.py`, which can be called with different arguments (see `-h`).

### Flower

TODO
