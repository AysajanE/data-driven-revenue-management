# Neural Network Approach to Data-Driven Revenue Management

Replication-focused implementation of the paper ["Revenue Management without Demand Forecasting: A Data-Driven Approach for Bid Price Generation"](https://arxiv.org/abs/2304.07391).

This repository is best read as a reverse-engineering exercise: the goal is to understand, implement, and test the paper's data-driven bid-price workflow rather than simply summarize the idea. It is a compact research repo showing how historical booking data can be transformed into bid-price proxies and then learned with a neural model.

## What the project does

The workflow in this repo follows the core logic of the paper:

1. generate or simulate booking-history style data
2. transform those observations into proxy bid prices
3. train a neural-network-based bid-price model
4. compare the resulting policy against benchmark / optimal-style baselines in simulation

## Repository contents

- `main.ipynb`: primary notebook for the full replication workflow
- `main.py`: exported script version of the notebook workflow
- `utils.py`: helper functions for simulation, training, interpolation, and visualization

## Why this repo matters

For portfolio purposes, this repo is evidence of:

- careful paper replication rather than surface-level commentary
- willingness to work from a research paper into executable code
- interest in revenue management, operations research, and ML-based approximation methods

## Recommended way to read the repo

Start with the notebook:

```bash
jupyter notebook
```

Then open:

1. `main.ipynb`
2. `utils.py`

The notebook is the clearest entry point because it shows the full workflow and the figures together.

## Core ingredients

- synthetic booking-data generation
- observation-building for bid-price proxies
- neural network training for bid-price estimation
- interpolation of learned bid prices over the booking horizon
- simulation-based evaluation of revenue and load-factor performance

## Reference

Eren, E. C., Zhang, Z., Rauch, J., Kumar, R., & Kallesen, R. (2023). *Revenue Management without Demand Forecasting: A Data-Driven Approach for Bid Price Generation*. arXiv:2304.07391.

## License

MIT
