# Neural Network Approach to Data-Driven Revenue Management

This repository contains a Python implementation of reproducing the results of the paper "[Revenue Management without Demand Forecasting: A Data-Driven Approach for Bid Price Generation](https://arxiv.org/abs/2304.07391)" by Ezgi C. Eren et al.

## Overview

The goal of this project is to reproduce a neural network approach for generating bid prices in revenue management systems without relying on demand forecasting. The methodology utilizes historical booking data to estimate the value of each unit of airline seat capacity at any given time-to-departure. It employs a neural network model to predict bid prices based on the remaining capacity and days-to-departure.

The implementation includes the following key components:
- Observation building: Transforming historical booking data into a proxy of bid prices.
- Neural network model: Training a deep neural network to estimate bid prices based on the transformed data.
- Simulation: Conducting simulations to evaluate the performance of the data-driven approach compared to optimal and benchmark methods utilizing dynamic programming formulation.

## Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (or any other deep learning library)

## Usage

1. Clone the repository:
```
git clone https://github.com/your-username/data-driven-revenue-management.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the main script:
```
python main.py
```

The script will perform the following steps:
- Generate historical booking data for training.
- Train the neural network model using the transformed historical data.
- Generate future data for testing.
- Evaluate the performance of the data-driven approach against optimal and benchmark methods.
- Visualize the results, including the simulation settings, revenue gaps, and load factor gaps.

## Results

The implementation reproduces the key findings of the paper, demonstrating the effectiveness and robustness of the data-driven approach for bid price generation in revenue management systems. The results include:
- Comparison of the data-driven approach with optimal and benchmark methods.
- Analysis of revenue gaps and load factor gaps under various demand scenarios.
- Visualization of the simulation results, including the distribution of demand ratios and performance metrics.

## Acknowledgments

This implementation is based on the methodology proposed in the following paper:
- Eren, E. C., Zhang, Z., Rauch, J., Kumar, R., & Kallesen, R. (2023). Revenue Management without Demand Forecasting: A Data-Driven Approach for Bid Price Generation. arXiv preprint [arXiv:2304.07391](https://arxiv.org/pdf/2304.07391).

## License

This project is licensed under the [MIT License](LICENSE).
