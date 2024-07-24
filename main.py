import numpy as np
import math, statistics
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")


from utils import (
    p0_func, alpha_func, pmax_func, Pw_func, 
    simulate_departure_training, simulate_departure_testing_baseline, simulate_departure_testing_robustness,
    generate_historical_data, observation_building, prepare_data, create_nn_model,
    train_nn_model, train_bid_price_model, predict_bid_prices, interpolate_bid_prices,
    get_bid_price, visualize_predicted_bid_prices, visualize_bid_price_proxies, plot_bid_price_proxy, generate_future_data_baseline
)

def main():
    seed = 2024  # Set a seed value

    # Parameter values
    C = 10  # Capacity
    T = 30  # Time horizon
    delta_t = 1  # Time interval
    num_simulations = 50  # Number of simulations
    num_departures = 10  # Number of departure dates
    lambda_range = [2.4, 3.6]  # Range for arrival rate (lambda)
    lambda_test_range = [1.8, 3.6]  # Range for arrival rate (lambda)

    # Define data collection points (DCPs) with a one-day interval
    dcps = list(range(T, 0, -1))
    days_prior = list(range(1, T + 1))

    # Measure execution time for parallel processing approach
    start_time = time.time()
    prices, times, total_booked_seats, load_factors, total_revenues, lambda_vals, lambda_test_vals = generate_historical_data(
        C, T, lambda_range, lambda_test_range, delta_t, num_simulations, num_departures, seed
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Parallel processing approach execution time: {execution_time:.2f} seconds")

    transformed_prices_all_simulations = []
    for sim_idx in range(num_simulations):
        sim_prices = prices[sim_idx]
        sim_times = times[sim_idx]
        transformed_prices_sim = observation_building(sim_prices, sim_times, C, num_departures, dcps)
        transformed_prices_all_simulations.append(transformed_prices_sim)

    epochs = 100
    batch_size = 128
    bid_price_models = []
    scalers = []
    for sim_prices in transformed_prices_all_simulations:
        model, scaler = train_bid_price_model(sim_prices, dcps, epochs, batch_size)
        bid_price_models.append(model)
        scalers.append(scaler)

    predicted_bid_prices_all_simulations = []
    for model, scaler in zip(bid_price_models, scalers):
        predicted_bid_prices = predict_bid_prices(model, scaler, dcps, C)
        predicted_bid_prices_all_simulations.append(predicted_bid_prices)

    interpolated_bid_prices_all_simulations = []
    for predicted_bid_prices in predicted_bid_prices_all_simulations:
        interpolated_bid_prices = interpolate_bid_prices(predicted_bid_prices, dcps, days_prior)
        interpolated_bid_prices_all_simulations.append(interpolated_bid_prices)

    # Get bid prices for specific scenarios
    sim_index = 0
    days_prior_values = [6, 7, 8, 9, 10, 11]
    remaining_capacity = 3
    for day in days_prior_values:
        bid_price = get_bid_price(sim_index, day, remaining_capacity, interpolated_bid_prices_all_simulations)
        print(f"Bid price for simulation {sim_index}, {day} days prior, and remaining capacity {remaining_capacity}: {bid_price}")

    remaining_capacity_values = [1, 2, 3, 4]
    for capacity in remaining_capacity_values:
        bid_price = get_bid_price(sim_index, 10, capacity, interpolated_bid_prices_all_simulations)
        print(f"Bid price for simulation {sim_index}, 10 days prior, and remaining capacity {capacity}: {bid_price}")

    # Visualize predicted bid prices
    visualize_predicted_bid_prices(predicted_bid_prices_all_simulations[0], len(dcps), C)

    # Visualize bid price proxies
    visualize_bid_price_proxies(transformed_prices_all_simulations[0], len(transformed_prices_all_simulations[0]), len(dcps), C)

    # Generate optimal bid prices and interpolated bid prices for each simulation
    dp_bid_prices_all_simulations = []
    dd_bid_prices_all_simulations = []
    for i in range(num_simulations):
        def lambda_func_constant(t):
            return lambda_vals[i]
        
        _, dp_bid_prices, _, _ = optimal_bid_prices(C, T, p0_func, pmax_func, lambda_func_constant, Pw_func, delta_t)
        dp_bid_prices_all_simulations.append(dp_bid_prices)
        
        dd_bid_prices = interpolate_bid_prices(predicted_bid_prices_all_simulations[i], dcps, days_prior)
        dd_bid_prices_all_simulations.append(dd_bid_prices)

    # Generate future data
    dp_load_factors_all, dd_load_factors_all, dp_revenues_all, dd_revenues_all = generate_future_data_baseline(
        C, T, lambda_vals, delta_t, num_simulations, num_departures, dp_bid_prices_all_simulations, dd_bid_prices_all_simulations, seed
    )

    # Calculate revenue gap and load factor gap for each simulation
    revenue_gaps = [(dd_revenues_all[i] - dp_revenues_all[i]) / dp_revenues_all[i] for i in range(num_simulations)]
    load_factor_gaps = [dd_load_factors_all[i] - dp_load_factors_all[i] for i in range(num_simulations)]

    # Plot Figure 5
    plt.figure(figsize=(8, 6))
    plt.scatter(dp_load_factors_all, revenue_gaps)
    z = np.polyfit(dp_load_factors_all, revenue_gaps, 1)
    p = np.poly1d(z)
    plt.plot(dp_load_factors_all, p(dp_load_factors_all), "r--")
    plt.xlabel('Average Optimal Load Factor')
    plt.ylabel('Average Relative Revenue Gap from Optimal')
    plt.title('Average Relative Revenue Gap vs. Average Optimal Load Factor')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
    plt.grid(True)
    plt.show()

    # Plot Figure 6
    plt.figure(figsize=(8, 6))
    plt.scatter(dp_load_factors_all, load_factor_gaps)
    z = np.polyfit(dp_load_factors_all, load_factor_gaps, 2)
    p = np.poly1d(z)
    plt.plot(dp_load_factors_all, p(dp_load_factors_all), "r--")
    plt.xlabel('Average Optimal Load Factor')
    plt.ylabel('Average Load Factor Gap from Optimal')
    plt.title('Average Load Factor Gap vs. Average Optimal Load Factor')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
