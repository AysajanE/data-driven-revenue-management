# utils.py

import numpy as np
from scipy.optimize import minimize_scalar

import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value for the random number generator.
    """
    np.random.seed(seed)

def compute_optimal_bid_prices(remaining_capacity, t, lambda_func, delta_t):
    """
    Compute the optimal bid prices using dynamic programming.

    Parameters:
    remaining_capacity (int): The remaining capacity (number of seats).
    t (int): The current time to departure.
    lambda_val (float): The arrival rate parameter.
    delta_t (float): The time increment.

    Returns:
    numpy.ndarray: The optimal bid prices array.
    """
    _, b_star, _, _ = optimal_bid_prices(remaining_capacity, t, lambda_func, delta_t)
    if b_star.ndim != 2:
        raise ValueError(f"b_star should be a 2D array but got shape {b_star.shape}")
    return b_star

def compute_optimal_price(t, b_star, remaining_capacity):
    """
    Compute the optimal price for a given time and bid price.

    Parameters:
    t (int): The current time to departure.
    b_star (numpy.ndarray): The bid prices array.
    remaining_capacity (int): The remaining capacity (number of seats).

    Returns:
    float: The optimal price.
    """
    if b_star.ndim != 2:
        raise ValueError(f"b_star should be a 2D array but got shape {b_star.shape}")
    p_star = max(p0_func(t), alpha_func(t) + b_star[remaining_capacity, t-1])
    return p_star

def simulate_customer_arrivals(lambda_val, delta_t):
    """
    Simulate customer arrivals using a Poisson process.

    Parameters:
    lambda_val (float): The arrival rate parameter.
    delta_t (float): The time increment.

    Returns:
    int: The number of customer arrivals.
    """
    return np.random.poisson(lambda_val * delta_t)

def simulate_purchase_decision(p_star, t):
    """
    Simulate a customer's purchase decision.

    Parameters:
    p_star (float): The optimal price.
    t (int): The current time to departure.

    Returns:
    bool: True if the customer decides to purchase, False otherwise.
    """
    purchase_prob = Pw_func(p_star, t)
    return np.random.rand() < purchase_prob

def simulate_departure_training(args):
    """
    Simulate the booking process during the training phase.

    Parameters:
    args (tuple): A tuple containing (C, T, lambda_val, delta_t, seed):
        - C (int): Capacity (number of seats).
        - T (int): Booking horizon (i.e., Total time to departure).
        - lambda_val (float): Arrival rate parameter.
        - delta_t (float): Time increment.
        - seed (int): Seed value for the random number generator.

    Returns:
    tuple: A tuple containing (departure_prices, departure_times, departure_booked_seats):
        - departure_prices (list): List of booked prices.
        - departure_times (list): List of booking times.
        - departure_booked_seats (int): Total number of booked seats.
    """
    C, T, lambda_val, delta_t, seed = args
    
    set_random_seed(seed)
    
    departure_prices = []
    departure_times = []
    departure_booked_seats = 0
    
    remaining_capacity = C
    t = T

    # Define a lambda function to pass to optimal_bid_prices
    lambda_func = lambda t_idx: lambda_val

    while remaining_capacity > 0 and t > 0:
        b_star = compute_optimal_bid_prices(remaining_capacity, t, lambda_func, delta_t)
        p_star = compute_optimal_price(t, b_star, remaining_capacity)
        
        num_arrivals = simulate_customer_arrivals(lambda_val, delta_t)
        for _ in range(num_arrivals):
            if simulate_purchase_decision(p_star, t):
                departure_prices.append(p_star)
                departure_times.append(t)
                remaining_capacity -= 1
                departure_booked_seats += 1
                if remaining_capacity == 0:
                    break

        t -= delta_t
    
    return departure_prices, departure_times, departure_booked_seats

def initialize_arrays(C, T):
    """
    Initialize arrays to store data for the booking process.

    Parameters:
    C (int): Capacity (number of seats).
    T (int): Booking horizon (i.e., Total time to departure).

    Returns:
    dict: A dictionary containing initialized arrays and values.
    """
    return {
        'dp_prices': [],
        'dd_prices': [],
        'dp_departure_times': [],
        'dd_departure_times': [],
        'dp_booked_seats': 0,
        'dd_booked_seats': 0,
        'dp_remaining_capacity': C,
        'dd_remaining_capacity': C,
        'time': T
    }

def update_departure_data(data, prices, times, capacity, booked_seats, price, time):
    """
    Update departure data arrays with new booking information.

    Parameters:
    data (dict): The data dictionary containing arrays and values.
    prices (list): The list of prices to be updated.
    times (list): The list of booking times to be updated.
    capacity (int): The remaining capacity (number of seats).
    booked_seats (int): The number of booked seats.
    price (float): The booking price.
    time (int): The booking time.

    Returns:
    tuple: Updated capacity and booked seats.
    """
    prices.append(price)
    times.append(time)
    capacity -= 1
    booked_seats += 1
    return capacity, booked_seats

# def simulate_departure_testing_baseline(args, dp_bid_prices, dd_bid_prices):
#     """
#     Simulate the booking process during the testing phase with baseline bid prices.

#     Parameters:
#     args (tuple): A tuple containing (C, T, lambda_val, delta_t, seed):
#         - C (int): Capacity (number of seats).
#         - T (int): Total time to departure.
#         - lambda_val (float): Arrival rate parameter.
#         - delta_t (float): Time increment.
#         - seed (int): Seed value for the random number generator.
#     dp_bid_prices (numpy.ndarray): Array of dynamic programming bid prices.
#     dd_bid_prices (numpy.ndarray): Array of data-driven bid prices.

#     Returns:
#     tuple: A tuple containing (dp_prices, dd_prices, dp_departure_times, dd_departure_times, dp_booked_seats, dd_booked_seats):
#         - dp_prices (list): List of dynamic programming booked prices.
#         - dd_prices (list): List of data-driven booked prices.
#         - dp_departure_times (list): List of dynamic programming booking times.
#         - dd_departure_times (list): List of data-driven booking times.
#         - dp_booked_seats (int): Total number of dynamic programming booked seats.
#         - dd_booked_seats (int): Total number of data-driven booked seats.
#     """
#     C, T, lambda_val, delta_t, seed = args
    
#     set_random_seed(seed)
    
#     data = initialize_arrays(C, T)

#     while (data['dp_remaining_capacity'] > 0 or data['dd_remaining_capacity'] > 0) and data['time'] > 0:
#         num_arrivals = simulate_customer_arrivals(lambda_val, delta_t)
        
#         for _ in range(num_arrivals):
#             if data['dp_remaining_capacity'] > 0:
#                 dp_b_star = dp_bid_prices[data['dp_remaining_capacity'], data['time']-1]
#                 dp_p_star = compute_optimal_price(data['time'], dp_b_star, data['dp_remaining_capacity'])
                
#                 if simulate_purchase_decision(dp_p_star, data['time']):
#                     data['dp_remaining_capacity'], data['dp_booked_seats'] = update_departure_data(
#                         data, data['dp_prices'], data['dp_departure_times'], data['dp_remaining_capacity'], data['dp_booked_seats'], dp_p_star, data['time']
#                     )
            
#             if data['dd_remaining_capacity'] > 0:
#                 dd_b_star = dd_bid_prices[data['dd_remaining_capacity'], data['time']-1]
#                 dd_p_star = compute_optimal_price(data['time'], dd_b_star, data['dd_remaining_capacity'])
                
#                 if simulate_purchase_decision(dd_p_star, data['time']):
#                     data['dd_remaining_capacity'], data['dd_booked_seats'] = update_departure_data(
#                         data, data['dd_prices'], data['dd_departure_times'], data['dd_remaining_capacity'], data['dd_booked_seats'], dd_p_star, data['time']
#                     )
        
#         data['time'] -= delta_t

#     return data['dp_prices'], data['dd_prices'], data['dp_departure_times'], data['dd_departure_times'], data['dp_booked_seats'], data['dd_booked_seats']

def simulate_departure_testing_baseline(args, dp_bid_prices, dd_bid_prices):
    """
    Simulate the booking process during the testing phase with baseline bid prices.

    Parameters:
    args (tuple): A tuple containing (C, T, lambda_val, delta_t, seed):
        - C (int): Capacity (number of seats).
        - T (int): Total time to departure.
        - lambda_val (float): Arrival rate parameter.
        - delta_t (float): Time increment.
        - seed (int): Seed value for the random number generator.
    dp_bid_prices (numpy.ndarray): Array of dynamic programming bid prices.
    dd_bid_prices (numpy.ndarray): Array of data-driven bid prices.

    Returns:
    tuple: A tuple containing (dp_prices, dd_prices, dp_departure_times, dd_departure_times, dp_booked_seats, dd_booked_seats):
        - dp_prices (list): List of dynamic programming booked prices.
        - dd_prices (list): List of data-driven booked prices.
        - dp_departure_times (list): List of dynamic programming booking times.
        - dd_departure_times (list): List of data-driven booking times.
        - dp_booked_seats (int): Total number of dynamic programming booked seats.
        - dd_booked_seats (int): Total number of data-driven booked seats.
    """
    C, T, lambda_val, delta_t, seed = args
    
    set_random_seed(seed)
    
    data = initialize_arrays(C, T)

    while (data['dp_remaining_capacity'] > 0 or data['dd_remaining_capacity'] > 0) and data['time'] > 0:
        num_arrivals = simulate_customer_arrivals(lambda_val, delta_t)
        
        for _ in range(num_arrivals):
            if data['dp_remaining_capacity'] > 0:
                dp_b_star = dp_bid_prices[data['dp_remaining_capacity'] - 1, data['time'] - 1]
                dp_p_star = compute_optimal_price(data['time'], dp_b_star, data['dp_remaining_capacity'])
                
                if simulate_purchase_decision(dp_p_star, data['time']):
                    data['dp_remaining_capacity'], data['dp_booked_seats'] = update_departure_data(
                        data, data['dp_prices'], data['dp_departure_times'], data['dp_remaining_capacity'], data['dp_booked_seats'], dp_p_star, data['time']
                    )
            
            if data['dd_remaining_capacity'] > 0:
                # dd_b_star = dd_bid_prices[data['dd_remaining_capacity'] - 1, data['time'] - 1]
                dd_b_star = dd_bid_prices[data['time'] - 1, data['dd_remaining_capacity'] - 1]
                dd_p_star = compute_optimal_price(data['time'], dd_b_star, data['dd_remaining_capacity'])
                
                if simulate_purchase_decision(dd_p_star, data['time']):
                    data['dd_remaining_capacity'], data['dd_booked_seats'] = update_departure_data(
                        data, data['dd_prices'], data['dd_departure_times'], data['dd_remaining_capacity'], data['dd_booked_seats'], dd_p_star, data['time']
                    )
        
        data['time'] -= delta_t

    return data['dp_prices'], data['dd_prices'], data['dp_departure_times'], data['dd_departure_times'], data['dp_booked_seats'], data['dd_booked_seats']

def simulate_departure_testing_robustness(args, opt_bid_prices, bench_bid_prices, dd_bid_prices):
    """
    Simulate the booking process during the testing phase with robustness bid prices.

    Parameters:
    args (tuple): A tuple containing (C, T, lambda_val, delta_t, seed):
        - C (int): Capacity (number of seats).
        - T (int): Booking horizon (i.e., Total time to departure).
        - lambda_val (float): Arrival rate parameter.
        - delta_t (float): Time increment.
        - seed (int): Seed value for the random number generator.
    opt_bid_prices (numpy.ndarray): Array of optimal bid prices.
    bench_bid_prices (numpy.ndarray): Array of benchmark bid prices.
    dd_bid_prices (numpy.ndarray): Array of data-driven bid prices.

    Returns:
    tuple: Contains the results of the simulation:
        - opt_prices (list): List of optimal booked prices.
        - opt_departure_times (list): List of optimal booking times.
        - bench_prices (list): List of benchmark booked prices.
        - bench_departure_times (list): List of benchmark booking times.
        - dd_prices (list): List of data-driven booked prices.
        - dd_departure_times (list): List of data-driven booking times.
        - opt_load_factor (float): Load factor for optimal approach.
        - bench_load_factor (float): Load factor for benchmark approach.
        - dd_load_factor (float): Load factor for data-driven approach.
        - opt_revenue (float): Total revenue for optimal approach.
        - bench_revenue (float): Total revenue for benchmark approach.
        - dd_revenue (float): Total revenue for data-driven approach.
    """
    C, T, lambda_val, delta_t, seed = args
    
    set_random_seed(seed)
    
    opt_data = initialize_arrays(C, T)
    bench_data = initialize_arrays(C, T)
    dd_data = initialize_arrays(C, T)

    while (opt_data['remaining_capacity'] > 0 or bench_data['remaining_capacity'] > 0 or dd_data['remaining_capacity'] > 0) and opt_data['time'] > 0:
        num_arrivals = simulate_customer_arrivals(lambda_val, delta_t)
        
        for _ in range(num_arrivals):
            if opt_data['remaining_capacity'] > 0:
                b_star = opt_bid_prices[opt_data['remaining_capacity']-1, opt_data['time']-1]
                p_opt = max(p0_func(opt_data['time']), alpha_func(opt_data['time']) + b_star)
                if simulate_purchase_decision(p_opt, opt_data['time']):
                    opt_data['remaining_capacity'], opt_data['booked_seats'] = update_departure_data(
                        opt_data, opt_data['prices'], opt_data['departure_times'], opt_data['remaining_capacity'], opt_data['booked_seats'], p_opt, opt_data['time']
                    )
            
            if bench_data['remaining_capacity'] > 0:
                b_bench = bench_bid_prices[bench_data['remaining_capacity']-1, bench_data['time']-1]
                p_bench = max(p0_func(bench_data['time']), alpha_func(bench_data['time']) + b_bench)
                if simulate_purchase_decision(p_bench, bench_data['time']):
                    bench_data['remaining_capacity'], bench_data['booked_seats'] = update_departure_data(
                        bench_data, bench_data['prices'], bench_data['departure_times'], bench_data['remaining_capacity'], bench_data['booked_seats'], p_bench, bench_data['time']
                    )
            
            if dd_data['remaining_capacity'] > 0:
                b_data_driven = dd_bid_prices[dd_data['time']-1][dd_data['remaining_capacity']-1]
                p_dd = max(p0_func(dd_data['time']), alpha_func(dd_data['time']) + b_data_driven)
                if simulate_purchase_decision(p_dd, dd_data['time']):
                    dd_data['remaining_capacity'], dd_data['booked_seats'] = update_departure_data(
                        dd_data, dd_data['prices'], dd_data['departure_times'], dd_data['remaining_capacity'], dd_data['booked_seats'], p_dd, dd_data['time']
                    )

        opt_data['time'] -= delta_t
        bench_data['time'] -= delta_t
        dd_data['time'] -= delta_t
    
    opt_load_factor = opt_data['booked_seats'] / C
    bench_load_factor = bench_data['booked_seats'] / C
    dd_load_factor = dd_data['booked_seats'] / C
    opt_revenue = sum(opt_data['prices'])
    bench_revenue = sum(bench_data['prices'])
    dd_revenue = sum(dd_data['prices'])
    
    return (opt_data['prices'], opt_data['departure_times'], bench_data['prices'], bench_data['departure_times'], 
            dd_data['prices'], dd_data['departure_times'], opt_load_factor, bench_load_factor, 
            dd_load_factor, opt_revenue, bench_revenue, dd_revenue)

# def optimal_bid_prices(C, T, lambda_func, delta_t):
#     """
#     Compute the optimal bid prices using dynamic programming.

#     Parameters:
#     C (int): Capacity (number of seats).
#     T (int): Total time to departure.
#     lambda_func (function): Function to get the arrival rate.
#     delta_t (float): Time increment.

#     Returns:
#     tuple: Containing V, b_star, p_star, and p_star_alternative.
#     """
#     def initialize_value_function_arrays(C, T):
#         """
#         Initialize value function arrays for the dynamic programming problem.

#         Parameters:
#         C (int): Capacity (number of seats).
#         T (int): Total time to departure.

#         Returns:
#         tuple: Containing V, p_star, b_star, and p_star_alternative arrays.
#         """
#         V = np.zeros((C+1, T+1))  # Value function for each state (x,t)
#         p_star = np.zeros((C+1, T+1))  # Optimal price for each state (x,t)
#         b_star = np.zeros((C+1, T+1))  # Bid price for each state (x,t)
#         p_star_alternative = np.zeros((C+1, T+1))  # Optimal price using alternative method

#         # Define the boundary conditions for value function
#         V[:, 0] = 0  # V(x, 0) = 0 for all x
#         V[0, :] = 0  # V(0, t) = 0 for all t

#         return V, p_star, b_star, p_star_alternative

#     def solve_dynamic_programming(V, p_star, b_star, p_star_alternative, C, T, lambda_func, delta_t):
#         """
#         Solve the dynamic programming problem using backward induction.

#         Parameters:
#         V (np.ndarray): Value function array.
#         p_star (np.ndarray): Optimal price array.
#         b_star (np.ndarray): Bid price array.
#         p_star_alternative (np.ndarray): Alternative optimal price array.
#         C (int): Capacity (number of seats).
#         T (int): Total time to departure.
#         lambda_func (function): Function to get the arrival rate.
#         delta_t (float): Time increment.

#         Returns:
#         tuple: Updated V, p_star, b_star, and p_star_alternative arrays.
#         """
#         for t_idx in range(1, T+1):
#             for x_idx in range(1, C+1):
#                 def revenue_func(p):
#                     lambda_val = lambda_func(t_idx)
#                     return -1 * lambda_val * delta_t * Pw_func(p, t_idx) * (p - (V[x_idx, t_idx-1] - V[x_idx-1, t_idx-1]))

#                 bounds = (p0_func(t_idx), pmax_func(t_idx))
#                 result = minimize_scalar(revenue_func, bounds=bounds, method='bounded')
#                 optimal_price = result.x
#                 max_val = -1 * result.fun + V[x_idx, t_idx-1]

#                 V[x_idx, t_idx] = max_val
#                 p_star[x_idx, t_idx] = optimal_price
#                 b_star[x_idx, t_idx] = V[x_idx, t_idx] - V[x_idx-1, t_idx]
#                 p_star_alternative[x_idx, t_idx] = max(p0_func(t_idx), alpha_func(t_idx) + b_star[x_idx, t_idx-1])

#         return V, p_star, b_star, p_star_alternative

#     V, p_star, b_star, p_star_alternative = initialize_value_function_arrays(C, T)
#     V, p_star, b_star, p_star_alternative = solve_dynamic_programming(V, p_star, b_star, p_star_alternative, C, T, lambda_func, delta_t)
#     return V, b_star, p_star, p_star_alternative

def optimal_bid_prices(C, T, lambda_func, delta_t):
    """
    Compute the optimal bid prices using dynamic programming.

    Parameters:
    C (int): Capacity (number of seats).
    T (int): Total time to departure.
    lambda_func (function): Function to get the arrival rate.
    delta_t (float): Time increment.

    Returns:
    tuple: Containing V, b_star, p_star, and p_star_alternative.
    """
    def initialize_value_function_arrays(C, T):
        """
        Initialize value function arrays for the dynamic programming problem.

        Parameters:
        C (int): Capacity (number of seats).
        T (int): Total time to departure.

        Returns:
        tuple: Containing V, p_star, b_star, and p_star_alternative arrays.
        """
        V = np.zeros((C + 1, T + 1))  # Value function for each state (x,t)
        p_star = np.zeros((C + 1, T + 1))  # Optimal price for each state (x,t)
        b_star = np.zeros((C + 1, T + 1))  # Bid price for each state (x,t)
        p_star_alternative = np.zeros((C + 1, T + 1))  # Optimal price using alternative method

        # Define the boundary conditions for value function
        V[:, 0] = 0  # V(x, 0) = 0 for all x
        V[0, :] = 0  # V(0, t) = 0 for all t

        return V, p_star, b_star, p_star_alternative

    def solve_dynamic_programming(V, p_star, b_star, p_star_alternative, C, T, lambda_func, delta_t):
        """
        Solve the dynamic programming problem using backward induction.

        Parameters:
        V (np.ndarray): Value function array.
        p_star (np.ndarray): Optimal price array.
        b_star (np.ndarray): Bid price array.
        p_star_alternative (np.ndarray): Alternative optimal price array.
        C (int): Capacity (number of seats).
        T (int): Total time to departure.
        lambda_func (function): Function to get the arrival rate.
        delta_t (float): Time increment.

        Returns:
        tuple: Updated V, p_star, b_star, and p_star_alternative arrays.
        """
        for t_idx in range(1, T + 1):
            for x_idx in range(1, C + 1):
                def revenue_func(p):
                    lambda_val = lambda_func(t_idx)
                    return -1 * lambda_val * delta_t * Pw_func(p, t_idx) * (p - (V[x_idx, t_idx - 1] - V[x_idx - 1, t_idx - 1]))

                bounds = (p0_func(t_idx), pmax_func(t_idx))
                result = minimize_scalar(revenue_func, bounds=bounds, method='bounded')
                optimal_price = result.x
                max_val = -1 * result.fun + V[x_idx, t_idx - 1]

                V[x_idx, t_idx] = max_val
                p_star[x_idx, t_idx] = optimal_price
                b_star[x_idx, t_idx] = V[x_idx, t_idx] - V[x_idx - 1, t_idx]
                p_star_alternative[x_idx, t_idx] = max(p0_func(t_idx), alpha_func(t_idx) + b_star[x_idx, t_idx - 1])

        return V, p_star, b_star, p_star_alternative

    V, p_star, b_star, p_star_alternative = initialize_value_function_arrays(C, T)
    V, p_star, b_star, p_star_alternative = solve_dynamic_programming(V, p_star, b_star, p_star_alternative, C, T, lambda_func, delta_t)
    return V, b_star, p_star, p_star_alternative


def p0_func(t):
    """
    Returns the minimum price.

    Parameters:
    t (int): The current time to departure.

    Returns:
    int: The minimum price.
    """
    return 100  # Minimum price

def alpha_func(t):
    """
    Returns the mean willingness-to-pay of customers.

    Parameters:
    t (int): The current time to departure.

    Returns:
    int: The mean willingness-to-pay.
    """
    return 50  # Mean willingness-to-pay of customers

def pmax_func(t, lambda_val=3):
    """
    Returns the maximum price.

    Parameters:
    t (int): The current time to departure.
    lambda_val (float): The arrival rate parameter (default is 3).

    Returns:
    float: The maximum price.
    """
    pmax = p0_func(t) - alpha_func(t) * np.log(85 / (lambda_val * 300))
    pmax_rounded = np.ceil(pmax / 50) * 50  # Round up to the nearest integer greater than or equal to the floating-point number.
    # return int(pmax_rounded)
    return 500

def Pw_func(p, t):
    """
    Returns the exponential purchase probability.

    Parameters:
    p (float): The price.
    t (int): The current time to departure.

    Returns:
    float: The purchase probability.
    """
    return np.exp(-(p - p0_func(t)) / alpha_func(t))


def generate_historical_data(C, T, lambda_range, lambda_test_range, delta_t, num_simulations, num_departures, seed=None):
    """
    Generate historical data for simulations.

    Parameters:
    C (int): Capacity
    T (int): Time horizon
    lambda_range (list): Range for training arrival rate
    lambda_test_range (list): Range for test arrival rate
    delta_t (int): Time interval
    num_simulations (int): Number of simulations
    num_departures (int): Number of departure dates
    seed (int, optional): Seed value for random number generator

    Returns:
    tuple: Historical data arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    prices = []
    times = []
    total_booked_seats = []
    load_factors = []
    total_revenues = []
    lambda_vals = []
    lambda_test_vals = []

    pool = mp.Pool()

    for i in range(num_simulations):
        lambda_val = np.random.uniform(lambda_range[0], lambda_range[1])
        lambda_vals.append(lambda_val)

        lambda_test_val = np.random.uniform(lambda_test_range[0], lambda_test_range[1])
        lambda_test_vals.append(lambda_test_val)
        
        args_list = [(C, T, lambda_val, delta_t, seed+i*num_departures+j) for j in range(num_departures)]
        results = pool.map(simulate_departure_training, args_list)
        
        simulation_prices, simulation_times, simulation_booked_seats = zip(*results)

        simulation_load_factors = [booked_seats / C for booked_seats in simulation_booked_seats]
        simulation_revenues = [sum(prices) for prices in simulation_prices]
        
        prices.append(simulation_prices)
        times.append(simulation_times)
        total_booked_seats.append(simulation_booked_seats)
        load_factors.append(simulation_load_factors)
        total_revenues.append(simulation_revenues)

    pool.close()

    return prices, times, total_booked_seats, load_factors, total_revenues, lambda_vals, lambda_test_vals

def observation_building(prices, times, capacity, flights, dcps):
    """
    Build observation matrix for each flight.

    Parameters:
    prices (list): List of historical prices for each flight
    times (list): List of time-to-departure for each flight
    capacity (int): Capacity of the flight
    flights (int): Number of flights (departures)
    dcps (list): List of data collection points (DCPs)

    Returns:
    list: List of transformed price matrices for each flight
    """
    transformed_prices = []

    for i in range(flights):
        price_matrix = []
        for dcp in dcps:
            filtered_data = [(p, t) for p, t in zip(prices[i], times[i]) if t <= dcp]
            filtered_prices = [p for p, _ in filtered_data]
            sorted_prices = sorted(filtered_prices, reverse=True)
            padded_prices = sorted_prices + [0] * (capacity - len(sorted_prices))
            price_matrix.append(padded_prices)
        transformed_prices.append(price_matrix)

    return transformed_prices

def prepare_data(transformed_prices, dcps):
    """
    Prepare input feature matrix X and target variable vector Y for the NN model.

    Parameters:
    transformed_prices (list): List of transformed price matrices for each flight
    dcps (list): List of data collection points (DCPs)

    Returns:
    tuple: A tuple containing the input feature matrix X and target variable vector Y
    """
    X = []
    Y = []

    for price_matrix in transformed_prices:
        num_dcps, num_prices = len(price_matrix), len(price_matrix[0])
        for i in range(num_dcps):
            for j in range(num_prices):
                X.append([dcps[i], j + 1])
                Y.append(price_matrix[i][j])

    return X, Y

def create_nn_model(input_shape):
    """
    Create a neural network model for bid price estimation.

    Parameters:
    input_shape (tuple): Shape of the input features

    Returns:
    tf.keras.Model: Compiled neural network model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='softplus')
    ])
    return model

def train_nn_model(X, Y, epochs, batch_size, validation_split=0.2, verbose=0):
    """
    Train the neural network model.

    Parameters:
    X (np.ndarray): Input feature matrix
    Y (np.ndarray): Target variable vector
    epochs (int): Number of epochs for training
    batch_size (int): Batch size for training
    validation_split (float): Fraction of data to be used as validation set
    verbose (int): Verbosity mode

    Returns:
    tf.keras.Model: Trained neural network model
    """
    input_shape = (X.shape[1],)
    model = create_nn_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=verbose)
    return model

def train_bid_price_model(transformed_prices, dcps, epochs, batch_size):
    """
    Train a neural network model to estimate bid prices.

    Parameters:
    transformed_prices (list): List of transformed price matrices for each flight
    dcps (list): List of data collection points (DCPs)
    epochs (int): Number of epochs for training
    batch_size (int): Batch size for training

    Returns:
    tuple: A tuple containing the trained model and the scaler used for normalization
    """
    X, Y = prepare_data(transformed_prices, dcps)
    X = np.array(X)
    Y = np.array(Y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = train_nn_model(X_scaled, Y, epochs, batch_size)
    return model, scaler

def predict_bid_prices(model, scaler, dcps, capacity):
    """
    Predict bid prices using the trained model.

    Parameters:
    model (tf.keras.Model): Trained neural network model
    scaler (StandardScaler): Scaler used for normalization
    dcps (list): List of data collection points (DCPs)
    capacity (int): Maximum remaining capacity

    Returns:
    np.ndarray: Predicted bid prices for all combinations of DCPs and remaining capacity
    """
    X_pred = []
    for dcp in dcps:
        for cap in range(1, capacity + 1):
            X_pred.append([dcp, cap])
    X_pred = np.array(X_pred)
    X_pred_scaled = scaler.transform(X_pred)
    bid_prices = model.predict(X_pred_scaled)
    bid_prices = bid_prices.reshape(len(dcps), capacity)
    return bid_prices

def interpolate_bid_prices(bid_prices, dcps, days_prior):
    """
    Interpolate bid prices between DCPs to get a separate bid price vector for each day prior to departure.

    Parameters:
    bid_prices (np.ndarray): Predicted bid prices for all combinations of DCPs and remaining capacity
    dcps (list): List of data collection points (DCPs)
    days_prior (list): List of days prior to departure

    Returns:
    list: Interpolated bid price vectors for each day prior to departure
    """
    interpolated_bid_prices = []
    min_dcp = min(dcps)
    max_dcp = max(dcps)
    
    for day in days_prior:
        if day in dcps:
            dcp_index = dcps.index(day)
            interpolated_bid_prices.append(bid_prices[dcp_index])
        else:
            if day < min_dcp:
                interpolated_bid_prices.append(bid_prices[0])
            elif day > max_dcp:
                interpolated_bid_prices.append(bid_prices[-1])
            else:
                prev_dcp = max(dcp for dcp in dcps if dcp < day)
                next_dcp = min(dcp for dcp in dcps if dcp > day)
                prev_index = dcps.index(prev_dcp)
                next_index = dcps.index(next_dcp)
                alpha = (day - prev_dcp) / (next_dcp - prev_dcp)
                interpolated_prices = (1 - alpha) * bid_prices[prev_index] + alpha * bid_prices[next_index]
                interpolated_bid_prices.append(interpolated_prices)

    return interpolated_bid_prices


def get_bid_price(sim_index, days_prior_value, remaining_capacity, interpolated_bid_prices_all_simulations):
    """
    Get the bid price for a specific simulation scenario, days prior-to-departure, and remaining capacity.

    Parameters:
    sim_index (int): Index of the simulation scenario.
    days_prior_value (int): Days prior-to-departure value.
    remaining_capacity (int): Remaining capacity value.
    interpolated_bid_prices_all_simulations (list): List of interpolated bid price vectors for each simulation scenario.

    Returns:
    float: Bid price for the specified simulation scenario, days prior-to-departure, and remaining capacity.
    """
    if days_prior_value < 0 or days_prior_value >= len(interpolated_bid_prices_all_simulations[sim_index]):
        raise ValueError(f"Invalid days_prior_value: {days_prior_value}")

    if remaining_capacity < 1 or remaining_capacity > len(interpolated_bid_prices_all_simulations[sim_index][days_prior_value]):
        raise ValueError(f"Invalid remaining_capacity: {remaining_capacity}")

    return interpolated_bid_prices_all_simulations[sim_index][days_prior_value][remaining_capacity - 1]

def visualize_predicted_bid_prices(predicted_bid_prices, num_dcps, capacity):
    """
    Visualize the predicted bid prices.

    Parameters:
    predicted_bid_prices (np.ndarray): Predicted bid prices for all combinations of DCPs and remaining capacity.
    num_dcps (int): Number of data collection points (DCPs).
    capacity (int): Maximum remaining capacity.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Predicted Bid Prices", fontsize=16)

    im = ax.imshow(predicted_bid_prices.T, cmap='viridis', aspect='auto')
    ax.set_ylabel("Remaining Capacity")
    ax.set_xlabel("Days Before Departure")

    y_ticks = np.arange(0, capacity, 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks + 1)

    x_ticks = np.arange(0, num_dcps, 3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(num_dcps - x_ticks)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Bid Price", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()

def visualize_bid_price_proxies(bid_price_proxies, num_flights, num_dcps, capacity):
    """
    Visualize bid price proxies.

    Parameters:
    bid_price_proxies (list): List of bid price proxies for each flight.
    num_flights (int): Number of flights (departures).
    num_dcps (int): Number of data collection points (DCPs).
    capacity (int): Maximum remaining capacity.
    """
    num_rows = (num_flights - 1) // 4 + 1
    num_cols = min(num_flights, 4)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))

    vmin = np.min([np.min(proxies) for proxies in bid_price_proxies])
    vmax = np.max([np.max(proxies) for proxies in bid_price_proxies])

    for flight in range(num_flights):
        flight_bid_price_proxies = np.array(bid_price_proxies[flight])
        
        row = flight // num_cols
        col = flight % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        im = ax.imshow(flight_bid_price_proxies.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"Bid Price Proxies - Flight {flight + 1}")
        ax.set_xlabel("Days Before Departure")
        ax.set_ylabel("Remaining Capacity")

        xticks = np.linspace(0, num_dcps - 1, 10, dtype=int)
        xticklabels = [num_dcps - x for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        yticks = np.linspace(0, capacity, 10, dtype=int)
        yticklabels = [capacity - y for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Bid Price", rotation=-90, va="bottom")

    fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.4)
    plt.show()

def plot_bid_price_proxy(transformed_prices, predicted_bid_prices, optimal_dp_bid_prices, dbd_threshold, capacity, num_departures, total_time_horizon, colors=None):
    """
    Plot bid price proxies.

    Parameters:
    transformed_prices (list): List of transformed prices for each flight.
    predicted_bid_prices (np.ndarray): Predicted bid prices.
    optimal_dp_bid_prices (np.ndarray): Optimal DP bid prices.
    dbd_threshold (int): Days before departure threshold.
    capacity (int): Maximum remaining capacity.
    num_departures (int): Number of departures.
    total_time_horizon (int): Total time horizon.
    colors (list, optional): List of colors for plotting.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, num_departures))

    for flight_idx in range(num_departures):
        flight_prices = transformed_prices[flight_idx]
        filtered_flight_prices = flight_prices[total_time_horizon-dbd_threshold:]
        filtered_bid_price_proxies = [price for prices in filtered_flight_prices for price in prices]
        remaining_capacity = [remaining_cap for _ in range(len(filtered_flight_prices)) for remaining_cap in range(1, capacity+1)]
        ax.scatter(remaining_capacity, filtered_bid_price_proxies, color=colors[flight_idx], label=f'Flight {flight_idx+1}', alpha=0.7)

    filtered_predicted_bid_prices = predicted_bid_prices[total_time_horizon-dbd_threshold:, :]
    avg_predicted_bid_prices = np.mean(filtered_predicted_bid_prices, axis=0)
    remaining_capacity_levels = np.arange(1, capacity + 1)
    ax.plot(remaining_capacity_levels, avg_predicted_bid_prices, color='blue', label='Estimated Bid Prices')

    filtered_optimal_bid_prices = optimal_dp_bid_prices[:, :dbd_threshold]
    avg_optimal_bid_prices = np.mean(filtered_optimal_bid_prices, axis=1)
    ax.plot(remaining_capacity_levels, avg_optimal_bid_prices, color='red', label='Optimal Bid Prices')

    ax.set_xlabel('Remaining Capacity')
    ax.set_xlim(0, capacity + 1)
    ax.set_ylabel('Bid Price Proxy')
    ax.set_title(f'Bid Price Proxy (Days Before Departure <= {dbd_threshold})')
    plt.tight_layout()
    plt.show()

def generate_future_data_baseline(C, T, lambda_vals, delta_t, num_simulations, num_departures, optimal_bid_prices_all_simulations, interpolated_bid_prices_all_simulations, seed=None):
    """
    Generate future data for baseline simulations.

    Parameters:
    C (int): Capacity.
    T (int): Time horizon.
    lambda_vals (list): List of lambda values for each simulation.
    delta_t (float): Time increment.
    num_simulations (int): Number of simulations.
    num_departures (int): Number of departure dates.
    optimal_bid_prices_all_simulations (list): List of optimal bid prices for each simulation.
    interpolated_bid_prices_all_simulations (list): List of interpolated bid prices for each simulation.
    seed (int, optional): Seed value for random number generator.

    Returns:
    tuple: Lists of load factors and revenues for DP and DD approaches.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dp_load_factors_all = []
    dd_load_factors_all = []
    dp_revenues_all = []
    dd_revenues_all = []

    pool = mp.Pool()

    for i in range(num_simulations):
        lambda_val = lambda_vals[i]
        dp_bid_prices = optimal_bid_prices_all_simulations[i]
        dd_bid_prices = interpolated_bid_prices_all_simulations[i]
        args_list = [(C, T, lambda_val, delta_t, seed+i*num_departures+j) for j in range(num_departures)]
        results = pool.starmap(simulate_departure_testing_baseline, [(args, dp_bid_prices, dd_bid_prices) for args in args_list])
        
        dp_prices, dp_departure_times, dd_prices, dd_departure_times, dp_load_factors, dd_load_factors, dp_revenues, dd_revenues = zip(*results)
        avg_dp_load_factor = np.mean(dp_load_factors)
        avg_dd_load_factor = np.mean(dd_load_factors)
        avg_dp_revenue = np.mean(dp_revenues)
        avg_dd_revenue = np.mean(dd_revenues)
        
        dp_load_factors_all.append(avg_dp_load_factor)
        dd_load_factors_all.append(avg_dd_load_factor)
        dp_revenues_all.append(avg_dp_revenue)
        dd_revenues_all.append(avg_dd_revenue)

    pool.close()

    return dp_load_factors_all, dd_load_factors_all, dp_revenues_all, dd_revenues_all





