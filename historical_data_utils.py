import numpy as np
from scipy.optimize import minimize_scalar

def simulate_departure_training(args):
    C, T, lambda_val, delta_t, seed = args
    
    # Set the seed for the random number generator
    np.random.seed(seed)
    
    # Initialize arrays to store data for each departure
    departure_prices = []
    departure_times = []
    departure_booked_seats = 0
    
    # Simulate booking process
    remaining_capacity = C
    t = T

    while remaining_capacity > 0 and t > 0:
        # Compute optimal bid prices using the optimal_bid_prices function
        _, b_star, _, _ = optimal_bid_prices(remaining_capacity, t, lambda_val, delta_t)
        
        # Calculate the optimal price using equation (6)
        p_star = max(p0_func(t), alpha_func(t) + b_star[remaining_capacity, t-1])  # Be careful, b_star time index is t-1, not t.

        # Simulate customer arrivals using Poisson process
        num_arrivals = np.random.poisson(lambda_val * delta_t)
        for _ in range(num_arrivals):
            # Simulate purchase decision
            purchase_prob = Pw_func(p_star, t)
            if np.random.rand() < purchase_prob:
                departure_prices.append(p_star)
                departure_times.append(t)
                remaining_capacity -= 1
                departure_booked_seats += 1  # Increment booked seats counter
                if remaining_capacity == 0:
                    break

        t -= delta_t
    
    return departure_prices, departure_times, departure_booked_seats

def simulate_departure_testing_baseline(args, dp_bid_prices, dd_bid_prices):
    C, T, lambda_val, delta_t, seed = args
    
    # Set the seed for the random number generator
    np.random.seed(seed)
    
    # Initialize arrays to store data for each departure
    dp_prices = []
    dd_prices = []
    dp_departure_times = []
    dd_departure_times = []
    dp_booked_seats = 0
    dd_booked_seats = 0
    
    # Simulate booking process
    dp_remaining_capacity = C
    dd_remaining_capacity = C
    t = T

    while (dp_remaining_capacity > 0 or dd_remaining_capacity > 0) and t > 0:
        # Get the optimal bid price for the current remaining capacity and time-to-departure
        b_star = dp_bid_prices[dp_remaining_capacity-1, t-1]  # Adjust indices to match the data structure
        
        # Get the data-driven bid price for the current remaining capacity and time-to-departure
        b_data_driven = dd_bid_prices[t-1][dd_remaining_capacity-1]  # Adjust indices to match the data structure
        
        # Calculate the price using the optimal bid price (dynamic programming approach)
        p_dp = max(p0_func(t), alpha_func(t) + b_star)
        
        # Calculate the price using the data-driven bid price
        p_dd = max(p0_func(t), alpha_func(t) + b_data_driven)

        # Simulate customer arrivals using Poisson process
        num_arrivals = np.random.poisson(lambda_val * delta_t)
        for _ in range(num_arrivals):
            # Simulate purchase decision for dynamic programming approach
            if dp_remaining_capacity > 0:
                purchase_prob_dp = Pw_func(p_dp, t)
                if np.random.rand() < purchase_prob_dp:
                    dp_prices.append(p_dp)
                    dp_departure_times.append(t)
                    dp_remaining_capacity -= 1
                    dp_booked_seats += 1  # Increment booked seats counter for dynamic programming approach
                    if dp_remaining_capacity == 0:
                        break

            # Simulate purchase decision for data-driven approach
            if dd_remaining_capacity > 0:
                purchase_prob_dd = Pw_func(p_dd, t)
                if np.random.rand() < purchase_prob_dd:
                    dd_prices.append(p_dd)
                    dd_departure_times.append(t)
                    dd_remaining_capacity -= 1
                    dd_booked_seats += 1  # Increment booked seats counter for data-driven approach
                    if dd_remaining_capacity == 0:
                        break

        t -= delta_t
    
    dp_load_factor = dp_booked_seats / C
    dd_load_factor = dd_booked_seats / C
    dp_revenue = sum(dp_prices)
    dd_revenue = sum(dd_prices)
    
    return dp_prices, dp_departure_times, dd_prices, dd_departure_times, dp_load_factor, dd_load_factor, dp_revenue, dd_revenue

def simulate_departure_testing_robustness(args, opt_bid_prices, bench_bid_prices, dd_bid_prices):
    C, T, lambda_val, delta_t, seed = args
    
    # Set the seed for the random number generator
    np.random.seed(seed)
    
    # Initialize arrays to store data for each departure
    opt_prices = []
    bench_prices = []
    dd_prices = []
    
    opt_departure_times = []
    bench_departure_times = []
    dd_departure_times = []
    
    opt_booked_seats = 0
    bench_booked_seats = 0
    dd_booked_seats = 0
    
    # Simulate booking process
    opt_remaining_capacity = C
    bench_remaining_capacity = C
    dd_remaining_capacity = C
    t = T

    while (opt_remaining_capacity > 0 or bench_remaining_capacity > 0 or dd_remaining_capacity > 0) and t > 0:
        # Get the optimal bid price for the current remaining capacity and time-to-departure
        b_star = opt_bid_prices[opt_remaining_capacity-1, t-1]  # Adjust indices to match the data structure

        # Get the benchmark bid price for the current remaining capacity and time-to-departure
        b_bench = bench_bid_prices[bench_remaining_capacity-1, t-1]  # Adjust indices to match the data structure
        
        # Get the data-driven bid price for the current remaining capacity and time-to-departure
        b_data_driven = dd_bid_prices[t-1][dd_remaining_capacity-1]  # Adjust indices to match the data structure
        
        # Calculate the price using the optimal bid price (dynamic programming approach)
        p_opt = max(p0_func(t), alpha_func(t) + b_star)

        # Calculate the price using the benchmark bid price (dynamic programming approach)
        p_bench = max(p0_func(t), alpha_func(t) + b_bench)
        
        # Calculate the price using the data-driven bid price
        p_dd = max(p0_func(t), alpha_func(t) + b_data_driven)

        # Simulate customer arrivals using Poisson process
        # All three approaches experience the same arrival stream
        num_arrivals = np.random.poisson(lambda_val * delta_t)
        for _ in range(num_arrivals):
            # Simulate purchase decision for optimal dynamic programming approach
            if opt_remaining_capacity > 0:
                purchase_prob_opt = Pw_func(p_opt, t)
                if np.random.rand() < purchase_prob_opt:
                    opt_prices.append(p_opt)
                    opt_departure_times.append(t)
                    opt_remaining_capacity -= 1
                    opt_booked_seats += 1  # Increment booked seats counter for dynamic programming approach
                    if opt_remaining_capacity == 0:
                        break

            # Simulate purchase decision for benchmark dynamic programming approach
            if bench_remaining_capacity > 0:
                purchase_prob_bench = Pw_func(p_bench, t)
                if np.random.rand() < purchase_prob_bench:
                    bench_prices.append(p_bench)
                    bench_departure_times.append(t)
                    bench_remaining_capacity -= 1
                    bench_booked_seats += 1  # Increment booked seats counter for dynamic programming approach
                    if bench_remaining_capacity == 0:
                        break

            # Simulate purchase decision for data-driven approach
            if dd_remaining_capacity > 0:
                purchase_prob_dd = Pw_func(p_dd, t)
                if np.random.rand() < purchase_prob_dd:
                    dd_prices.append(p_dd)
                    dd_departure_times.append(t)
                    dd_remaining_capacity -= 1
                    dd_booked_seats += 1  # Increment booked seats counter for data-driven approach
                    if dd_remaining_capacity == 0:
                        break

        t -= delta_t
    
    opt_load_factor = opt_booked_seats / C
    bench_load_factor = bench_booked_seats / C
    dd_load_factor = dd_booked_seats / C
    opt_revenue = sum(opt_prices)
    bench_revenue = sum(bench_prices)
    dd_revenue = sum(dd_prices)
    
    return opt_prices, opt_departure_times, bench_prices, bench_departure_times, dd_prices, dd_departure_times, opt_load_factor, bench_load_factor, dd_load_factor, opt_revenue, bench_revenue, dd_revenue

def optimal_bid_prices(C, T, lambda_val, delta_t):
    # Step 1: Define state space
    X = np.arange(0, C+1)  # Remaining inventory levels
    t = np.arange(0, T+1)  # Time periods

    # Step 2: Initialize value function matrix, optimal price matrix, and bid price matrix
    V = np.zeros((C+1, T+1))  # V(x,t)
    p_star = np.zeros((C+1, T+1))  # Optimal price for each state (x,t)
    b_star = np.zeros((C+1, T+1))  # Bid price for each state (x,t)

    p_star_alternative = np.zeros((C+1, T+1))  # Optimal price for each state (x,t) using Equation (6)

    # Define the boundary conditions for value function
    V[:, 0] = 0  # V(x, 0) = 0 for all x
    V[0, :] = 0  # V(0, t) = 0 for all t

    # Step 3: Solve the dynamic programming problem using backward induction
    for t_idx in range(1, T+1):
        for x_idx in range(1, C+1):
            # Define the revenue function to be maximized
            def revenue_func(p):
                return -1 * lambda_val * delta_t * Pw_func(p, t_idx) * (p - (V[x_idx, t_idx-1] - V[x_idx-1, t_idx-1]))

            # Use golden-section search to find the optimal price
            bounds = (p0_func(t_idx), pmax_func(t_idx))
            result = minimize_scalar(revenue_func, bounds=bounds, method='bounded')
            optimal_price = result.x
            max_val = -1 * result.fun + V[x_idx, t_idx-1]

            V[x_idx, t_idx] = max_val
            p_star[x_idx, t_idx] = optimal_price
            b_star[x_idx, t_idx] = V[x_idx, t_idx] - V[x_idx-1, t_idx]

            # Calculate the optimal price using equation (6)
            p_star_alternative[x_idx, t_idx] = max(p0_func(t_idx), alpha_func(t_idx) + b_star[x_idx, t_idx-1])

    return V, b_star, p_star, p_star_alternative

def p0_func(t):
    return 100  # Minimum price

def alpha_func(t):
    return 50  # Mean willingness-to-pay of customers

def pmax_func(t, lambda_val=3):
    pmax = p0_func(t) - alpha_func(t) * np.log(85 / (lambda_val * 300))
    pmax_rounded = np.ceil(pmax / 50) * 50 # Round up the result to the nearest integer greater than or equal to the floating-point number.
    # return int(pmax_rounded)
    return 500

def Pw_func(p, t):
    return np.exp(-(p - p0_func(t)) / alpha_func(t))  # Exponential purchase probability