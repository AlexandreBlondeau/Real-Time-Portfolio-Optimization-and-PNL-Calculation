import numpy as np
import logging
import pandas as pd
import datetime
from scipy.optimize import minimize
from scipy.stats import norm
from joblib import Parallel, delayed
from alpha_vantage.timeseries import TimeSeries


class RealTimePNL:
    @staticmethod
    def load_data_from_csv(filepath):
        df = pd.read_csv(filepath)
        portfolio = {}
        for index, row in df.iterrows():
            portfolio[row['Asset']] = (row['Position'], row['Price'])
        return portfolio

    def __init__(self, initial_portfolio, liquidity_constraints=None, api_key=None):
        self.portfolio = initial_portfolio
        self.liquidity_constraints = liquidity_constraints
        self.pnl_history = []
        self.api_key = api_key  # API Key for Alpha Vantage
        logging.basicConfig(level=logging.INFO)
        self.real_time_data_cache = {}  # Add cache

    def calculate_PNL(self):
        pnl = sum(position * price for position, price in self.portfolio.values())
        self.pnl_history.append(pnl)
        logging.info(f"Updated PNL: {pnl}")
        return pnl

    def update_real_time_data(self, asset):
        # Vérification du cache
        if asset in self.real_time_data_cache:
            last_update_timestamp, cached_price = self.real_time_data_cache[asset]
            # Let's assume that we consider the data to be fresh for 5 minutes
            if (datetime.datetime.now() - last_update_timestamp).seconds < 300:
                real_time_price = cached_price
            else:
                real_time_price = self.fetch_real_time_price(asset)
                self.real_time_data_cache[asset] = (datetime.datetime.now(), real_time_price)
        else:
            real_time_price = self.fetch_real_time_price(asset)
            self.real_time_data_cache[asset] = (datetime.datetime.now(), real_time_price)

        # Portfolio updated with real-time prices
        if asset in self.portfolio:
            position, _ = self.portfolio[asset]
            self.portfolio[asset] = (position, real_time_price)

    def fetch_real_time_price(self, asset):
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        data, meta_data = ts.get_quote_endpoint(symbol=asset)
        return float(data['05. price'].iloc[0])

    # Estimate Sharpe ratio for given portfolio weights
    def estimate_sharpe_ratio(self, weights, positions, prices):
        portfolio_return = np.dot(weights, positions * prices)
        portfolio_volatility = np.std(self.pnl_history)
        return portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0

    # Conduct portfolio optimization
    def multi_objective_optimization(self):
        # Extract asset names, positions, and prices
        assets = list(self.portfolio.keys())
        positions = np.array([self.portfolio[asset][0] for asset in assets])
        prices = np.array([self.portfolio[asset][1] for asset in assets])

        # Define the optimization objective function
        def objective(weights):
            return -self.estimate_sharpe_ratio(weights, positions, prices)

        # Define the constraints
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

        # Incorporate liquidity constraints if specified
        if self.liquidity_constraints:
            for asset, constraint in self.liquidity_constraints.items():
                if asset in self.portfolio:
                    i = assets.index(asset)
                    constraints.append(
                        {'type': 'ineq', 'fun': lambda weights, i=i, constraint=constraint: constraint - weights[i]})

        # Define bounds for the asset weights
        bounds = [(0, 1) for _ in assets]
        initial_weights = [1. / len(assets) for _ in assets]

        # Run the optimization
        solution = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

        # Update portfolio if optimization is successful
        if solution.success:
            optimized_weights = solution.x
            for i, asset in enumerate(assets):
                position, price = self.portfolio[asset]
                self.portfolio[asset] = (position * optimized_weights[i], price)
        else:
            logging.warning("Portfolio optimization failed.")


# Define the derived class for Advanced PNL calculation
class AdvancedRealTimePNL(RealTimePNL):
    def __init__(self, initial_portfolio, liquidity_constraints=None, transaction_cost=0.001, api_key=None):
        super().__init__(initial_portfolio, liquidity_constraints, api_key)
        self.transaction_cost = transaction_cost  # Transaction cost rate
        self.returns = None  # To store the returns matrix

    def fetch_price_data(self, asset):
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=asset, outputsize='full')
        return data['4. close']  # Get only the closing prices

    def calculate_returns_matrix(self):
        price_data = {}
        for asset in self.portfolio:
            price_data[asset] = self.fetch_price_data(asset)
        price_data_df = pd.DataFrame(price_data)
        self.returns = np.log(price_data_df / price_data_df.shift(1)).dropna()

    def calculate_portfolio_returns(self, weights):
        if self.returns is None:
            self.calculate_returns_matrix()
        weighted_returns = self.returns * weights[:, np.newaxis]
        portfolio_returns = np.sum(weighted_returns, axis=1)
        return portfolio_returns

    # Calculate PNL considering transaction costs
    def calculate_PNL(self):
        pnl = super().calculate_PNL()
        return pnl - self.transaction_cost * sum(abs(position) for position, _ in self.portfolio.values())

    # Calculate Value at Risk (VaR) at given confidence level
    def VaR(self, weights, alpha=0.05):
        portfolio_returns = self.calculate_portfolio_returns(weights)
        if len(portfolio_returns) == 0:
            return 0
        return -np.percentile(portfolio_returns, 100 * alpha)

    def CVaR(self, weights, alpha=0.05):
        portfolio_returns = self.calculate_portfolio_returns(weights)
        if len(portfolio_returns) == 0:
            return 0
        var = self.VaR(weights, alpha)
        return -portfolio_returns[portfolio_returns <= -var].mean()

    def estimate_sharpe_ratio(self, weights, positions, prices):
        # Calculation of portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(weights)

        # Calculation of the volatility of portfolio retur
        portfolio_volatility = np.std(portfolio_returns)

        # Calculation of average portfolio return
        portfolio_mean_return = np.mean(portfolio_returns)

        return portfolio_mean_return / portfolio_volatility if portfolio_volatility != 0 else 0

    # Conduct multi-objective portfolio optimization
    def multi_objective_optimization(self):
        # Extract asset names, positions, and prices
        assets = list(self.portfolio.keys())
        positions = np.array([self.portfolio[asset][0] for asset in assets])
        prices = np.array([self.portfolio[asset][1] for asset in assets])

        # Define the multi-objective optimization function
        def objective(weights):
            adjusted_positions = positions * weights
            transaction_costs = self.transaction_cost * np.sum(np.abs(adjusted_positions))
            var = self.VaR(weights=weights)  # Assume VaR method accepts weights
            cvar = self.CVaR(weights=weights)  # Assume CVaR method accepts weights
            risk_penalty = var + cvar  # Adjust this to your liking
            portfolio_returns = self.calculate_portfolio_returns(weights)
            return (-self.estimate_sharpe_ratio(weights, positions, prices) +
                    np.std(portfolio_returns) + transaction_costs + risk_penalty)


# Parallel and GPU-Accelerated class derived from AdvancedRealTimePNL
class ParallelGPURealTimePNL(AdvancedRealTimePNL):
    def __init__(self, initial_portfolio, liquidity_constraints=None, transaction_cost=0.001, n_simulations=1000):
        super().__init__(initial_portfolio, liquidity_constraints, transaction_cost)
        self.n_simulations = n_simulations  # Number of Monte Carlo simulations

    # Run Monte Carlo simulations in parallel
    def monte_carlo_simulation(self):
        pnl_simulations = Parallel(n_jobs=-1)(delayed(self.simulate_single_run)() for _ in range(self.n_simulations))
        return pnl_simulations

    # Simulate a single run for Monte Carlo
    def simulate_single_run(self):
        simulated_pnl = 0
        for asset, (position, price) in self.portfolio.items():
            simulated_price = price * (1 + np.random.normal(0, 0.01))
            simulated_pnl += position * simulated_price
        return simulated_pnl - self.transaction_cost * sum(abs(position) for position, _ in self.portfolio.items())


# The code for data loading and execution starts here.
if __name__ == "__main__":
    initial_portfolio = RealTimePNL.load_data_from_csv("your_file.csv")  # Replace with your actual CSV file path
    pnl_calculator = AdvancedRealTimePNL(initial_portfolio,
                                         api_key="YOUR_ALPHA_VANTAGE_API_KEY",  # Replace with your actual API key
                                         transaction_cost=0.001)  # Specify transaction cost if different from default
    pnl_calculator.update_real_time_data('AAPL')  # Update real-time data for the 'AAPL' asset
    pnl_calculator.calculate_returns_matrix()  # Calculate the returns matrix using the new method
