# Real-Time Portfolio Optimization and PNL Calculation

## Description
This Python project offers a comprehensive framework for real-time portfolio Profit and Loss (PNL) calculation. Leveraging the Alpha Vantage API, it fetches real-time asset prices, providing a live perspective on a portfolio's performance. The code includes three main classes for standard, advanced, and parallel PNL calculations, each introducing incremental features to suit various portfolio management needs.

## Features
- **Real-Time PNL Calculation**: Computes the PNL based on the current positions in the portfolio and real-time asset prices.
- **Portfolio Optimization**: Optimizes the portfolio based on the Sharpe ratio while considering liquidity constraints and other factors.
- **Advanced Risk Management**: Incorporates Value at Risk (VaR) and Conditional Value at Risk (CVaR) metrics to assess portfolio risk.
- **Historical Data Analysis**: Fetches historical asset prices to calculate a returns matrix and aids in advanced portfolio metrics.
- **Transaction Cost Adjustment**: Incorporates transaction costs when calculating PNL.
- **Real-Time Asset Price Updates**: Utilizes the Alpha Vantage API to fetch real-time asset prices.
- **Monte Carlo Simulations**: Employs parallel Monte Carlo simulations in the `ParallelGPURealTimePNL` class to estimate PNL. This simulates potential future scenarios based on random price movements.
- **Data Caching**: Implements a caching mechanism to prevent excessive API calls.

## Requirements
- Python 3.x
- NumPy
- pandas
- SciPy
- Joblib
- Alpha Vantage API (Python library)
- datetime (standard library)

## Usage
1. Clone this GitHub repository.
2. Install the above-mentioned dependencies.
3. Replace `YOUR_ALPHA_VANTAGE_API_KEY` in the code with your actual Alpha Vantage API key.
4. If you have a portfolio CSV, update the path in the `if __name__ == "__main__":` section.
5. Run the code.

## Contact
For any questions or suggestions, feel free to open a GitHub issue.
