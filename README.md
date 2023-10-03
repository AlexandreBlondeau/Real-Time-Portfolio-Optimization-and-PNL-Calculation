# Real-Time Portfolio Optimization and PNL Calculation

## Description
This Python project offers a comprehensive framework for real-time portfolio optimization and Profit and Loss (PNL) calculation. It integrates advanced features like risk management, transaction cost optimization, and real-time asset price updates via the Alpha Vantage API. Designed for extensibility, the code also employs parallelization techniques to improve computational performance.

## Features
- **Real-Time PNL Calculation**: Updates the PNL based on current portfolio positions and real-time asset prices.
- **Portfolio Optimization**: Uses quadratic programming to find the optimal asset weights in the portfolio.
- **Advanced Risk Management**: Computes risk measures like Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- **Transaction Cost Optimization**: Accounts for transaction costs in portfolio optimization.
- **Real-Time Asset Price Updates**: Fetches real-time asset prices via the Alpha Vantage API.
- **Parallelization and GPU Acceleration**: Utilizes the Joblib library for CPU parallelization and annotations for GPU acceleration.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Joblib
- Alpha Vantage API (Python library)
- pandas

## Usage
1. Clone this GitHub repository.
2. Install the above-mentioned dependencies.
3. Replace `YOUR_ALPHA_VANTAGE_API_KEY` in the code with your actual Alpha Vantage API key.
4. Run the code.

## Contact
For any questions or suggestions, feel free to open a GitHub issue.
