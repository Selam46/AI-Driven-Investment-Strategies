# AI-Driven Investment Strategies

## Overview
This project implements advanced data analysis and machine learning techniques for financial market analysis and portfolio management. Developed for GMF Investments, it provides comprehensive tools for analyzing market trends, assessing risks, and optimizing investment strategies.

## Features
- Historical financial data analysis for multiple assets (TSLA, BND, SPY)
- Comprehensive data preprocessing and cleaning
- Advanced time series analysis including:
  - Price trend analysis
  - Volatility analysis
  - Seasonality detection
  - Risk metrics calculation
- Interactive visualizations for market insights
- Statistical analysis of market behavior

## Installation

1. Clone the repository:

## Project Structure
```
project/
│
├── notebooks/
│   └── 01_data_preprocessing_and_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   └── financial_metrics.py
│
├── data/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage

1. Data Preprocessing and EDA:
```bash
jupyter notebook notebooks/01_data_preprocessing_and_eda.ipynb
```

## Data Sources
- Tesla (TSLA): High-growth, high-risk stock
- Vanguard Total Bond Market ETF (BND): Stable bond market exposure
- S&P 500 ETF (SPY): Broad market exposure
- Time Period: January 1, 2015 to January 31, 2025

## Analysis Components
1. Price Evolution Analysis
   - Normalized price comparisons
   - Volume analysis
   - Moving averages

2. Returns Analysis
   - Distribution analysis
   - Q-Q plots
   - Statistical moments

3. Volatility Analysis
   - Rolling volatility
   - Volatility regimes
   - Risk metrics (VaR, Sharpe Ratio)

4. Seasonality Analysis
   - Trend decomposition
   - Monthly patterns
   - Day-of-week effects

