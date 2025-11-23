# Forecast

SARIMAX time series forecasting and backtesting framework for financial/business data.

## Overview

This repository provides a rolling backtest framework using SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) models. It performs automated parameter grid search and generates forecasts with prediction intervals.

## Features

- Rolling window backtesting with configurable test period length
- Automatic SARIMAX parameter optimization via grid search
- Support for quarterly data with various date formats
- Multiple symmetric prediction intervals (2.5/97.5, 5/95, 10/90, 20/80)
- Configurable via dataclass for easy customization

## Installation

```bash
pip install pandas numpy statsmodels openpyxl
```

## Usage

### Basic Usage

```python
from pathlib import Path
from sarimax_backtest import BacktestConfig, run_all_backtests

# Define your input files
files = [
    "/path/to/dataset1.xlsx",
    "/path/to/dataset2.xlsx",
]

# Run with default configuration
run_all_backtests(files)
```

### Custom Configuration

```python
from sarimax_backtest import BacktestConfig, run_all_backtests

config = BacktestConfig(
    last_n_periods=8,           # Number of periods for backtest
    seasonal_period=4,          # Quarterly seasonality
    p_range=(0, 3),             # AR order search range
    d_range=(0, 2),             # Differencing order search range
    q_range=(0, 3),             # MA order search range
    P_range=(0, 2),             # Seasonal AR order search range
    D_range=(0, 2),             # Seasonal differencing search range
    Q_range=(0, 2),             # Seasonal MA order search range
    percentiles=(2.5, 5, 10, 20),  # Prediction interval percentiles
    output_dir=Path("./results"),
)

run_all_backtests(files, config)
```

### Single File Processing

```python
from pathlib import Path
from sarimax_backtest import BacktestConfig, run_backtest

config = BacktestConfig(output_dir=Path("./results"))
success, message, output_path = run_backtest(Path("dataset.xlsx"), config)
```

## Input Data Format

Each Excel file should contain:

| Date | Target | Exog1 | Exog2 | ... |
|------|--------|-------|-------|-----|
| 2020 Q1 | 100.5 | 50.2 | 30.1 | ... |
| 2020 Q2 | 102.3 | 51.0 | 31.5 | ... |

- **Date column**: Supports "YYYY QN" format (e.g., "2024 Q1") or standard datetime
- **Target**: First non-Date column is treated as the forecast target
- **Exogenous features**: All remaining columns are used as exogenous variables

## Output Format

Each output CSV contains:

| Column | Description |
|--------|-------------|
| `as_of` | Forecast origin date |
| `h` | Forecast horizon (1 = next period, 2 = two periods ahead, etc.) |
| `Date` | Target date being forecasted |
| `actual` | Actual observed value |
| `mean` | Point forecast |
| `lower_2.5` | 2.5th percentile (95% PI lower bound) |
| `upper_97.5` | 97.5th percentile (95% PI upper bound) |
| `lower_5` | 5th percentile (90% PI lower bound) |
| `upper_95` | 95th percentile (90% PI upper bound) |
| `lower_10` | 10th percentile (80% PI lower bound) |
| `upper_90` | 90th percentile (80% PI upper bound) |
| `lower_20` | 20th percentile (60% PI lower bound) |
| `upper_80` | 80th percentile (60% PI upper bound) |

## Module Structure

```
sarimax_backtest.py
├── Configuration
│   └── BacktestConfig (dataclass)
├── Data Loading
│   ├── coerce_dates_to_quarter_end()
│   └── load_dataset()
├── Data Preparation
│   ├── compute_test_indices()
│   └── prepare_data()
├── Model Selection
│   ├── calculate_rmse()
│   └── grid_search_sarimax()
├── Forecasting
│   ├── extract_prediction_intervals()
│   └── generate_forecast_rows()
└── Main Execution
    ├── run_backtest()
    └── run_all_backtests()
```

## License

MIT
