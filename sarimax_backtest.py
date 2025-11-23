"""
SARIMAX Dynamic Forecast Framework

This module performs rolling backtests and future forecasts using SARIMAX models.
Supports both CSV and Excel file uploads with automatic format detection.

Features:
- Dynamic CSV/Excel file loading
- Rolling backtest with prediction intervals
- Future forecasting beyond available data
- Configurable parameters via CLI or programmatically

Outputs: CSV with mean forecast and symmetric prediction intervals
         (2.5/97.5, 5/95, 10/90, 20/80 percentiles).
"""

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ForecastConfig:
    """Configuration parameters for SARIMAX forecasting."""

    # Backtest settings
    last_n_periods: int = 8
    seasonal_period: int = 4

    # Future forecast settings
    forecast_periods: int = 4  # Number of periods to forecast into the future

    # Data settings
    date_column: Optional[str] = None  # Auto-detect if None
    target_column: Optional[str] = None  # First non-date column if None
    freq: str = "Q"  # Frequency: Q=quarterly, M=monthly, W=weekly, D=daily

    # ARIMA order search ranges
    p_range: tuple[int, int] = (0, 3)
    d_range: tuple[int, int] = (0, 2)
    q_range: tuple[int, int] = (0, 3)

    # Seasonal order search ranges
    P_range: tuple[int, int] = (0, 2)
    D_range: tuple[int, int] = (0, 2)
    Q_range: tuple[int, int] = (0, 2)

    # Model constraints
    disallow_pure_differencing: bool = True
    prefer_stationary: bool = True

    # Prediction interval percentiles (lower tail)
    percentiles: tuple[float, ...] = (2.5, 5, 10, 20)

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("forecast_results"))


# Alias for backwards compatibility
BacktestConfig = ForecastConfig


# =============================================================================
# Default File List
# =============================================================================

DEFAULT_FILES = [
    "/content/Procter_Gamble_Company_dataset.xlsx",
    "/content/PepsiCo_Inc_dataset.xlsx",
    "/content/Unilever_PLC_dataset.xlsx",
    "/content/Nestle_S_A_dataset.xlsx",
    "/content/Coca_Cola_Company_dataset.xlsx",
    "/content/Starbucks_Corporation_dataset.xlsx",
    "/content/LOreal_S_A_dataset.xlsx",
    "/content/Philip_Morris_International_Inc_dataset.xlsx",
    "/content/Bunge_Global_SA_dataset.xlsx",
    "/content/Danone_SA_dataset.xlsx",
    "/content/Emmi_AG_dataset.xlsx",
    "/content/Siemens_Aktiengesellschaft_dataset.xlsx",
    "/content/Archer_Daniels_Midland_Company_dataset.xlsx",
    "/content/Mondelez_International_Inc_Class_A_dataset.xlsx",
    "/content/Chocoladefabriken_Lindt_Spruengli_AG_dataset.xlsx",
    "/content/Hershey_Company_dataset.xlsx",
    "/content/Estee_Lauder_Companies_Inc_Class_A_dataset.xlsx",
    "/content/Beiersdorf_AG_dataset.xlsx",
]


# =============================================================================
# Data Loading Utilities
# =============================================================================


def coerce_dates_to_period_end(series: pd.Series, freq: str = "Q") -> pd.DatetimeIndex:
    """
    Convert various date formats to period-end timestamps.

    Handles both "YYYY QN" format (e.g., "2024 Q1") and standard datetime strings.
    Supports multiple frequencies: Q (quarterly), M (monthly), W (weekly), D (daily).

    Args:
        series: Series containing date values in various formats.
        freq: Frequency string - Q, M, W, or D.

    Returns:
        DatetimeIndex with period-end dates.
    """
    series = pd.Series(series)
    as_str = series.astype(str).str.strip()

    # Check if data is in "YYYY QN" format
    is_quarter_format = as_str.str.contains(r"^\d{4}\s*Q[1-4]$", regex=True)

    if is_quarter_format.mean() > 0.5:
        # Parse quarter format
        period_str = as_str.str.replace(r"\s+", "", regex=True)
        idx = pd.PeriodIndex(period_str, freq="Q").to_timestamp(how="end")
    else:
        # Parse as standard datetime
        dt = pd.to_datetime(as_str, errors="coerce")
        if dt.notna().mean() > 0.8:
            try:
                idx = dt.dt.to_period(freq).dt.to_timestamp(how="end")
            except Exception:
                idx = dt
        else:
            idx = dt

    return idx


# Alias for backwards compatibility
def coerce_dates_to_quarter_end(series: pd.Series) -> pd.DatetimeIndex:
    """Backwards compatibility wrapper."""
    return coerce_dates_to_period_end(series, freq="Q")


def detect_file_type(path: Path) -> str:
    """
    Detect file type based on extension.

    Args:
        path: Path to the file.

    Returns:
        File type string: 'csv', 'excel', or raises ValueError.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    elif suffix in [".xlsx", ".xls"]:
        return "excel"
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls")


def detect_date_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the date column in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Name of the detected date column.
    """
    # Common date column names
    date_names = ["date", "Date", "DATE", "time", "Time", "TIME", "period", "Period"]

    for name in date_names:
        if name in df.columns:
            return name

    # Try first column
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col].head(10), errors="raise")
        return first_col
    except Exception:
        pass

    # Check for quarter format in first column
    first_col_str = df[first_col].astype(str).str.strip()
    if first_col_str.str.contains(r"^\d{4}\s*Q[1-4]$", regex=True).mean() > 0.5:
        return first_col

    raise ValueError("Could not auto-detect date column. Please specify date_column in config.")


def load_dataset(
    path: Union[str, Path],
    config: Optional[ForecastConfig] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV or Excel dataset and prepare it for forecasting.

    Supports automatic file type detection and flexible date parsing.

    Args:
        path: Path to the CSV or Excel file.
        config: Optional configuration for column names and frequency.

    Returns:
        Tuple of (prepared DataFrame with datetime index, target column name).
    """
    path = Path(path)
    config = config or ForecastConfig()

    # Detect and load file
    file_type = detect_file_type(path)

    if file_type == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Detect or use specified date column
    date_col = config.date_column or detect_date_column(df)

    # Rename to standard "Date" if different
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    # Detect or use specified target column
    non_date_cols = [c for c in df.columns if c != "Date"]
    if not non_date_cols:
        raise ValueError("Dataset must have at least one column besides Date")

    target = config.target_column if config.target_column else non_date_cols[0]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Convert dates based on frequency
    idx = coerce_dates_to_period_end(df["Date"], config.freq)
    df = df.set_index(idx).sort_index().drop(columns=["Date"])

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, target


# =============================================================================
# Data Preparation
# =============================================================================


def compute_test_indices(
    df: pd.DataFrame, target_col: str, n_periods: int = 8
) -> pd.DatetimeIndex:
    """
    Compute the indices for the last N periods to use as test set.

    Args:
        df: DataFrame with datetime index.
        target_col: Name of the target column.
        n_periods: Number of periods to include in test set.

    Returns:
        DatetimeIndex of the last N periods with non-null target values.

    Raises:
        ValueError: If there are fewer than n_periods data points.
    """
    y = df[target_col].dropna()

    if len(y) < n_periods:
        raise ValueError(
            f"{target_col}: insufficient data points for last {n_periods} backtest"
        )

    return y.index[-n_periods:]


def prepare_data(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    Prepare target series and exogenous features.

    Args:
        df: DataFrame with target and exogenous columns.
        target_col: Name of the target column.

    Returns:
        Tuple of (target series, exogenous DataFrame, list of exogenous column names).
    """
    exog_cols = [c for c in df.columns if c != target_col]
    y = df[target_col]
    X = df[exog_cols].ffill().bfill()

    return y, X, exog_cols


# =============================================================================
# Model Selection
# =============================================================================


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(np.nanmean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def grid_search_sarimax(
    y_full: pd.Series,
    X_full: pd.DataFrame,
    test_idx: pd.DatetimeIndex,
    config: BacktestConfig,
) -> dict:
    """
    Perform grid search to find optimal SARIMAX parameters.

    Args:
        y_full: Full target series.
        X_full: Full exogenous features DataFrame.
        test_idx: Test period indices.
        config: Backtest configuration.

    Returns:
        Dictionary with best 'order' and 'seasonal' parameters.
    """
    # Training data: everything before test period
    y_train = y_full.loc[y_full.index < test_idx[0]].dropna()
    X_train = X_full.loc[y_train.index]

    best_result: Optional[dict] = None

    # Generate parameter grid
    p_values = range(config.p_range[0], config.p_range[1])
    d_values = range(config.d_range[0], config.d_range[1])
    q_values = range(config.q_range[0], config.q_range[1])
    P_values = range(config.P_range[0], config.P_range[1])
    D_values = range(config.D_range[0], config.D_range[1])
    Q_values = range(config.Q_range[0], config.Q_range[1])

    for _p in p_values:
        for _d in d_values:
            for _q in q_values:
                for _P in P_values:
                    for _D in D_values:
                        for _Q in Q_values:
                            # Skip pure differencing models if configured
                            if config.disallow_pure_differencing:
                                if _p + _q + _P + _Q == 0:
                                    continue

                            try:
                                # Fit model
                                model = sm.tsa.SARIMAX(
                                    y_train,
                                    exog=X_train,
                                    order=(_p, _d, _q),
                                    seasonal_order=(
                                        _P,
                                        _D,
                                        _Q,
                                        config.seasonal_period,
                                    ),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                result = model.fit(disp=False)

                                # Generate forecast
                                forecast = result.get_forecast(
                                    steps=len(test_idx), exog=X_full.loc[test_idx]
                                )
                                predictions = pd.Series(
                                    forecast.predicted_mean, index=test_idx
                                )

                                # Calculate RMSE on test set
                                comparison = pd.concat(
                                    [y_full.loc[test_idx], predictions], axis=1
                                ).dropna()

                                if comparison.empty:
                                    continue

                                score = calculate_rmse(
                                    comparison.iloc[:, 0], comparison.iloc[:, 1]
                                )

                                # Update best if improved
                                if best_result is None or score < best_result["score"]:
                                    best_result = {
                                        "score": score,
                                        "order": (_p, _d, _q),
                                        "seasonal": (_P, _D, _Q),
                                    }

                            except Exception:
                                continue

    # Return best or default
    if best_result:
        return best_result

    return {"order": (1, 0, 1), "seasonal": (0, 0, 0)}


# =============================================================================
# Forecasting
# =============================================================================


def extract_prediction_intervals(
    forecast, percentiles: tuple[float, ...]
) -> dict[str, np.ndarray]:
    """
    Extract prediction interval bounds from a forecast object.

    Args:
        forecast: statsmodels forecast object.
        percentiles: Lower-tail percentiles for intervals.

    Returns:
        Dictionary mapping column names to bound arrays.
    """
    bounds = {}

    for pctl in percentiles:
        alpha = 2 * pctl / 100

        try:
            summary = forecast.summary_frame(alpha=alpha)
            lower = (
                summary.get("obs_ci_lower", summary.get("mean_ci_lower"))
            ).to_numpy()
            upper = (
                summary.get("obs_ci_upper", summary.get("mean_ci_upper"))
            ).to_numpy()
        except Exception:
            conf_int = forecast.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0].to_numpy()
            upper = conf_int.iloc[:, 1].to_numpy()

        bounds[f"lower_{pctl}"] = lower
        bounds[f"upper_{100 - pctl}"] = upper

    return bounds


def generate_forecast_rows(
    y_full: pd.Series,
    X_full: pd.DataFrame,
    test_idx: pd.DatetimeIndex,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int],
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Generate rolling forecasts for each test period.

    Args:
        y_full: Full target series.
        X_full: Full exogenous features DataFrame.
        test_idx: Test period indices.
        order: ARIMA order (p, d, q).
        seasonal: Seasonal order (P, D, Q).
        config: Backtest configuration.

    Returns:
        DataFrame with forecast results in long format.
    """
    rows = []

    for i, as_of_date in enumerate(test_idx):
        # Training data: everything before current as_of date
        y_train = y_full.loc[y_full.index < as_of_date].dropna()
        X_train = X_full.loc[y_train.index]

        # Forecast horizon
        n_steps = len(test_idx) - i
        X_forecast = X_full.loc[test_idx[i] :]

        # Fit model
        model = sm.tsa.SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=(seasonal[0], seasonal[1], seasonal[2], config.seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        # Generate forecast
        forecast = result.get_forecast(steps=n_steps, exog=X_forecast.iloc[:n_steps])
        predicted_mean = forecast.predicted_mean.values

        # Extract prediction intervals
        bounds = extract_prediction_intervals(forecast, config.percentiles)

        # Get actual values
        forecast_dates = test_idx[i:]
        actual_values = y_full.loc[forecast_dates].values

        # Build result rows
        for h, (date, mean_val) in enumerate(zip(forecast_dates, predicted_mean), start=1):
            row = {
                "as_of": as_of_date,
                "h": h,
                "Date": date,
                "actual": float(actual_values[h - 1])
                if pd.notna(actual_values[h - 1])
                else np.nan,
                "mean": float(mean_val),
            }

            # Add interval bounds
            for key, values in bounds.items():
                row[key] = (
                    float(values[h - 1]) if np.isfinite(values[h - 1]) else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Future Forecasting
# =============================================================================


def generate_future_dates(
    last_date: pd.Timestamp,
    n_periods: int,
    freq: str = "Q",
) -> pd.DatetimeIndex:
    """
    Generate future dates for forecasting.

    Args:
        last_date: The last date in the historical data.
        n_periods: Number of future periods to generate.
        freq: Frequency string - Q, M, W, or D.

    Returns:
        DatetimeIndex of future dates.
    """
    future_periods = pd.date_range(
        start=last_date,
        periods=n_periods + 1,
        freq=freq,
    )[1:]  # Skip the first one (which is the last historical date)

    # Convert to period end
    return future_periods.to_period(freq).to_timestamp(how="end")


def generate_future_forecast(
    y_full: pd.Series,
    X_full: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int],
    config: ForecastConfig,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate future forecasts beyond the available data.

    Args:
        y_full: Full target series.
        X_full: Full exogenous features DataFrame.
        order: ARIMA order (p, d, q).
        seasonal: Seasonal order (P, D, Q).
        config: Forecast configuration.
        exog_future: Optional future exogenous values. If None, uses last known values.

    Returns:
        DataFrame with future forecast results.
    """
    # Fit model on all available data
    y_train = y_full.dropna()
    X_train = X_full.loc[y_train.index]

    model = sm.tsa.SARIMAX(
        y_train,
        exog=X_train if not X_train.empty else None,
        order=order,
        seasonal_order=(seasonal[0], seasonal[1], seasonal[2], config.seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    # Generate future dates
    last_date = y_train.index[-1]
    future_dates = generate_future_dates(last_date, config.forecast_periods, config.freq)

    # Prepare future exogenous values
    if exog_future is not None:
        X_forecast = exog_future.iloc[: config.forecast_periods]
    elif not X_full.empty:
        # Use last known exogenous values (forward fill)
        last_exog = X_full.iloc[-1:].values
        X_forecast = pd.DataFrame(
            np.tile(last_exog, (config.forecast_periods, 1)),
            index=future_dates,
            columns=X_full.columns,
        )
    else:
        X_forecast = None

    # Generate forecast
    forecast = result.get_forecast(
        steps=config.forecast_periods,
        exog=X_forecast if X_forecast is not None and not X_forecast.empty else None,
    )
    predicted_mean = forecast.predicted_mean.values

    # Extract prediction intervals
    bounds = extract_prediction_intervals(forecast, config.percentiles)

    # Build result rows
    rows = []
    as_of_date = last_date

    for h, (date, mean_val) in enumerate(zip(future_dates, predicted_mean), start=1):
        row = {
            "as_of": as_of_date,
            "h": h,
            "Date": date,
            "actual": np.nan,  # No actuals for future
            "mean": float(mean_val),
            "type": "future_forecast",
        }

        # Add interval bounds
        for key, values in bounds.items():
            row[key] = float(values[h - 1]) if np.isfinite(values[h - 1]) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main Execution
# =============================================================================


def run_forecast(
    file_path: Union[str, Path],
    config: Optional[ForecastConfig] = None,
    include_backtest: bool = True,
    include_future: bool = True,
) -> tuple[bool, str, Optional[Path]]:
    """
    Run SARIMAX forecast on a single CSV or Excel file.

    This is the main entry point for dynamic forecasting. It performs:
    1. Data loading with automatic format detection
    2. Grid search for optimal SARIMAX parameters
    3. Rolling backtest (optional)
    4. Future forecast (optional)

    Args:
        file_path: Path to the input CSV or Excel file.
        config: Forecast configuration (uses defaults if None).
        include_backtest: Whether to include backtest results.
        include_future: Whether to include future forecasts.

    Returns:
        Tuple of (success, message, output_path or None).
    """
    file_path = Path(file_path)
    config = config or ForecastConfig()

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and prepare data
        print(f"Loading data from {file_path.name}...")
        df, target = load_dataset(file_path, config)
        print(f"  Target variable: {target}")
        print(f"  Data range: {df.index[0]} to {df.index[-1]}")
        print(f"  Observations: {len(df)}")

        y_full, X_full, exog_cols = prepare_data(df, target)
        print(f"  Exogenous variables: {len(exog_cols)}")

        results_dfs = []

        # Backtest
        if include_backtest:
            print(f"\nRunning backtest (last {config.last_n_periods} periods)...")
            test_idx = compute_test_indices(df, target, config.last_n_periods)

            # Find best model
            print("  Searching for optimal parameters...")
            best_params = grid_search_sarimax(y_full, X_full, test_idx, config)
            print(f"  Best order: {best_params['order']}")
            print(f"  Best seasonal: {best_params['seasonal']}")

            # Generate backtest forecasts
            print("  Generating rolling forecasts...")
            backtest_df = generate_forecast_rows(
                y_full,
                X_full,
                test_idx,
                order=best_params["order"],
                seasonal=best_params["seasonal"],
                config=config,
            )
            backtest_df["type"] = "backtest"
            results_dfs.append(backtest_df)
        else:
            # Still need to find best params for future forecast
            test_idx = compute_test_indices(df, target, min(config.last_n_periods, len(df) - 2))
            best_params = grid_search_sarimax(y_full, X_full, test_idx, config)

        # Future forecast
        if include_future and config.forecast_periods > 0:
            print(f"\nGenerating {config.forecast_periods}-period future forecast...")
            future_df = generate_future_forecast(
                y_full,
                X_full,
                order=best_params["order"],
                seasonal=best_params["seasonal"],
                config=config,
            )
            results_dfs.append(future_df)
            print(f"  Forecast range: {future_df['Date'].min()} to {future_df['Date'].max()}")

        # Combine results
        if results_dfs:
            results_df = pd.concat(results_dfs, ignore_index=True)

            # Save results
            output_name = f"{file_path.stem}_forecast.csv"
            output_path = config.output_dir / output_name
            results_df.to_csv(output_path, index=False)

            print(f"\nResults saved to: {output_path}")
            return True, f"Saved to {output_path.name}", output_path
        else:
            return False, "No forecasts generated", None

    except Exception as e:
        return False, str(e), None


# Backwards compatible alias
def run_backtest(
    file_path: Path,
    config: ForecastConfig,
) -> tuple[bool, str, Optional[Path]]:
    """
    Run SARIMAX backtest on a single file.

    This is a backwards-compatible wrapper around run_forecast.

    Args:
        file_path: Path to the input file.
        config: Backtest configuration.

    Returns:
        Tuple of (success, message, output_path or None).
    """
    return run_forecast(file_path, config, include_backtest=True, include_future=False)


def run_all_backtests(
    files: list[str],
    config: Optional[BacktestConfig] = None,
) -> dict[str, dict]:
    """
    Run SARIMAX backtests on multiple files.

    Args:
        files: List of file paths to process.
        config: Backtest configuration (uses defaults if None).

    Returns:
        Dictionary mapping filenames to result info.
    """
    if config is None:
        config = BacktestConfig()

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for file_path_str in files:
        file_path = Path(file_path_str)
        success, message, output_path = run_backtest(file_path, config)

        status = "[OK]" if success else "[ERROR]"
        print(f"{status} {file_path.name}: {message}")

        results[file_path.name] = {
            "success": success,
            "message": message,
            "output_path": output_path,
        }

    print(f"\nAll done! Results saved in: {config.output_dir.as_posix()}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="SARIMAX Dynamic Forecast Framework - Upload CSV/Excel files for time series forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - forecast from a CSV file
  python sarimax_backtest.py data.csv

  # Forecast from Excel file with custom output directory
  python sarimax_backtest.py sales.xlsx --output ./my_results

  # Only run future forecast (no backtest)
  python sarimax_backtest.py data.csv --future-only --periods 8

  # Monthly data with custom target column
  python sarimax_backtest.py monthly_sales.csv --freq M --target Revenue

  # Process multiple files
  python sarimax_backtest.py file1.csv file2.xlsx file3.csv
        """,
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="CSV or Excel file(s) to process",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("forecast_results"),
        help="Output directory for results (default: forecast_results)",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (default: first non-date column)",
    )

    parser.add_argument(
        "--date-column",
        type=str,
        default=None,
        help="Date column name (default: auto-detect)",
    )

    parser.add_argument(
        "--freq",
        type=str,
        choices=["Q", "M", "W", "D"],
        default="Q",
        help="Data frequency: Q=quarterly, M=monthly, W=weekly, D=daily (default: Q)",
    )

    parser.add_argument(
        "--backtest-periods",
        type=int,
        default=8,
        help="Number of periods for backtesting (default: 8)",
    )

    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=4,
        help="Number of future periods to forecast (default: 4)",
    )

    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=4,
        help="Seasonal period length (default: 4 for quarterly)",
    )

    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Only run backtest, no future forecast",
    )

    parser.add_argument(
        "--future-only",
        action="store_true",
        help="Only run future forecast, no backtest",
    )

    return parser


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Build configuration from arguments
    config = ForecastConfig(
        last_n_periods=args.backtest_periods,
        forecast_periods=args.forecast_periods,
        seasonal_period=args.seasonal_period,
        date_column=args.date_column,
        target_column=args.target,
        freq=args.freq,
        output_dir=args.output,
    )

    # Determine what to run
    include_backtest = not args.future_only
    include_future = not args.backtest_only

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SARIMAX Dynamic Forecast Framework")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Frequency: {config.freq}")
    print(f"Backtest periods: {config.last_n_periods}")
    print(f"Forecast periods: {config.forecast_periods}")
    print(f"Include backtest: {include_backtest}")
    print(f"Include future: {include_future}")
    print("=" * 60)

    # Process each file
    results = {}
    for file_path_str in args.files:
        file_path = Path(file_path_str)
        print(f"\n{'=' * 60}")
        print(f"Processing: {file_path.name}")
        print("=" * 60)

        success, message, output_path = run_forecast(
            file_path,
            config,
            include_backtest=include_backtest,
            include_future=include_future,
        )

        status = "[OK]" if success else "[ERROR]"
        print(f"\n{status} {file_path.name}: {message}")

        results[file_path.name] = {
            "success": success,
            "message": message,
            "output_path": output_path,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for r in results.values() if r["success"])
    print(f"Processed: {len(results)} files")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Results saved in: {config.output_dir.as_posix()}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
