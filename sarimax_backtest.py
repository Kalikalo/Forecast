"""
SARIMAX Last-N Backtest Framework

This module performs rolling backtests using SARIMAX models with exogenous variables.
Each dataset must contain: Date column + Target variable + exogenous features.

Outputs: One CSV per file with mean forecast and symmetric prediction intervals
         (2.5/97.5, 5/95, 10/90, 20/80 percentiles).
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BacktestConfig:
    """Configuration parameters for SARIMAX backtesting."""

    # Backtest settings
    last_n_periods: int = 8
    seasonal_period: int = 4

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
    output_dir: Path = field(default_factory=lambda: Path("sarimax_results"))


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


def coerce_dates_to_quarter_end(series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert various date formats to quarter-end timestamps.

    Handles both "YYYY QN" format (e.g., "2024 Q1") and standard datetime strings.

    Args:
        series: Series containing date values in various formats.

    Returns:
        DatetimeIndex with quarter-end dates.
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
                idx = dt.dt.to_period("Q").dt.to_timestamp(how="end")
            except Exception:
                idx = dt
        else:
            idx = dt

    return idx


def load_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load an Excel dataset and prepare it for backtesting.

    The dataset should have a Date column (or use first column as Date)
    and the first non-Date column is treated as the target variable.

    Args:
        path: Path to the Excel file.

    Returns:
        Tuple of (prepared DataFrame with datetime index, target column name).
    """
    df = pd.read_excel(path)

    # Ensure Date column exists
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    # First non-Date column is the target
    target = [c for c in df.columns if c != "Date"][0]

    # Convert dates and set as index
    idx = coerce_dates_to_quarter_end(df["Date"])
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
# Main Execution
# =============================================================================


def run_backtest(
    file_path: Path,
    config: BacktestConfig,
) -> tuple[bool, str, Optional[Path]]:
    """
    Run SARIMAX backtest on a single file.

    Args:
        file_path: Path to the input Excel file.
        config: Backtest configuration.

    Returns:
        Tuple of (success, message, output_path or None).
    """
    try:
        # Load and prepare data
        df, target = load_dataset(file_path)
        test_idx = compute_test_indices(df, target, config.last_n_periods)
        y_full, X_full, _ = prepare_data(df, target)

        # Find best model
        best_params = grid_search_sarimax(y_full, X_full, test_idx, config)

        # Generate forecasts
        results_df = generate_forecast_rows(
            y_full,
            X_full,
            test_idx,
            order=best_params["order"],
            seasonal=best_params["seasonal"],
            config=config,
        )

        # Save results
        output_path = config.output_dir / f"{file_path.stem}_last{config.last_n_periods}_backtest.csv"
        results_df.to_csv(output_path, index=False)

        return True, f"Saved to {output_path.name}", output_path

    except Exception as e:
        return False, str(e), None


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


if __name__ == "__main__":
    # Configure backtest
    config = BacktestConfig(
        last_n_periods=8,
        seasonal_period=4,
        output_dir=Path("/content/sarimax_results_explicit"),
    )

    # Run backtests
    run_all_backtests(DEFAULT_FILES, config)
