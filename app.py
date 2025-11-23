"""
Forecast Web Platform

A Flask web application for time series forecasting using SARIMAX models.
Upload CSV/Excel files and download forecast results.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

from sarimax_backtest import ForecastConfig, run_forecast

# Configuration
UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("forecast_results")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "forecast-platform-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and run forecast."""
    # Check if file was submitted
    if "file" not in request.files:
        flash("No file selected", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload CSV or Excel files (.csv, .xlsx, .xls)", "error")
        return redirect(url_for("index"))

    # Get forecast parameters from form
    freq = request.form.get("freq", "Q")
    backtest_periods = int(request.form.get("backtest_periods", 8))
    forecast_periods = int(request.form.get("forecast_periods", 4))
    seasonal_period = int(request.form.get("seasonal_period", 4))
    target_column = request.form.get("target_column", "").strip() or None
    include_backtest = request.form.get("include_backtest") == "on"
    include_future = request.form.get("include_future") == "on"

    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())[:8]
    original_filename = secure_filename(file.filename)
    filename = f"{unique_id}_{original_filename}"
    filepath = UPLOAD_FOLDER / filename

    # Save uploaded file
    file.save(filepath)

    try:
        # Configure forecast
        config = ForecastConfig(
            last_n_periods=backtest_periods,
            forecast_periods=forecast_periods,
            seasonal_period=seasonal_period,
            target_column=target_column,
            freq=freq,
            output_dir=RESULTS_FOLDER,
        )

        # Run forecast
        success, message, output_path = run_forecast(
            filepath,
            config,
            include_backtest=include_backtest,
            include_future=include_future,
        )

        if success and output_path:
            # Return the forecast file for download
            return send_file(
                output_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"{Path(original_filename).stem}_forecast.csv"
            )
        else:
            flash(f"Forecast failed: {message}", "error")
            return redirect(url_for("index"))

    except Exception as e:
        flash(f"Error processing file: {str(e)}", "error")
        return redirect(url_for("index"))

    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """API endpoint for programmatic access."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    # Get parameters from form or JSON
    freq = request.form.get("freq", "Q")
    backtest_periods = int(request.form.get("backtest_periods", 8))
    forecast_periods = int(request.form.get("forecast_periods", 4))
    seasonal_period = int(request.form.get("seasonal_period", 4))
    target_column = request.form.get("target_column", "").strip() or None
    include_backtest = request.form.get("include_backtest", "true").lower() == "true"
    include_future = request.form.get("include_future", "true").lower() == "true"

    # Save file
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{secure_filename(file.filename)}"
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    try:
        config = ForecastConfig(
            last_n_periods=backtest_periods,
            forecast_periods=forecast_periods,
            seasonal_period=seasonal_period,
            target_column=target_column,
            freq=freq,
            output_dir=RESULTS_FOLDER,
        )

        success, message, output_path = run_forecast(
            filepath,
            config,
            include_backtest=include_backtest,
            include_future=include_future,
        )

        if success and output_path:
            return send_file(
                output_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"{Path(file.filename).stem}_forecast.csv"
            )
        else:
            return jsonify({"error": message}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
