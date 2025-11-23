"""
Forecast Web Platform

A Flask web application for time series forecasting using SARIMAX models.
Upload CSV/Excel files and download forecast results.
"""

import os
import sys
import uuid
import logging
import traceback
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

from sarimax_backtest import ForecastConfig, run_forecast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("forecast_results")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "txt"}

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
    filepath = None

    try:
        # Check if file was submitted
        if "file" not in request.files:
            flash("No file selected", "error")
            return redirect(url_for("index"))

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload CSV, TXT, or Excel files (.csv, .txt, .xlsx, .xls)", "error")
            return redirect(url_for("index"))

        # Get forecast parameters from form
        freq = request.form.get("freq", "Q")
        backtest_periods = int(request.form.get("backtest_periods", 8))
        forecast_periods = int(request.form.get("forecast_periods", 4))
        seasonal_period = int(request.form.get("seasonal_period", 4))
        target_column = request.form.get("target_column", "").strip() or None
        include_backtest = request.form.get("include_backtest") == "on"
        include_future = request.form.get("include_future") == "on"

        # Ensure at least one forecast type is selected
        if not include_backtest and not include_future:
            include_future = True  # Default to future forecast

        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Settings: freq={freq}, backtest_periods={backtest_periods}, forecast_periods={forecast_periods}")
        logger.info(f"include_backtest={include_backtest}, include_future={include_future}")

        # Generate unique filename to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        original_filename = secure_filename(file.filename)
        filename = f"{unique_id}_{original_filename}"
        filepath = UPLOAD_FOLDER / filename

        # Save uploaded file
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")

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
        logger.info("Running forecast...")
        success, message, output_path = run_forecast(
            filepath,
            config,
            include_backtest=include_backtest,
            include_future=include_future,
        )

        if success and output_path:
            logger.info(f"Forecast successful: {output_path}")
            # Return the forecast file for download
            return send_file(
                output_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"{Path(original_filename).stem}_forecast.csv"
            )
        else:
            logger.error(f"Forecast failed: {message}")
            flash(f"Forecast failed: {message}", "error")
            return redirect(url_for("index"))

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Value error: {error_msg}")
        flash(f"Data error: {error_msg}", "error")
        return redirect(url_for("index"))

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing file: {error_msg}")
        logger.error(traceback.format_exc())
        flash(f"Error: {error_msg}", "error")
        return redirect(url_for("index"))

    finally:
        # Clean up uploaded file
        if filepath and filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"Cleaned up: {filepath}")
            except Exception as e:
                logger.warning(f"Could not delete file: {e}")


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """API endpoint for programmatic access."""
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file. Supported formats: .csv, .txt, .xlsx, .xls"}), 400

        # Get parameters from form or JSON
        freq = request.form.get("freq", "Q")
        backtest_periods = int(request.form.get("backtest_periods", 8))
        forecast_periods = int(request.form.get("forecast_periods", 4))
        seasonal_period = int(request.form.get("seasonal_period", 4))
        target_column = request.form.get("target_column", "").strip() or None
        include_backtest = request.form.get("include_backtest", "true").lower() == "true"
        include_future = request.form.get("include_future", "true").lower() == "true"

        # Ensure at least one forecast type is selected
        if not include_backtest and not include_future:
            include_future = True

        # Save file
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

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

    except ValueError as e:
        return jsonify({"error": f"Data error: {str(e)}"}), 400

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        if filepath and filepath.exists():
            try:
                filepath.unlink()
            except Exception:
                pass


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    flash("An internal error occurred. Please check your file format and try again.", "error")
    return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large errors."""
    flash("File is too large. Maximum size is 16 MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
