# Titanic Survival Prediction

A small Flask web app for predicting Titanic passenger survival using a saved machine learning model. This repository contains the app code, dataset, trained model artifact, and a simple HTML interface for submitting prediction requests.

## Project Structure

- `app.py` — Flask application that loads a saved model, accepts user input, and returns survival predictions.
- `requirements.txt` — Python dependencies required to run the app.
- `data/Titanic_Data.csv` — Raw Titanic dataset used for training.
- `models/titanic_logistic_regression_model.pkl` — Saved trained Logistic Regression model.
- `templates/index.html` — Web UI template used by Flask.
- `Notebooks/Titanic_model.ipynb` — Jupyter notebook with model training and export logic.

## Features

- Web-based prediction interface using Flask
- Supports form submissions and JSON API requests
- Loads a persisted model artifact from `models/`
- Handles feature encoding and prediction formatting

## Requirements

- Python 3.11 (or compatible)
- `venv` or another virtual environment tool

## Setup

```powershell
cd D:\Synoris_training\ml-training-day4
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you are using PowerShell and get an execution policy warning, run:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

## Run the App

```powershell
python app.py
```

Then open your browser at:

- `http://127.0.0.1:5000/`

## Usage

- Use the web form to enter passenger details and submit a prediction.
- The app also supports JSON requests to the `/predict` endpoint.

### Example JSON request

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 72.72,
  "Embarked": "C"
}
```

## Notes

- The app expects model artifacts in `models/`.
- If you retrain the model, ensure the feature columns are preserved or saved alongside the model.
- The Flask app is configured for development only. For production deployment, use a proper WSGI server.

## License

This repository has no license specified. Add one if you plan to distribute or share the project publicly.
