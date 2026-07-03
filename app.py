from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Load Model and Training Columns
# -----------------------------
MODEL_PATH = "models/titanic_logistic_regression_model.pkl"
COLUMNS_PATH = "models/columns.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}!")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

model_columns = None
if os.path.exists(COLUMNS_PATH):
    with open(COLUMNS_PATH, "rb") as f:
        model_columns = pickle.load(f)
else:
    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                model_columns = list(step.feature_names_in_)
                break

if model_columns is None:
    raise FileNotFoundError(
        f"columns.pkl not found at {COLUMNS_PATH}! "
        "Save the training columns to this file or pickle a model with feature_names_in_."
    )


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Running",
        "model": "Titanic Survival Prediction API"
    })


# -----------------------------
# Prediction Function
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        if request.is_json:

            data = request.get_json()

        else:

            data = {
                "Pclass": request.form["Pclass"],
                "Sex": request.form["Sex"],
                "Age": request.form["Age"],
                "SibSp": request.form["SibSp"],
                "Parch": request.form["Parch"],
                "Fare": request.form["Fare"],
                "Embarked": request.form["Embarked"]
            }

        required_fields = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        df = pd.DataFrame([data])

        numeric_cols = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # One-Hot Encoding
        df = pd.get_dummies(
            df,
            columns=["Sex", "Embarked"],
            drop_first=True
        )

        # Match Training Columns
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][1]

        result = "Survived" if prediction == 1 else "Did Not Survive"

        # JSON Request
        if request.is_json:
            return jsonify({
                "prediction": int(prediction),
                "result": result,
                "survival_probability": round(float(probability), 4)
            })

        # HTML Form Request
        return render_template(
            "index.html",
            prediction=result,
            probability=round(float(probability) * 100, 2)
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)