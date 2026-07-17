from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import os

# ==========================================================
# Flask App
# ==========================================================

app = Flask(
    __name__,
    template_folder="website/templates",
    static_folder="website/static"
)

@app.route("/health")
def health():
    return "OK"

# ==========================================================
# Load Model
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest.pkl"
)

model = joblib.load(MODEL_PATH)

print("Random Forest Model Loaded Successfully!")

# ==========================================================
# SHAP Explainer
# ==========================================================

explainer = shap.TreeExplainer(model)

print("SHAP Explainer Created Successfully!")

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["Age"])
    gender = int(request.form["Gender"])
    polyuria = int(request.form["Polyuria"])
    polydipsia = int(request.form["Polydipsia"])
    sudden_weight_loss = int(request.form["SuddenWeightLoss"])
    weakness = int(request.form["Weakness"])
    polyphagia = int(request.form["Polyphagia"])
    genital_thrush = int(request.form["GenitalThrush"])
    visual_blurring = int(request.form["VisualBlurring"])
    itching = int(request.form["Itching"])
    irritability = int(request.form["Irritability"])
    delayed_healing = int(request.form["DelayedHealing"])
    partial_paresis = int(request.form["PartialParesis"])
    muscle_stiffness = int(request.form["MuscleStiffness"])
    alopecia = int(request.form["Alopecia"])
    obesity = int(request.form["Obesity"])

    patient = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Polyuria": [polyuria],
        "Polydipsia": [polydipsia],
        "SuddenWeightLoss": [sudden_weight_loss],
        "Weakness": [weakness],
        "Polyphagia": [polyphagia],
        "GenitalThrush": [genital_thrush],
        "VisualBlurring": [visual_blurring],
        "Itching": [itching],
        "Irritability": [irritability],
        "DelayedHealing": [delayed_healing],
        "PartialParesis": [partial_paresis],
        "MuscleStiffness": [muscle_stiffness],
        "Alopecia": [alopecia],
        "Obesity": [obesity]
    })

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    shap_values = explainer.shap_values(patient)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        diabetes_shap = shap_values[1][0]
    else:
        diabetes_shap = shap_values[:, :, 1][0]

    contribution = pd.DataFrame({
        "Feature": patient.columns,
        "Value": patient.iloc[0].values,
        "SHAP": diabetes_shap
    })

    contribution["Absolute SHAP"] = contribution["SHAP"].abs()

    contribution["Contribution (%)"] = (
        contribution["Absolute SHAP"]
        / contribution["Absolute SHAP"].sum()
    ) * 100

    contribution = contribution.sort_values(
        by="Contribution (%)",
        ascending=False
    )

    display_values = []

    for feature, value in zip(
        contribution["Feature"],
        contribution["Value"]
    ):

        if feature == "Gender":
            display_values.append("Male" if value == 1 else "Female")

        elif feature == "Age":
            display_values.append(int(value))

        else:
            display_values.append("Yes" if value == 1 else "No")

    result = "Positive" if prediction == 1 else "Negative"

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.70:
        risk = "Moderate"
    else:
        risk = "High"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2),
        risk=risk,
        feature_names=contribution["Feature"].tolist(),
        input_values=display_values,
        feature_values=contribution["Contribution (%)"].round(2).tolist()
    )

# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)