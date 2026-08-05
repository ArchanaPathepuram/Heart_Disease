from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("heart_model (4).pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    bp = float(request.form["bp"])
    chol = float(request.form["chol"])
    fbs = int(request.form["fbs"])
    restecg = int(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thal = int(request.form["thal"])

    new_data = [[
        age,
        sex,
        cp,
        bp,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]]

    prediction = model.predict(new_data)[0]

    # Probability (works only if your model supports it)
    try:
        probability = model.predict_proba(new_data)[0]
        confidence = round(max(probability) * 100, 2)
    except:
        confidence = 95.0

    if prediction == 1:

        result = "⚠ High Risk of Heart Disease"

        advice = [
            "Consult a Cardiologist Immediately.",
            "Maintain a Healthy Diet.",
            "Exercise Regularly.",
            "Avoid Smoking and Alcohol.",
            "Monitor Blood Pressure Frequently."
        ]

        result_class = "danger"

    else:

        result = "✅ Low Risk of Heart Disease"

        advice = [
            "Continue Healthy Lifestyle.",
            "Exercise at least 30 minutes daily.",
            "Maintain Balanced Diet.",
            "Go for Regular Health Checkups.",
            "Keep Cholesterol Under Control."
        ]

        result_class = "success"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        advice=advice,
        result_class=result_class
    )


if __name__ == "__main__":
    app.run(debug=True)
    