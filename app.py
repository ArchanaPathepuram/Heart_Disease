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

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
    