import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd

# GANTI dengan Run ID kamu
MODEL_URI = "runs:/805e434c634e4176805c738792a9247f/model"

model = mlflow.pyfunc.load_model(MODEL_URI)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    preds = model.predict(df)
    return jsonify(preds.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
