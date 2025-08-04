"""
Simple model serving with Flask.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# load model from registry
mlflow.set_tracking_uri("http://host.docker.internal:5001")
model_name = "playtime-prediction-model"
stage = "Staging"
model_uri = f"models:/{model_name}/{stage}"

print(f"Loading model: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data] if isinstance(data, dict) else data)
    
    # use same features as in your script
    features = ["main_story", "main_story_polled", "main_plus_sides", "main_plus_sides_polled"]
    X = df[features]
    
    predictions = model.predict(X)
    
    if len(predictions) == 1:
        return jsonify({"prediction": float(predictions[0])})
    else:
        return jsonify({"predictions": [float(p) for p in predictions]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)