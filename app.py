from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model GBR
model = joblib.load("gbr_camera.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    print(data)

    # Pastikan urutan fitur sama seperti training model
    features = [
        float(data['Brand']),
        float(data['Megapixel']),
        float(data['Weight']),
        float(data['Dimensions'])
    ]

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({"price": round(float(prediction), 2)})

@app.route("/")
def home():
    return "Camera GBR Model Running!"

if __name__ == "__main__":
    app.run(debug=True)
