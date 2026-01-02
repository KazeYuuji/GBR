from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model GBR beserta encoder dan feature names
model_data = joblib.load("gbr_camera.pkl")
model = model_data["model"]
encoder = model_data["encoder"]
feature_names = model_data["feature_names"]

print("=" * 60)
print("🚀 Camera GBR API Started!")
print("=" * 60)
print(f"📊 Dataset: camera_dataset.csv (979 samples)")
print(f"📋 Features: {feature_names}")
print(f"📊 Model Performance: R²={model_data['performance']['r2']:.4f}, RMSE=${model_data['performance']['rmse']:.2f}")
print(f"📷 Brands: {list(encoder.classes_)}")
print("=" * 60)

# Mapping untuk brand detection dari nama model
BRAND_PATTERNS = {
    "Canon": ["canon", "powershot", "eos", "ixus"],
    "Sony": ["sony", "cyber-shot", "dsc"],
    "Nikon": ["nikon", "coolpix"],
    "Fujifilm": ["fuji", "finepix"],
    "Olympus": ["olympus"],
    "Casio": ["casio"],
    "Pentax": ["pentax"],
    "Panasonic": ["panasonic", "lumix"],
    "Samsung": ["samsung"],
    "Kodak": ["kodak"],
    "Agfa": ["agfa"],
    "Toshiba": ["toshiba"],
    "Ricoh": ["ricoh"],
    "Leica": ["leica"],
    "HP": ["hp"],
    "Epson": ["epson"],
    "Sigma": ["sigma"],
    "Contax": ["contax"],
    "JVC": ["jvc"],
    "Sanyo": ["sanyo"],
    "Konica": ["konica"],
    "Minolta": ["minolta"],
    "Kyocera": ["kyocera"]
}


def detect_brand(model_name):
    """Deteksi brand dari nama model kamera"""
    if pd.isna(model_name):
        return "Unknown"
    model_lower = str(model_name).lower()
    for brand, keywords in BRAND_PATTERNS.items():
        for keyword in keywords:
            if keyword in model_lower:
                return brand
    return "Unknown"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(f"\n📥 Received data: {data}")

    try:
        # ===== DUKUNG DUA MODE INPUT =====
        
        # Mode 1: Input user-friendly
        if 'Brand' in data:
            # Deteksi brand
            brand = data['Brand']
            if brand.lower() in ['auto', 'detect', '']:
                if 'Model' in data:
                    brand = detect_brand(data['Model'])
                else:
                    return jsonify({"error": "Need 'Model' name for auto-detect"}), 400
            
            # Encode brand
            try:
                brand_encoded = encoder.transform([brand])[0]
            except ValueError:
                brand = detect_brand(brand)
                try:
                    brand_encoded = encoder.transform([brand])[0]
                except ValueError:
                    brand_encoded = 0  # Default ke brand pertama
            
            # Dimension sudah berupa volume (tidak perlu WxHxD)
            dimension_volume = float(data.get('Dimensions', data.get('Dimension', 100)))
            
            features = [
                float(brand_encoded),
                float(data['Effective pixels']),
                float(data['Weight']),
                dimension_volume
            ]
            
            result = {
                "price": round(float(model.predict(np.array(features).reshape(1, -1))[0]), 2),
                "brand": brand,
                "brand_id": int(brand_encoded),
                "effective_pixels": float(data['Effective pixels']),
                "weight": float(data['Weight']),
                "dimension_volume": dimension_volume
            }
        
        # Mode 2: Input lengkap (sama seperti fitur training)
        elif all(k in data for k in ['brand_id', 'Effective pixels', 'Weight (inc. batteries)', 'dimension_volume']):
            features = [
                float(data['brand_id']),
                float(data['Effective pixels']),
                float(data['Weight (inc. batteries)']),
                float(data['dimension_volume'])
            ]
            
            brand_id = int(data['brand_id'])
            try:
                brand_name = encoder.inverse_transform([brand_id])[0]
            except:
                brand_name = "Unknown"
            
            result = {
                "price": round(float(model.predict(np.array(features).reshape(1, -1))[0]), 2),
                "brand": brand_name,
                "brand_id": brand_id,
                "effective_pixels": float(data['Effective pixels']),
                "weight": float(data['Weight (inc. batteries)']),
                "dimension_volume": float(data['dimension_volume'])
            }
        
        else:
            return jsonify({
                "error": "Invalid input format",
                "mode_1_example": {
                    "Brand": "Canon",
                    "Model": "Canon PowerShot A100",  # optional, untuk auto-detect
                    "Effective pixels": 3.0,
                    "Weight": 225,
                    "Dimensions": 110  # sudah volume (mm³)
                },
                "mode_2_example": {
                    "brand_id": 1,
                    "Effective pixels": 3.0,
                    "Weight (inc. batteries)": 225,
                    "dimension_volume": 110
                }
            }), 400

        print(f"🎯 Prediction: ${result['price']}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-page")
def predict_page():
    return render_template("index.html")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/brands", methods=['GET'])
def get_brands():
    """Get all supported brands"""
    brands = {brand: int(idx) for idx, brand in enumerate(encoder.classes_)}
    return jsonify({
        "brands": brands,
        "total": len(encoder.classes_),
        "features_used": feature_names
    })


@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "GradientBoostingRegressor",
        "features": feature_names,
        "performance": model_data["performance"],
        "brands_count": len(encoder.classes_)
    })


@app.route("/test", methods=['POST'])
def test():
    """Quick test endpoint"""
    sample = request.json or {
        "Brand": "Canon",
        "Effective pixels": 12.0,
        "Weight": 500,
        "Dimensions": 120
    }
    
    try:
        brand = sample.get('Brand', 'Canon')
        brand_encoded = encoder.transform([brand])[0]
        dim_vol = float(sample.get('Dimensions', 120))
        
        features = [float(brand_encoded), float(sample['Effective pixels']), float(sample['Weight']), dim_vol]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]
        
        return jsonify({
            "input": sample,
            "processed": {
                "brand_id": int(brand_encoded),
                "effective_pixels": float(sample['Effective pixels']),
                "weight": float(sample['Weight']),
                "dimension_volume": dim_vol
            },
            "predicted_price": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host='0.0.0.0', port=port)

