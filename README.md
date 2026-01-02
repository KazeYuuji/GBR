# Camera GBR Project

Project Machine Learning untuk memprediksi harga kamera menggunakan **Gradient Boosting Regression (GBR)**.

## 📊 Model Details

| Aspek | Detail |
|-------|--------|
| Algorithm | Gradient Boosting Regressor |
| Libraries | scikit-learn, Flask, Pandas |
| Features | brand_id, megapixel, weight, dimension_volume |

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (Opsional, jika ingin retrain)
```bash
python train_model.py
```

### 3. Jalankan API
```bash
python app.py
```

API akan berjalan di `http://localhost:5000`

## 📡 API Endpoints

### GET /
```bash
Response: "Camera GBR Model Running!"
```

### GET /health
```bash
Response:
{
  "status": "healthy",
  "model": "GradientBoostingRegressor",
  "features": ["brand_id", "megapixel", "weight", "dimension_volume"]
}
```

### POST /predict
```bash
URL: http://localhost:5000/predict
Method: POST
Content-Type: application/json

Request Body:
{
  "Brand": "Sony",
  "Megapixel": 24,
  "Weight": 450,
  "Width": 127,
  "Height": 94,
  "Depth": 60
}

Response:
{
  "price": 1053.33,
  "brand_encoded": 2,
  "dimension_volume": 707280
}
```

## 🔧 Cara Kerja

```
┌─────────────────────────────────────────────────────────────┐
│                    FLOW DIAGRAM                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. TRAINING (train_model.py)                               │
│     ┌──────────────┐    ┌──────────────┐    ┌─────────────┐│
│     │   Raw Data   │───▶│  Preprocess  │───▶│    Train    ││
│     │  (brand, MP, │    │  - Encode    │    │   GBR       ││
│     │   weight,    │    │  - Volume    │    │   Model     ││
│     │   dims)      │    │  calculation │    │             ││
│     └──────────────┘    └──────────────┘    └──────┬──────┘│
│                                                    │       │
│                                                    ▼       │
│                                             ┌─────────────┐│
│                                             │  Save to    ││
│                                             │gbr_camera   ││
│                                             │    .pkl     ││
│                                             └─────────────┘│
│                                                              │
│  2. PREDICTION (app.py)                                     │
│     ┌──────────────┐    ┌──────────────┐    ┌─────────────┐│
│     │   User API   │───▶│  Preprocess  │───▶│  Predict    ││
│     │  Request     │    │  - Encode    │    │   with GBR  ││
│     │              │    │  - Calculate │    │    Model    ││
│     │              │    │    Volume    │    │             ││
│     └──────────────┘    └──────────────┘    └──────┬──────┘│
│                                                    │       │
│                                                    ▼       │
│                                             ┌─────────────┐│
│                                             │   Return    ││
│                                             │   Price     ││
│                                             └─────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Feature Mapping

| Input Field | Processing | Output Feature |
|-------------|------------|----------------|
| Brand | `LabelEncoder` | `brand_id` (0-4) |
| Megapixel | Direct | `megapixel` |
| Weight | Direct | `weight` |
| Width, Height, Depth | `Width × Height × Depth` | `dimension_volume` |

### Brand Encoding:
- Canon -> 0
- Fuji -> 1
- Nikon -> 2
- Sony -> 3

## 📁 File Structure

```
GBR/
├── app.py              # Flask API
├── train_model.py      # Script untuk train model
├── camera.ipynb        # Jupyter notebook (development)
├── gbr_camera.pkl      # Model terlatih
├── requirements.txt    # Dependencies
├── Procfile            # Railway deployment
├── railway.json        # Railway config
└── README.md           # Dokumentasi
```

## 🧪 Test dengan cURL

```bash
# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Brand": "Sony",
    "Megapixel": 24,
    "Weight": 450,
    "Width": 127,
    "Height": 94,
    "Depth": 60
  }'
```

## ☁️ Deployment ke Railway

1. Push ke GitHub
2. Connect ke Railway
3. Railway akan auto-detect dan deploy

## ⚠️ Catatan

- Dataset training sangat kecil (8 sampel)
- Model mungkin tidak akurat untuk data baru
- Disarankan menggunakan dataset yang lebih besar

