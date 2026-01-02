"""
Script untuk melatih model GBR dari camera_dataset.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("=" * 60)
print("📦 Camera GBR Model Training")
print("=" * 60)

# ===== 1️⃣ Load Dataset dari CSV =====
df = pd.read_csv("camera_dataset.csv")
print(f"\n📊 Dataset loaded: {len(df)} samples")
print(f"📋 Columns: {list(df.columns)}")

# ===== 2️⃣ Preprocessing =====

# Extract Brand dari Model (ambil kata pertama)
def extract_brand(model_name):
    if pd.isna(model_name):
        return "Unknown"
    
    # Pattern matching untuk brand kamera
    model_lower = str(model_name).lower()
    
    brand_patterns = {
        "Canon": ["canon", "powershot", "eos", "ixus"],
        "Sony": ["sony", "cyber-shot", "dsc"],
        "Nikon": ["nikon", "coolpix"],
        "Fujifilm": ["fuji", "finepix", "fujifilm"],
        "Olympus": ["olympus"],
        "Casio": ["casio"],
        "Pentax": ["pentax"],
        "Panasonic": ["panasonic", "lumix"],
        "Samsung": ["samsung"],
        "Kodak": ["kodak"],
        "Agfa": ["agfa"],
        "Toshiba": ["toshiba"],
        "Ricoh": ["ricoh"],
        "Konica": ["konica"],
        "Minolta": ["minolta"],
        "Sanyo": ["sanyo"],
        "Epson": ["epson"],
        "HP": ["hp"],
        "Leica": ["leica"],
        "JVC": ["jvc"],
        "Contax": ["contax"],
        "Sigma": ["sigma"],
        "Kyocera": ["kyocera"]
    }
    
    for brand, keywords in brand_patterns.items():
        for keyword in keywords:
            if keyword in model_lower:
                return brand
    
    # Default: ambil kata pertama
    return str(model_name).split()[0]

df["brand"] = df["Model"].apply(extract_brand)

# Encode brand
encoder = LabelEncoder()
df["brand_id"] = encoder.fit_transform(df["brand"])

print("\n📋 Brand Distribution:")
for brand, count in df["brand"].value_counts().items():
    id_val = encoder.transform([brand])[0] if brand in encoder.classes_ else -1
    print(f"   {brand} (id:{id_val}): {count} cameras")

# Kolom Dimensions sudah berupa volume (mm³) - tidak perlu parsing
df["dimension_volume"] = df["Dimensions"]

print(f"\n📊 Dimension volume stats:")
print(f"   Min: {df['dimension_volume'].min()}")
print(f"   Max: {df['dimension_volume'].max()}")
print(f"   Mean: {df['dimension_volume'].mean():.2f}")

# ===== 3️⃣ Clean Data =====
# Hapus rows dengan missing values di kolom penting
df_clean = df.dropna(subset=["Effective pixels", "Weight (inc. batteries)", "dimension_volume", "Price"])

# Hapus outliers (weight atau dimension = 0)
df_clean = df_clean[
    (df_clean["Weight (inc. batteries)"] > 0) & 
    (df_clean["dimension_volume"] > 0) &
    (df_clean["Effective pixels"] > 0)
]

print(f"\n📊 Dataset setelah cleaning: {len(df_clean)} samples")

# ===== 4️⃣ Features & Target =====
feature_names = ["brand_id", "Effective pixels", "Weight (inc. batteries)", "dimension_volume"]
X = df_clean[feature_names]
y = df_clean["Price"]

print(f"\n📊 Features: {feature_names}")
print(f"📊 Target: Price")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📈 Training samples: {len(X_train)}")
print(f"📈 Test samples: {len(X_test)}")

# ===== 5️⃣ Train Model =====
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=3
)

model.fit(X_train, y_train)
print("\n✅ Model trained!")

# ===== 6️⃣ Evaluate =====
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   RMSE: ${rmse:.2f}")
print(f"   R² Score: {r2:.4f}")

print("\n🎯 Feature Importance:")
for name, imp in sorted(zip(feature_names, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {name}: {imp:.4f}")

# ===== 7️⃣ Save Model =====
model_data = {
    "model": model,
    "encoder": encoder,
    "feature_names": feature_names,
    "performance": {"rmse": float(rmse), "r2": float(r2)}
}

joblib.dump(model_data, "gbr_camera.pkl")
print(f"\n💾 Model saved: gbr_camera.pkl")

# ===== 8️⃣ Test Predictions =====
print("\n" + "=" * 60)
print("🎯 Sample Predictions (Test Set):")
print("=" * 60)

sample_indices = [0, 5, 10, 15, 20]
for idx in sample_indices:
    if idx < len(X_test):
        sample = X_test.iloc[idx]
        actual = y_test.iloc[idx]
        predicted = model.predict([sample])[0]
        
        # Find brand name
        brand_id = int(sample["brand_id"])
        brand_name = encoder.inverse_transform([brand_id])[0]
        
        print(f"\n   Sample {idx+1}:")
        print(f"   Brand: {brand_name} (id:{brand_id})")
        print(f"   Effective pixels: {sample['Effective pixels']}")
        print(f"   Weight: {sample['Weight (inc. batteries)']}g")
        print(f"   Volume: {sample['dimension_volume']:.0f}mm³")
        print(f"   Actual Price: ${actual:.2f}")
        print(f"   Predicted Price: ${predicted:.2f}")
        print(f"   Error: ${abs(actual - predicted):.2f}")

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("=" * 60)

