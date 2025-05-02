import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("sports-car-prices-dataset/Sport car price.csv")

# Pastikan semua nama kolom sesuai dengan dataset
# Contoh: 'Car_Make_370Z' adalah dummy variable hasil one-hot encoding
# Jika belum ada, kamu harus melakukan encoding terlebih dahulu
# Berikut adalah contoh encoding:
if 'Car_Make' in df.columns:
    df = pd.get_dummies(df, columns=['Car_Make'], drop_first=True)

# Periksa kolom-kolom yang tersedia
print("Available columns:", df.columns)

# Pilih fitur (ganti nama sesuai kolom yang tersedia)
X = df[['horsepower', 'engine_size', 'year', 'Car_Make_370Z']]  # pastikan 'Car_Make_370Z' benar
y = df['price']  # Sesuaikan jika kolom target-nya punya nama lain

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(model, 'sports_car_price_predictor.pkl')

print("✅ Model berhasil dilatih ulang dan disimpan sebagai 'sports_car_price_predictor.pkl'")