import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Forza CPU

import pandas as pd
import joblib
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Caricamento

df = pd.read_csv("./output_elab/roma_subiaco_v2.csv")
X = df.iloc[:, :-1].values # 27 features
y = df.iloc[:, -1].values  # target_ip

# 2. Pre-processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, './output_elab/roma_subiaco_v2.gz')
# 3. Architettura RNA (Modello di Campo)
model = models.Sequential([
    layers.Input(shape=(27,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Addestramento RNA in corso...")
model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1)
model.save("./output_elab/roma_subiaco_v2.keras")

print("Modello salvato.")