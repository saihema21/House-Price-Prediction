import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("data/housing.csv")

# FEATURES & TARGET
X = data[['area']]
y = data['price']

# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# -----------------------------
# PREDICTION SYSTEM
# -----------------------------
print("\n🏠 House Price Prediction System")

area = float(input("Enter area (sq ft): "))

prediction = model.predict([[area]])

print(f"\n💰 Estimated Price: {prediction[0]:,.2f}")