"""
day03_instance_vs_model.py
Goal: Reproduce the book's GDP-vs-life-satisfaction example with a
model-based approach (linear regression) and an instance-based
approach (k-nearest neighbors), and compare their predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Load the data (same source the book uses)
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# --- MODEL-BASED: Linear Regression ---
model_based = LinearRegression()
model_based.fit(X, y)
theta_0 = model_based.intercept_[0]
theta_1 = model_based.coef_[0][0]
print(f"Model-based (Linear Regression): theta_0={theta_0:.4f}, theta_1={theta_1:.6e}")

# --- INSTANCE-BASED: k-Nearest Neighbors (k=3) ---
instance_based = KNeighborsRegressor(n_neighbors=3)
instance_based.fit(X, y)

# Predict for Puerto Rico (not in the OECD data)
X_new = [[33_442.8]]
pred_model_based = model_based.predict(X_new)
pred_instance_based = instance_based.predict(X_new)

print(f"\nPrediction for Puerto Rico (GDP/capita = $33,442.80):")
print(f"  Model-based (linear regression): {pred_model_based[0][0]:.2f}")
print(f"  Instance-based (3-NN):           {pred_instance_based[0][0]:.2f}")

# Show WHICH 3 neighbors the k-NN model actually used -- this is the
# part that makes instance-based learning tangible rather than abstract.
distances, indices = instance_based.kneighbors(X_new)
print("\n3 nearest neighbors used by k-NN:")
for idx in indices[0]:
    country_row = lifesat.iloc[idx]
    print(f"  GDP/capita={country_row['GDP per capita (USD)']:.0f}, "
          f"life satisfaction={country_row['Life satisfaction']}")