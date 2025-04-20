import pickle
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import os

os.makedirs("Model", exist_ok=True)

# Dummy training data
X = np.array([
    [2011, 0, 63.5, 0],
    [2012, 1, 85.0, 1],
    [2013, 2, 87.0, 2],
    [2014, 3, 21.5, 3],
    [2015, 4, 163.1, 4],
    [2016, 5, 23.6, 5],
])
y = np.array([2.5, 4.1, 6.0, 1.2, 10.5, 3.3])

model = DecisionTreeRegressor()
model.fit(X, y)

with open("Model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved at Model/model.pkl")
