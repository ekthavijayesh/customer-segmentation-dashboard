
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features for clustering
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].dropna()

# Create a pipeline that scales the data and trains K-Means
model = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    ))
])

# Train the model
model.fit(X)

# Save the trained model and feature names
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "customer_segmentation_model.joblib"
)

print("Model trained successfully!")
print("Number of customers:", len(X))
print("Number of clusters:", 5)
print("Model saved successfully!")
