import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_data
import dagshub
import mlflow

# --- Configure MLflow environment variables ---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/MeghaRathi-ee/house_price_predict.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

print("🔍 Checking DagsHub environment variables...")
print("DAGSHUB_USER:", os.getenv("DAGSHUB_USER"))
print("DAGSHUB_TOKEN exists:", bool(os.getenv("DAGSHUB_TOKEN")))
print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))

# --- Initialize manual MLflow tracking instead of dagshub.init() ---
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("House_Price_Predict")
print("✅ MLflow tracking set up for DagsHub")


def train():
    mlflow.set_experiment("House_Price_Predict")

    with mlflow.start_run():
        # Load data
        df = load_data()

        # Define features
        ordinal_features = ['BHK', 'bathrooms']
        categorical_features = ['locality', 'facing', 'parking']
        continuous_features = df.drop(ordinal_features + categorical_features + ['price_per_sqft'], axis=1).columns.tolist()

        # Handle missing values
        df[categorical_features] = df[categorical_features].fillna('Missing')
        df[ordinal_features + categorical_features] = df[ordinal_features + categorical_features].astype('category')

        # Split
        X = df.drop('price_per_sqft', axis=1)
        y = df['price_per_sqft']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        # Preprocessing
        preprocessor = ColumnTransformer([
            ('continuous', Pipeline([('scaler', StandardScaler())]), continuous_features),
            ('categorical', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
            ('ordinal', Pipeline([('ordinalenc', OrdinalEncoder())]), ordinal_features)
        ])

        # Model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', linear_model.LinearRegression())
        ])

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"✅ RMSE: {rmse:.2f}, R²: {r2:.2f}")

        # MLflow logging
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_param("model_type", "LinearRegression")

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # Log model artifact to MLflow/DagsHub
        mlflow.log_artifact("models/model.pkl")
        print("✅ Model saved and logged successfully at models/model.pkl")
        

if __name__ == "__main__":
    train()
