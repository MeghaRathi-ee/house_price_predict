import pandas as pd
import os

def load_data(data_path="data/houseprices.csv"):
    # Get absolute path relative to project root
    base_path = os.path.dirname(os.path.dirname(__file__))  # goes up one folder from src
    full_path = os.path.join(base_path, data_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data not found at {full_path}")

    df = pd.read_csv(full_path)
    print(f"✅ Data loaded successfully! Shape = {df.shape}")
    return df
