import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model import train_and_save_model

def evaluate_performance(pipeline, X_test, y_test):
    print("\n--- Model Evaluation ---")
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Popularity Points")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} Popularity Points")
    print(f"R-Squared (R2 Score):           {r2:.4f}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_spotify_tracks.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    pipeline, X_test, y_test = train_and_save_model(data_path, models_dir)
    evaluate_performance(pipeline, X_test, y_test)