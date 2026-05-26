import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

def train_and_save_model(data_path, models_dir):
    print("Loading cleaned car dataset...")
    df = pd.read_csv(data_path)
    
    X = df[['brand', 'year', 'km_driven', 'fuel', 'transmission']]
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building Decision Tree Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['year', 'km_driven']),
            # handle_unknown='ignore' prevents crashes if a user enters a weird car brand later
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'fuel', 'transmission'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

    print("Pruning the Tree (GridSearchCV)...")
    param_grid = {
        'regressor__max_depth': [7, 10, 15, None],
        'regressor__min_samples_split': [10, 20, 50],
        'regressor__min_samples_leaf': [5, 10, 20]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print(f"Best Tree Parameters: {grid_search.best_params_}")

    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, 'car_valuation_pipeline.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    print(f"Valuation Engine saved to {artifact_path}")
    return best_pipeline, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_cars.csv')
    models_dir = os.path.join(base_dir, 'models')
    train_and_save_model(data_path, models_dir)