import pickle
import pandas as pd
import os

def load_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, 'models', 'dt_pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)

def run_inference(input_dict):
    pipeline = load_pipeline()
    
    # Backend feature engineering execution
    input_dict['dance_energy_index'] = input_dict['danceability'] * input_dict['energy']
    
    df_input = pd.DataFrame([input_dict])
    prediction = pipeline.predict(df_input)[0]
    
    return prediction