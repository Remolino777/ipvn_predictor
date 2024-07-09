import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:
    
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    
def get_feature_store() -> FeatureStore:
    
    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features:pd.DataFrame) -> float:
    
    predictions = model.predict(features)
    print(predictions)
    return predictions

# def load_batch_features_from_store() -> pd.DataFrame:
    
#     fs = get_feature_store()
    
#     #Get feature view
#     feature_view = fs.get_feature_view(
#     name=config.FEATURE_VIEW_NAME,
#     version= config.FEATURE_GROUP_VERSION
#     )
#     ft_data, _ = feature_view.training_data(
#     description= 'month_features_target_bogota'
#     )
#     # obtain the las 11 values 
#     ft = ft_data.tail(1)
#     ft.drop('month_1', axis=1, inplace=True)

#     return ft_data

def load_batch_time_series_from_store() -> pd.DataFrame:
    
    fs = get_feature_store()
    
    #Get feature view
    feature_view = fs.get_feature_view(
    name=config.FEATURE_VIEW_NAME_TS,
    version= config.FEATURE_VIEW_VERSION
    )
    ts_data, _ = feature_view.training_data(
    description= 'monthly_time_serie_data_bogota'
    )
    df_sorted = ts_data.sort_values(by='fecha')
    # obtain the las 11 values 
    ts = df_sorted.tail(12)
    #print(ts)
    return ts

def transform_ts_to_training_data(df) -> pd.DataFrame:
    df.set_index('fecha', inplace=True)
    df_transposed = df.T
    return df_transposed

def load_model_from_registry():
    import joblib
    from pathlib import Path
    
    project = get_hopsworks_project()
    
    model_registry = project.get_model_registry()
    
    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version= config.MODEL_VERSION
    )
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'ml_bogota.pkl')
    return model