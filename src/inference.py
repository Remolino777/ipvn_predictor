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

def load_prediction_registry(df: pd.DataFrame):
    import hopsworks
    import src.config as config
    
    #connect to hopsworks
    project= hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    
    feature_store = project.get_feature_store()
    
    feature_group = feature_store.get_or_create_feature_group(
                                                            name= config.FEATURE_GROUP_NAME_PREDICTIONS,
                                                            version= config.FEATURE_GROUP_VERSION ,
                                                            description= 'Predictions',
                                                            primary_key= ['fecha']
    )
    
    feature_group.insert(df, write_options={'wait_for_job':False})
    
    try:
        # Intentar obtener el Feature View existente
        feature_view = feature_store.get_feature_view(name=config.FEATURE_VIEW_NAME_PREDICTIONS, version=config.FEATURE_VIEW_VERSION)
        
        # Si el Feature View existe, eliminarlo para sobrescribirlo
        if feature_view:
            feature_store.delete_feature_view(name=config.FEATURE_VIEW_NAME_PREDICTIONS, version=config.FEATURE_VIEW_VERSION)
            print(f"Feature View '{config.FEATURE_VIEW_NAME_PREDICTIONS}' versión '{config.FEATURE_VIEW_VERSION}' eliminado correctamente.")
        
        # Crear el nuevo Feature View
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_NAME_PREDICTIONS,
            version=config.FEATURE_VIEW_VERSION,
            query=feature_group.select_all()
        )
        print(f"Feature View '{config.FEATURE_VIEW_NAME_PREDICTIONS}' versión '{config.FEATURE_VIEW_VERSION}' creado exitosamente.")
    
    except Exception as e:
        print(f"Error al crear o actualizar Feature View '{config.FEATURE_VIEW_NAME_PREDICTIONS}' versión '{config.FEATURE_VIEW_VERSION}': {str(e)}")
        
def download_prediction_registry_from_store() -> pd.DataFrame:
    
    fs = get_feature_store()
    
    #Get feature view
    feature_view = fs.get_feature_view(
    name=config.FEATURE_VIEW_NAME_PREDICTIONS,
    version= config.FEATURE_VIEW_VERSION
    )
    data, _ = feature_view.training_data(
    description= 'monthly_predictions_bogota'
    )
    
    #print(ts)
    return data