from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm 
import hopsworks
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

from src.paths import RAW_DATA_DIR, PARENT_DIR, TRANSFORMED_DATA_DIR

project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value = HOPSWORKS_API_KEY
)


def load_and_validate_data(response):
    try:        
        print(response)
        if response.status_code == 200:
            excel_file = response.content
            
            df = pd.read_excel(excel_file,
                        engine='openpyxl',
                        sheet_name='Serie_IPVNBR',
                        header=2,
                        skiprows=2,
                        skipfooter=8
                        )
            return df
        else:
            print('Invalid URL: Received status code', response.status_code)
    except Exception as e:
        print('An error occurred:', e)
        
def transform_raw_data_into_ts_data(df):
    
    c_list = ['bogota', 'alrededores de bogota', 'medellin', 'cali']
    
    df1 = df[['Unnamed: 0','Bogotá.1','Alrededores de Bogotá4,5','Medellín.1','Cali.1']]
    df1 = df1[df1['Unnamed: 0'] >= '2006-01-01']
    #Rename columns
    df1 = df1.rename(columns={'Unnamed: 0':'fecha',
                    'Bogotá.1':'bogota',
                    'Alrededores de Bogotá4,5':'alrededores de bogota',
                    'Medellín.1':'medellin',
                    'Cali.1':'cali'
                    })
    dataframes_list = []
    
    for city in c_list:
        df_place = df1[['fecha', city]]  # Crear el DataFrame con la columna correspondiente
        dataframes_list.append(df_place)  # Guardar el DataFrame en el diccionario con la ciudad como clave
    
    return dataframes_list  



# Funcion para cortar indices y cerar los dataframes
def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = n_features
        subseq_last_idx = n_features + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices
    
def transform_ts_data_to_features_and_target(df, ciudad):
        
    df_c = df
    df_c.reset_index(drop=True, inplace=True)  # Asegúrate de que los índices se restablecen

    n_features = 12
    step_size = 1

    indices = get_cutoff_indices(df_c, n_features, step_size)
    
    n_examples = len(indices)
    x = np.ndarray(shape=(n_examples, n_features), dtype=np.float32)
    y = np.ndarray(shape=(n_examples), dtype=np.float32)
    
    months = []

    for u, idx in tqdm(enumerate(indices)):
        x[u, :] = df_c.iloc[idx[0]:idx[1]][ciudad].values
        y[u] = df_c.iloc[idx[1]:idx[2]][ciudad].values  # Asegúrate de seleccionar la columna correcta
        months.append(df_c.iloc[idx[1]]['fecha'])
        
        
        # Convertir la lista de meses a un DataFrame
    # months = pd.to_datetime(months)
    # months_df = pd.DataFrame(months, columns=['fecha'])
    # months_df['month'] = months_df['fecha'].dt.to_period('M').astype(str)  # Por compatibilidad con la hopsworks store
    
        
    feature = pd.DataFrame(x, columns=[f'month_{j+1}' for j in range(n_features)])    
    target = pd.DataFrame(y, columns=['Target'])
    
    # Crear el DataFrame con los datos concatenados.
    df_ready = pd.concat([feature, target], axis=1)
    
    return df_ready
    
   
    
 
           
        
    

#  feature_group = feature_store.get_or_create_feature_group(
#                                                                 name= feature_group_name,
#                                                                 version= group_version,
#                                                                 description= 'features and target',
#                                                                 primary_key= ['Target'],
#                                                                 event_time='month'
#     )
#    feature_group.insert(df_ready, write_options={'wait_for_job':False})
#     # Exportar los datos al archivo train