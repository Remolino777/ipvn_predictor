from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

#ploating libraries
import streamlit as st 
# import pydeck as pdk

from src.inference import transform_ts_to_training_data,load_batch_time_series_from_store, load_model_from_registry, get_model_predictions, load_prediction_registry, download_prediction_registry_from_store

from src.paths import DATA_DIR
from src.plot import plot_x

st.set_page_config(layout='wide')

progress_bar = st.sidebar.header('Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

with st.spinner(text='Fetching batch of time serie data'):
    time_series = load_batch_time_series_from_store()
    st.sidebar.write('✅ Time series fetched from the store')
    progress_bar.progress(1/N_STEPS) 
    print(f'Time series:{time_series}')
    print(time_series.columns)
    
with st.spinner(text='Transforming time series to features'):
    features = transform_ts_to_training_data(time_series)
    st.sidebar.write('✅ Time series transformed')
    progress_bar.progress(2/N_STEPS) 
    print(f'{features}')

    
with st.spinner(text='Loading ML model from the registry'):
    model = load_model_from_registry()
    st.sidebar.write('✅ ML model was load from the registry')
    progress_bar.progress(3/N_STEPS) 
    
with st.spinner(text='Computing model predictions'):
    prediction = get_model_predictions(model, features)
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(4/N_STEPS) 
    
with st.spinner(text='Downloading prediction history'):
    prediction_history = download_prediction_registry_from_store() 
    st.sidebar.write('✅ History reclaimed')
    progress_bar.progress(5/N_STEPS) 

#Obtener el ultimo mes    

prediction_history['fecha'] = pd.to_datetime(prediction_history['fecha'])
prediction_history['fecha'] = prediction_history['fecha'].dt.date
  
#Verificar que es un nuevo valor de la prediccion se es asi. Actualizar los registros.
last_prediction = prediction_history['prediccion'].tail(1)

last_month = time_series.index[-1] #Obtener el ultimo indice, la columna fecha es el indice
last_month = pd.to_datetime(last_month)
next_month = last_month + pd.DateOffset(months=1)

 
pre_month = next_month.strftime("%B") 
 
if last_prediction.iloc[0] != prediction[0]:  #comparar dos series.
    
    
    # Crear el dataframe
    df_prediction = pd.DataFrame({
    'fecha':next_month,
    'prediccion':prediction
    }) 
    df_prediction['fecha'] = pd.to_datetime(df_prediction['fecha'])
    
    df_prediction['fecha'] = df_prediction['fecha'].dt.date  
    
    #La tienda requiere que fecha este en formato str
    df_prediction['fecha'] = df_prediction['fecha'].astype(str)
    prediction_history['fecha'] = prediction_history['fecha'].astype(str)

    prediction_update = pd.concat([prediction_history, df_prediction], axis=0)

    load_prediction_registry(prediction_update)
    print('Prediciton registry updated')

else:
    prediction_update = prediction_history
    print('No new predicitons')
        
 

#Imprimir el historico de predicciones y los datos reales.   
  
with st.spinner(text='Ploting time series data and predictions for '):
    
    plot_x(time_series, prediction_update)
    st.sidebar.write('✅ Graph plotted')
    progress_bar.progress(6/N_STEPS) 
    
st.subheader(f'Mes: {pre_month}')
st.subheader(f'IPVNBR Predecido: {prediction}')