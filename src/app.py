from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

#ploating libraries
import streamlit as st 
# import pydeck as pdk

from src.inference import transform_ts_to_training_data, load_batch_time_series_from_store, load_model_from_registry, get_model_predictions

from src.paths import DATA_DIR
from src.plot import plot_one_sample, plot_ts

st.set_page_config(layout='wide')


# Obtener la fecha actual
current_date = datetime.today()

# Calcular el próximo mes
prediction_month = current_date.replace(day=28) + timedelta(days=4)  # Esto asegurará que estemos en el siguiente mes
prediction_month = prediction_month.replace(day=1)  # Establecer el día al primero del mes siguiente

# Obtener el nombre del próximo mes en español
prediction_month_name = prediction_month.strftime("%B")

st.header(f"Bogotá's city new home price index (IPVN) prediction for the month of {prediction_month_name}")

progress_bar = st.sidebar.header('Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner(text='Fetching batch of time serie data'):
    features = load_batch_time_series_from_store()
    st.sidebar.write('✅ Time series fetched from the store')
    progress_bar.progress(1/N_STEPS) 
    print(f'{features}')
    
with st.spinner(text='Transforming time series to features'):
    features = transform_ts_to_training_data(features)
    st.sidebar.write('✅ Time series transformed')
    progress_bar.progress(2/N_STEPS) 
    print(f'{features}')

    
with st.spinner(text='Loading ML model from the registry'):
    model = load_model_from_registry()
    st.sidebar.write('✅ ML model was load from the registry')
    progress_bar.progress(3/N_STEPS) 
    
with st.spinner(text='Computing model predictions'):
    results = get_model_predictions(model, features)
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(4/N_STEPS) 
    
