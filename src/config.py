import os
from pathlib import Path
from dotenv import load_dotenv

from src.paths  import PARENT_DIR



#load key-value pairs from .env file located in parent directory

load_dotenv(PARENT_DIR/'.env')

HOPSWORKS_PROJECT_NAME = 'tasa_de_interes_colombia'

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an environment file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'features_target_monthly_bogota'
FEATURE_GROUP_NAME_TS = 'time_series_24_months_bogota'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_NAME_PREDICTIONS = 'predictions'
FEATURE_VIEW_NAME = 'features_target_monthly_bogota'
FEATURE_VIEW_NAME_TS = 'time_series_24_months_bogota'
FEATURE_VIEW_NAME_PREDICTIONS = 'predictions_bogota'
FEATURE_VIEW_VERSION = 2


MODEL_NAME = 'monthly_ipvn_bogota'
MODEL_VERSION= 1
print