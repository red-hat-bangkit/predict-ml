import pandas as pd
import numpy as np
import pickle
import json
from datetime import timedelta
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Transformer:
  """Convert non numeric categorical data to numeric value
  """
  def __init__(self):
    self.mapping_keys = {}
    self.mapping_location = os.path.join(BASE_DIR, "transformer-map.json")

  def load(self):
    self.mapping_keys = json.load(open(self.mapping_location, "r"))
    return self.mapping_location

  def fit_transform(self, df):
    columns = df.columns.values
    self.mapping_keys = {}

    for column in columns:
      text_digit_vals = {}
      if df[column].dtype != np.int64 and df[column].dtype != np.float64:
        column_contents = df[column].values.tolist()
        unique_elements = sorted(set(column_contents), key=lambda x: str(x))
        x = 0
        for unique in unique_elements:
          if unique not in text_digit_vals:
            text_digit_vals[unique] = x
            x += 1
        df[column] = df.apply(lambda row: text_digit_vals[row[column]], axis=1)
        self.mapping_keys[column] = text_digit_vals

      return df

  def transform(self, df):
    columns = df.columns.values
    for column in columns:
      if df[column].dtype != np.int64 and df[column].dtype != np.float64:
        df[column] = df.apply(lambda row: self.mapping_keys[column][row[column]], axis=1)
    return df
  
  def save_mapping(self):
    json.dump(self.mapping_keys, open(self.mapping_location, "w"))
    return self.mapping_location

# Load datasheet
df_tmrain_tmwater_days = pd.read_csv(os.path.join(BASE_DIR, "df_tmrain_tmwater_days.csv"))
def get_rainfall_by_location(location, upto_date=None):
  if upto_date:
    return df_tmrain_tmwater_days[["Date", "Location", "Rainfall"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= upto_date]
  return df_tmrain_tmwater_days[["Date", "Location", "Rainfall"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")

def get_supported_locations():
    return df_tmrain_tmwater_days["Location"].unique()

# load transfomer
transformer = Transformer()
banjir_mapping_location = transformer.load()

# loads all models to memory for fast response
## classification model
models = {}
mknn_banjir = pickle.load(open(os.path.join(BASE_DIR, "models/mknn-predict-banjir.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
models["mknn-banjir"] = mknn_banjir

## regression models
locations = json.load(open(banjir_mapping_location, "r"))["Location"].keys()
for location in locations:
  m = pickle.load(open(os.path.join(BASE_DIR, "models/mreg-predict-banjir-%s.pkl"%location), "rb"))
  models[location] = m

window = 10
def predict_banjir(location: str, future_days=1, after_date=None):
  # get future forecast
  if after_date:
    data = get_rainfall_by_location(location, upto_date=after_date)
  else:  
    data = get_rainfall_by_location(location)
  
  # Normalize Data
  data["Rainfall"] = scaler.transform(data["Rainfall"].values.reshape(-1,1)).ravel()
  
  time = data[['Date']]
  data = data[['Rainfall']]
  dataset = data.values # convert the data frame to numpy array
  len_dataset = len(dataset)
  history_data = dataset[len_dataset-2*window:, :] # get last 20 days data

  # predict rainfall next future_days
  rmses = []
  for i in range(future_days):
    # predict rainfall next day
    
    x_test = []
    y_test = history_data

    for i in range(window, len(history_data)):
        x_test.append(history_data[i-window:i, 0])

    # convert the data  to numpy array
    x_test = np.array(x_test)

    forecast_set = models[location].predict(x_test)

    # calculate error
    rmse = np.sqrt(np.mean(forecast_set - y_test)**2)
    rmses.append(rmse)

    # new history based on prediction
    history_data = np.concatenate((history_data, [[forecast_set[-1]]]))
    history_data = np.delete(history_data, 0, axis=0)
  
  # predict banjir or no using knn model
  input_value = []
  for i in range(future_days):
    input_value.append([location, forecast_set[i]])
  # if ENABLE_OUTPUT:  print("input_value (scaled)=",input_value)
  df_banjir_to_predict = pd.DataFrame(input_value, columns=["Location", "Rainfall"])
  df_banjir_to_predict = transformer.transform(df_banjir_to_predict)
  predictions = models['mknn-banjir'].predict(df_banjir_to_predict)

  # prepare output prediction
  # 95% Confidence Interval (CI) example. CI = y_hat +/- 1.96*rmse

  # Inverse transform Rainfall
  forecast_set = scaler.inverse_transform(forecast_set.reshape(-1,1)).ravel()

  output = []
  pred_time = pd.to_datetime(time.values[-1][0])
  for i in range(future_days):
    pred_time =  pred_time + timedelta(days=1)
    output.append([predictions[i], rmses[i], forecast_set[i], pred_time])
  return output, ("prediction", "rmse", "forecasted_rainfall", "date")

# output, col_name = predict_banjir("Sumur Batu", future_days=3)