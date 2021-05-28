# -*- coding: utf-8 -*-
"""Predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LQiLIdDtBfumeOb-JZT4P_Xu97gd1DVD

#Flood Prediction

Import Required Libraries
"""

import numpy as np
import pandas as pd

"""## Telemetri Data"""

def gsheet_to_csv(url: str):
  sheet_url = url
  csv = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
  return csv

df_telemetri_rainfall = pd.read_csv("TelemetriRainfall.csv")
df_telemetri_rainfall

df_telemetri_waterlevel = pd.read_csv("TelemetriWaterLevel.csv")
df_telemetri_waterlevel.rename(columns = {'Water Level (cm)': 'water_level', 'Date ':'Date'}, inplace = True)
df_telemetri_waterlevel

# Pandas Left Join is Out of Memory
# pd.merge(df_telemetri_rainfall, df_telemetri_waterlevel, how="left", on="Location")
# Let's try to use sql

"""**Pandas Left Join is Out of Memory**
```python
pd.merge(df_telemetri_rainfall, df_telemetri_waterlevel, how="left", on="Location")
```
Let's try to use sql instead
"""

from sqlalchemy import create_engine
engine = create_engine('sqlite:///./db.sqlite3', echo=False)

# Save to sql
df_telemetri_waterlevel.to_sql("telemetri_waterlevel", if_exists="replace", con=engine)
df_telemetri_rainfall.to_sql("telemetri_rainfall", if_exists="replace", con=engine)

# check length
print(len(df_telemetri_rainfall))
print(len(df_telemetri_waterlevel))

"""Join operations ini untuk menggabungkan data dengan cara mengambil semua data yang memiliki nilai **water_level** dan **Rainfall** (jika ada label data banjir dari PUPR)
```
SELECT 
  tr.Date AS Date,
  tr.Time AS Time,
  tr.Location AS Location,
  tr.Rainfall AS Rainfall,
  tr.Status AS RainfallStatus,
  tw.water_level AS WaterLevel
FROM telemetri_rainfall AS tr
LEFT JOIN telemetri_waterlevel AS tw
ON tr.Location = tw.Location AND tr.Date = tw.Date AND tr.Time = tw.Time

UNION

SELECT 
  tw.Date AS Date,
  tw.Time AS Time,
  tw.Location AS Location,
  tr.Rainfall AS Rainfall,
  tr.Status AS RainfallStatus,
  tw.water_level AS WaterLevel
FROM telemetri_waterlevel AS tw
LEFT JOIN telemetri_rainfall AS tr
ON tr.Location = tw.Location AND tr.Date = tw.Date AND tr.Time = tw.Time
```
Namun karena menggunakan asumsi bahwa jika **RainfallStatus** == lebat  sebagai label banjir maka digunakan join ini
```
SELECT 
  tr.Date AS Date,
  tr.Time AS Time,
  tr.Location AS Location,
  tr.Rainfall AS Rainfall,
  tr.Status AS RainfallStatus,
  tw.water_level AS WaterLevel
FROM telemetri_rainfall AS tr
LEFT JOIN telemetri_waterlevel AS tw
ON tr.Location = tw.Location AND tr.Date = tw.Date AND tr.Time = tw.Time
```
"""

query = engine.execute('''
SELECT 
  tr.Date AS Date,
  tr.Time AS Time,
  tr.Location AS Location,
  tr.Rainfall AS Rainfall,
  tr.Status AS RainfallStatus,
  tw.water_level AS WaterLevel
FROM telemetri_rainfall AS tr
LEFT JOIN telemetri_waterlevel AS tw
ON tr.Location = tw.Location AND tr.Date = tw.Date AND tr.Time = tw.Time
''')
df_tmrain_tmwater = pd.DataFrame(query.fetchall())
df_tmrain_tmwater.columns = query.keys()
df_tmrain_tmwater

df_tmrain_tmwater.describe()

"""## Preprocessing Data

Number of Missing Value
"""

df_tmrain_tmwater.isnull().sum()

df_tmrain_tmwater['RainfallStatus'].unique()

# Clean Column RainfallStatus
df_tmrain_tmwater.loc[df_tmrain_tmwater['RainfallStatus']=='Cerah\\', 'RainfallStatus'] = "Cerah"
df_tmrain_tmwater.loc[df_tmrain_tmwater['RainfallStatus']=='Light', 'RainfallStatus'] = "Ringan"
df_tmrain_tmwater.loc[df_tmrain_tmwater['RainfallStatus']=='Is', 'RainfallStatus'] = "Sedang"
df_tmrain_tmwater.loc[df_tmrain_tmwater['RainfallStatus']=='Bright', 'RainfallStatus'] = "Cerah"
df_tmrain_tmwater = df_tmrain_tmwater[df_tmrain_tmwater['RainfallStatus'].notna()]
df_tmrain_tmwater['RainfallStatus'].unique()

"""### Data metrics defenition
* Rainfall (mm)

  1mm rainfall means every one square meter area is filled with the water of height 1mm.
  1 square meter = 1000mm length ×1000mm breath
  So 1mm rain means 1000mm length × 1000mm breath × 1mm height = 1litre of water.
  Every square meter has one litre of water
"""

df_tmrain_tmwater[df_tmrain_tmwater['WaterLevel'].notna()][df_tmrain_tmwater['RainfallStatus'] == "Lebat"]

# Convert metrics to float

from re import compile as re_compile, sub as re_sub

re_rainfall_metric = re_compile(r'[^\d.]')
re_waterlevel_metric = re_compile(r'[^\d,]')

def rainfall_metric_to_float(metric):
  try:
    return float(re_rainfall_metric.sub('', metric))
  except Exception as err:
    return -1.0
def waterlevel_metric_to_float(metric):
  try:
    return float(re_sub(r',', '', re_waterlevel_metric.sub('', metric)))
  except Exception as err:
    return -1.0
ds_tmrain_tmwater_rainfall = df_tmrain_tmwater.apply(lambda row: rainfall_metric_to_float(str(row["Rainfall"])), axis=1)
ds_tmrain_tmwater_waterlevel = df_tmrain_tmwater.apply(lambda row: waterlevel_metric_to_float(str(row["WaterLevel"])), axis=1)

ds_tmrain_tmwater_waterlevel.plot()

water_level_value_replacement = ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel != -1].std()
water_level_value_replacement

# There is high spikes (probably measurement error)
max_idx = ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel == ds_tmrain_tmwater_waterlevel.max()].index[0]
ds_tmrain_tmwater_waterlevel[max_idx] = water_level_value_replacement
# ds_tmrain_tmwater_waterlevel.plot()

# Let's also replace missing value (-1) there
ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel == -1] = water_level_value_replacement
# ds_tmrain_tmwater_waterlevel.plot()

# ds_tmrain_tmwater_rainfall.plot()

# Looks good, let's join to main df
df_tmrain_tmwater["WaterLevel"] = ds_tmrain_tmwater_waterlevel
df_tmrain_tmwater["Rainfall"] = ds_tmrain_tmwater_rainfall
df_tmrain_tmwater

from datetime import datetime
def convert_to_datetime(date: str):
  try:
    return datetime.strptime(date, '%d-%b-%y')
  except ValueError:
    return datetime.strptime(date, '%d-%B-%y')

df_tmrain_tmwater['Date'] = df_tmrain_tmwater.apply(lambda row: convert_to_datetime(row["Date"]), axis=1)
df_tmrain_tmwater

# See if time is regularly updated

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]

df_tmrain_tmwater[df_tmrain_tmwater["Location"] == "Sumur Batu"].plot("Date", "Time", kind="scatter", s=1)

"""There are many missing values! (Not regular)

Let's just take by date and not by time
"""

df_tmrain_tmwater_days = df_tmrain_tmwater.sort_values("Rainfall", ascending=False).groupby(["Location", "Date"]).head(1)
df_tmrain_tmwater_days

# Asumsi Labeling
df_tmrain_tmwater_days['Banjir'] = df_tmrain_tmwater_days.apply(lambda row: 1 if str(row['RainfallStatus']).lower() == "lebat" or str(row['RainfallStatus']).lower() == "sangat lebat" else 0, axis=1)
df_tmrain_tmwater_days

# Berapa banjir ya
df_tmrain_tmwater_days["Banjir"].sum()

# Bagaimana korelasi WaterLevel, Rainfall, dan Banjir?
import seaborn as sb
sb.heatmap(df_tmrain_tmwater_days[["Rainfall", "WaterLevel", "Banjir"]].corr(), cmap="Blues", annot=True)

ax1 = df_tmrain_tmwater_days[df_tmrain_tmwater_days['Banjir'] == 1].plot(kind='scatter', x='Rainfall', y='WaterLevel', color='blue', alpha=0.5, figsize=(10, 7), logx=True, logy=True)
df_tmrain_tmwater_days[df_tmrain_tmwater_days['Banjir'] == 0].plot(kind='scatter', x='Rainfall', y='WaterLevel', color='magenta', alpha=0.5, figsize=(10 ,7), ax=ax1, logx=True, logy=True)
plt.legend(labels=['Banjir', 'Tidak Banjir'])
plt.title('Relationship between Rainfall and WaterLevel', size=24)
plt.xlabel('Rainfall (mm)', size=18)
plt.ylabel('Waterlevel (cm)', size=18);

"""Tidak terlalu bagus untuk WaterLevel vs Banjir (Mungkin karena data tidak lengkap)


Karena tidak bagus korelasinya, maka digunakan satu kolom saja yakni 

X[Rainfall]

Y[Banjir]

## Create Model

### Models for detecting banjir(1) or tidak banjir(0)
"""

import pickle
import json

class Transformer:
  """Convert non numeric categorical data to numeric value
  """
  def __init__(self):
    self.mapping_keys = {}

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
  
  def save_mapping(self, location):
    json.dump(self.mapping_keys, open(location, "w"))

# Convert categorical data to mapping numbers

transformer = Transformer()
df_banjir = transformer.fit_transform(df_tmrain_tmwater_days[["Location", "Rainfall", "Banjir"]])

# Save banjir mapping
banjir_mapping_location = "banjir-location-mapping.json" 
transformer.save_mapping(banjir_mapping_location)

# Training
import numpy as np
from sklearn import preprocessing, model_selection, neighbors # cross_validation has been deprecated so use model_selection instead
import pandas as pd
import os
if not os.path.exists('models'): os.makedirs('models') # model saving location



X = np.array(df_banjir.drop(['Banjir'], 1))
y = np.array(df_banjir['Banjir'])

# Normalization of Rainfall values
scaler = preprocessing.StandardScaler()
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1,1)).ravel()
pickle.dump(scaler, open("scaler.pkl", "wb"))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

# save classifier model
pickle.dump(clf, open("models/mknn-predict-banjir.pkl", "wb"))

accuracy = clf.score(X_test, y_test)
print('Accuracy=', accuracy)

# Predicting

def predict_banjir(input_value):
  df_banjir_to_predict = pd.DataFrame(input_value, columns=["Location", "Rainfall"])
  df_banjir_to_predict = transformer.transform(df_banjir_to_predict)

  # Normalize Rainfall
  df_banjir_to_predict["Rainfall"] = scaler.transform(df_banjir_to_predict["Rainfall"].values.reshape(-1, 1)).ravel()

  print('Input=',input_value)
  predictions = clf.predict(df_banjir_to_predict.values)
  print('Predictions', predictions)
  return predictions

predict_banjir([["Cawang", 213.5],["Kampung Kelapa", 0],["Cawang", 213.5]])

"""## Rainfall Forecasting Model"""

from sklearn.linear_model import LinearRegression
def get_rainfall_by_location(location, upto_date=None):
  if upto_date:
    return df_tmrain_tmwater_days[["Date", "Location", "Rainfall"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= upto_date]
  return df_tmrain_tmwater_days[["Date", "Location", "Rainfall"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")

# Generate all regression model for each location
locations = json.load(open(banjir_mapping_location, "r"))["Location"].keys()
window = 10 # can predict 10 days ahead

for location in locations:
  df = get_rainfall_by_location(location)

  # Normalize Rainfall
  df["Rainfall"] = scaler.transform(df["Rainfall"].values.reshape(-1,1)).ravel()

  # Plot data
  # plotting the data
  plt.figure(figsize=(16, 8))
  plt.title('Historical Rainfall data from %s'%location)
  plt.plot(df["Date"],df['Rainfall'], color='red')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Rainfall (mm)', fontsize=18)
  # plt.show()

  # create new data frame with only 'Rainfall' column
  data = df[['Rainfall']]
  dataset = data.values # convert the data frame to numpy array
  trainning_data_len = np.math.ceil(len(dataset)*.8) # number of rows to train model

  # Create feature using sliding window
  train_data = dataset[0:trainning_data_len, :]

  # Split the data into x_train, y_train datasets
  x_train = []
  y_train = []
  for i in range(window, len(train_data)):
      x_train.append(train_data[i-window:i, 0])
      y_train.append(train_data[i, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  clf = LinearRegression(n_jobs=-1, normalize=True)
  clf.fit(x_train, y_train)

  test_data = dataset[trainning_data_len-window:, :]
  # Create the data sets x_test and y_test
  x_test = []
  y_test = dataset[trainning_data_len:, :]
  for i in range(window, len(test_data)):
      x_test.append(test_data[i-window:i, 0])

  # convert the data  to numpy array
  x_test = np.array(x_test)

  forecast_set = clf.predict(x_test)

  # Accuracy (RMSE)
  rmse = np.sqrt(np.mean(forecast_set - y_test)**2)

  # plot the data
  train = data[:trainning_data_len]
  valid = data[trainning_data_len:]
  valid['Predictions'] = forecast_set

  # visualization of the data
  plt.figure(figsize=(16,8))
  plt.title('Model %s, RMSE(%s)'%(location, rmse))
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Rainfall', fontsize=18)
  # plt.plot(train['Rainfall'], linewidth=3.5)
  plt.plot(df[trainning_data_len:]["Date"], valid[['Rainfall', 'Predictions']],  linewidth=3.5)
  plt.legend(['Valid','Predictions'], loc='upper center')
  # plt.show()

  pickle.dump(clf, open("models/mreg-predict-banjir-%s.pkl"%location, "wb"))

"""# Usage In Cloud"""

# Save model
import pickle
a_model = "abcd"
pickle.dump(a_model, open("model.pkl", "wb"))

# Load model
model = pickle.load(open("model.pkl", "rb"))


def an_api():
  # use model
  output = model.predict("something")

  return JsonResponse({"output": output})

from datetime import timedelta
# loads all models to memory for fast response
## classification model
models = {}
mknn_banjir = pickle.load(open("models/mknn-predict-banjir.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
models["mknn-banjir"] = mknn_banjir

## regression models
locations = json.load(open(banjir_mapping_location, "r"))["Location"].keys()
for location in locations:
  m = pickle.load(open("models/mreg-predict-banjir-%s.pkl"%location, "rb"))
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
  print("input_value (scaled)=",input_value)
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

output, col_name = predict_banjir("Sumur Batu", future_days=3)
pd.DataFrame(output, columns=col_name)

# Kampung Kelapa before 2019-12-24

df_tmrain_tmwater_days[df_tmrain_tmwater_days["Location"] == "Kampung Kelapa"].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= "2019-12-24"].tail(10)

output, out_name = predict_banjir("Kampung Kelapa", future_days=10, after_date="2019-12-14")
pd.DataFrame(output, columns=out_name)

