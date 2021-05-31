# -*- coding: utf-8 -*-
"""Predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tdIXKWMAAvZfMaKewd3JX_Ho_863moJA

#Flood Prediction

Import Required Libraries
"""

import numpy as np
import pandas as pd

ENABLE_OUTPUT = True

"""## Telemetri Data"""

def gsheet_to_csv(url: str):
  sheet_url = url
  csv = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
  return csv

df_telemetri_rainfall = pd.read_csv(gsheet_to_csv("https://docs.google.com/spreadsheets/d/1nI8m27noE1mMiXQXde8jyXD6-qhuMQ2tE-gXxBkuxi4/edit#gid=0"))
df_telemetri_rainfall

df_telemetri_waterlevel = pd.read_csv(gsheet_to_csv("https://docs.google.com/spreadsheets/d/1xy16th0oBqQ9kux8XGKmkk6AO1flGq1hj1kYqDLn4YI/edit#gid=0"))
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
engine = create_engine('sqlite://', echo=False)

# Save to sql
df_telemetri_waterlevel.to_sql("telemetri_waterlevel", if_exists="replace", con=engine)
df_telemetri_rainfall.to_sql("telemetri_rainfall", if_exists="replace", con=engine)

# check length
if ENABLE_OUTPUT:
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
if ENABLE_OUTPUT: df_tmrain_tmwater

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
if ENABLE_OUTPUT: df_tmrain_tmwater['RainfallStatus'].unique()

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

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]

if ENABLE_OUTPUT: ds_tmrain_tmwater_waterlevel.plot()

water_level_value_replacement = ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel != -1].std()
water_level_value_replacement

# There is high spikes (probably measurement error)
max_idx = ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel == ds_tmrain_tmwater_waterlevel.max()].index[0]
ds_tmrain_tmwater_waterlevel[max_idx] = water_level_value_replacement
if ENABLE_OUTPUT: ds_tmrain_tmwater_waterlevel.plot()

# Let's also replace missing value (-1) there
ds_tmrain_tmwater_waterlevel[ds_tmrain_tmwater_waterlevel == -1] = water_level_value_replacement
if ENABLE_OUTPUT: ds_tmrain_tmwater_waterlevel.plot()

if ENABLE_OUTPUT: ds_tmrain_tmwater_rainfall.plot()

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
if ENABLE_OUTPUT: df_tmrain_tmwater

# See if time is regularly updated

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
if ENABLE_OUTPUT: sb.heatmap(df_tmrain_tmwater_days[["Rainfall", "WaterLevel", "Banjir"]].corr(), cmap="Blues", annot=True)

if ENABLE_OUTPUT: 
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

### Classification Based On Place And Rainfall Size (KNN)
"""

import pickle
import json

class Transformer:
  """Convert non numeric categorical data to numeric value
  """
  def __init__(self):
    self.mapping_keys = {}
    self.mapping_location = "transformer-map.json"

  def load(self):
    self.mapping_keys = json.load(open(self.mapping_location, "r"))
    return self.mapping_location

  def fit_transform(self, df):
    columns = df.columns.values
    self.mapping_keys = {}

    for column in columns:
      text_digit_vals = {}
      if df[column].dtype != np.int64 and df[column].dtype != np.float64 and not np.issubdtype(df[column].dtype, np.datetime64):
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

import sklearn
from sklearn import preprocessing
# Convert categorical data to mapping numbers
transformer = Transformer()
df_tmrain_tmwater_days = transformer.fit_transform(df_tmrain_tmwater_days)
# save to csv
df_tmrain_tmwater_days.to_csv("df_tmrain_tmwater_days.csv", index=False)
# Normalization of Rainfall values
scaler = sklearn.preprocessing.StandardScaler()
df_tmrain_tmwater_days["Rainfall"] = scaler.fit_transform(df_tmrain_tmwater_days["Rainfall"].values.reshape(-1,1)).ravel()
pickle.dump(scaler, open("scaler.pkl", "wb"))
# Save banjir mapping
banjir_mapping_location = transformer.save_mapping()

# Training
import numpy as np
from sklearn import model_selection, neighbors # cross_validation has been deprecated so use model_selection instead
import pandas as pd
import os
if not os.path.exists('models'): os.makedirs('models') # model saving location

X = np.array(df_tmrain_tmwater_days[["Location", "Rainfall"]])
y = np.array(df_tmrain_tmwater_days['Banjir'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

# save classifier model
pickle.dump(clf, open("models/mknn-predict-banjir.pkl", "wb"))

accuracy = clf.score(X_test, y_test)
if ENABLE_OUTPUT: print('Accuracy=', accuracy)

# Predicting

def predict_banjir(input_value):
  df_banjir_to_predict = pd.DataFrame(input_value, columns=["Location", "Rainfall"])
  df_banjir_to_predict = transformer.transform(df_banjir_to_predict)

  # Normalize Rainfall
  df_banjir_to_predict["Rainfall"] = scaler.transform(df_banjir_to_predict["Rainfall"].values.reshape(-1, 1)).ravel()

  if ENABLE_OUTPUT: print('Input=',input_value)
  predictions = clf.predict(df_banjir_to_predict.values)
  if ENABLE_OUTPUT: print('Predictions', predictions)
  return predictions

predict_banjir([["Cawang", 213.5],["Kampung Kelapa", 0],["Cawang", 213.5]])

"""### Rainfall Forecasting Model (Linear Regression)"""

def get_rainfall_by_location(location, upto_date=None):
  if upto_date:
    return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= upto_date]
  return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")

from sklearn.linear_model import LinearRegression
# Generate all regression model for each location
locations = json.load(open(banjir_mapping_location, "r"))["Location"].values()
window = 10 # can predict 10 days ahead

for location in locations:
  print(location)
  df = get_rainfall_by_location(location)

  # Plot data
  # plotting the data
  if ENABLE_OUTPUT: 
    plt.figure(figsize=(16, 8))
    plt.title('Historical Rainfall data from %s'%location)
    plt.plot(df["Date"],df['Rainfall'], color='red')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Rainfall (mm)', fontsize=18)
    plt.show()

  # create new data frame with only 'Rainfall' column
  data = df[['Rainfall']]
  dataset = data.values # convert the data frame to numpy array
  trainning_data_len = np.math.ceil(len(dataset)*.8) # number of rows to train model

  # Create feature using sliding window
  train_data = dataset[0:trainning_data_len-window, :]

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
  if ENABLE_OUTPUT: 
    plt.figure(figsize=(16,8))
    plt.title('Model %s, RMSE(%s)'%(location, rmse))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Rainfall', fontsize=18)
    # plt.plot(train['Rainfall'], linewidth=3.5)
    plt.plot(df[trainning_data_len:]["Date"], valid[['Rainfall', 'Predictions']],  linewidth=3.5)
    plt.legend(['Valid','Predictions'], loc='upper center')
    plt.show()

  pickle.dump(clf, open("models/mreg-predict-banjir-%s.pkl"%location, "wb"))

"""### Classification Based On Place And Rainfall Size (Tensorflow)"""

def train_valid_test_split(data, ratio):
  """Split data based on the ratio
  Args:
    data (dataframe): ratio
    ratio (tuple): train:validation:test; example: (0.8, 0.1, 0.1)
  Returns:
    train_data, validation_data, test_data
  """
  ratio_train, ratio_valid, ratio_test = ratio
  assert ratio_valid == ratio_test
  assert np.sum(ratio) == 1
  training_data_len = np.math.ceil(len(data)*ratio_train)
  valid_data_len = np.math.ceil(len(data)*ratio_valid)
  train_data = data.values[0:training_data_len]
  valid_data = data.values[training_data_len:training_data_len+valid_data_len]
  test_data = data.values[training_data_len+valid_data_len:]
  return train_data, valid_data, test_data

def generate_forecast_feature(data, window=10):
    """Generate Windowing feature for forecasting
    Args:
      data (np.array): data to forecast
      window (int): window size; default 10
    Returns:
      X (np.array): features
      y (np.array): labels
    """
    X = []
    y = []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    #### convert to numpy array
    X, y = np.array(X), np.array(y)
    #### reshape to 3d (expected by keras)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

locations = json.load(open(banjir_mapping_location, "r"))["Location"].values()
transformer = Transformer()

## Need to get 80:10:10 for all locations (same)
train_data_all = []
valid_data_all = []
test_data_all = []

for location in locations:
  df = get_rainfall_by_location(location)
  # split data
  train_data, valid_data, test_data = train_valid_test_split(df[["Location", "Rainfall", "Banjir"]], (0.8, 0.1, 0.1))

  train_data_all.append(train_data)
  valid_data_all.append(valid_data)
  test_data_all.append(test_data)

train_data = np.vstack(train_data_all)
valid_data = np.vstack(valid_data_all)
test_data = np.vstack(test_data_all)

## build train, valid and test data
def reshape_to_3d(X):
  """Reshape 2d np array X to 3d np array X
  Args:
    X (np.array): 2d np array
  Returns:
    X (np.array): 3d np array
  """
  X = np.reshape(X, (X.shape[0], X.shape[1], 1))
  return X
  
x_train, y_train = train_data[:, :2], train_data[:, 2]
x_train = reshape_to_3d(x_train)

x_valid, y_valid = valid_data[:, :2], valid_data[:, 2]
x_valid = reshape_to_3d(x_valid)

x_test, y_test = test_data[:, :2], test_data[:, 2]
x_test = reshape_to_3d(x_test)

import os
if not os.path.exists('models'): os.makedirs('models/arch') # model saving location

# Callback model
class Callback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
        self.history = {'loss':[], 'val_loss':[], 'avg_pass_tresshold': 5}
  def on_epoch_end(self, epoch, logs={}):
    # Stop to prevent overfit but not too early
    if epoch > 1:
        val_avg = np.average(self.history["val_loss"])
        if logs.get('val_loss') > val_avg and self.history["avg_pass_tresshold"] <= 0:
          print("Validation loss not improving, stopping!")
          self.model.stop_training = True
        elif logs.get('val_loss') < val_avg:
          self.history["avg_pass_tresshold"] += 1
        else:
          self.history["avg_pass_tresshold"] -= 1
          
    if logs.get("val_loss") >= logs.get("loss") and epoch >= 10 and self.history["avg_pass_tresshold"] <= 0:
      print("\nval_loss >= loss")
      self.model.stop_training = True
early_stop_callback = Callback()

# Create Model
model = Sequential()
model.add(Dense(512, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=["accuracy"])

# Train the model
logs = model.fit(
    x_train, 
    y_train,
    epochs=1000,
    callbacks=[early_stop_callback],
    validation_data=(x_valid, y_valid),
)

# Show Model Performance
plt.plot(logs.history["loss"])
plt.plot(logs.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper right")
plt.show()

plt.plot(logs.history["accuracy"])
plt.plot(logs.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="lower right")
plt.show()


# Try to predict
y_predict = model.predict(x_test)

# Accuracy (RMSE)
rmse = np.sqrt(np.mean(y_predict - y_test)**2)
print("RMSE", rmse)
model.save("models/kerasbinary-predict-banjir.h5")

"""### Rainfall Forecasting Model (Tensorflow)"""

# Generate all lstm model for each location
locations = json.load(open(banjir_mapping_location, "r"))["Location"].values()
window = 10 # can predict 10 days ahead

# Callback model
class Callback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
        self.history = {'loss':[], 'val_loss':[], 'avg_pass_tresshold': 5}
  def on_epoch_end(self, epoch, logs={}):
    # mse = history.get("mse")
    # if (mse<0.05):
    #   self.model.stop_training = True
    # Stop to prevent overfit but not too early
    if epoch > 1:
        val_avg = np.average(self.history["val_loss"])
        if logs.get('val_loss') > val_avg and self.history["avg_pass_tresshold"] <= 0:
          print("Validation loss not improving, stopping!")
          self.model.stop_training = True
        elif logs.get('val_loss') < val_avg:
          self.history["avg_pass_tresshold"] += 1
        else:
          self.history["avg_pass_tresshold"] -= 1
          
    if logs.get("val_loss") >= logs.get("loss") and epoch >= 10 and self.history["avg_pass_tresshold"] <= 0:
      print("\nval_loss >= loss")
      self.model.stop_training = True
    if epoch >= 10:
      self.model.stop_training = True

early_stop_callback = Callback()

for location in locations:
  df = get_rainfall_by_location(location)

  # create new data frame with only 'Rainfall' column
  data = df[['Rainfall']]
  if len(data) <= 31: continue
  
  # split data
  train_data, valid_data, test_data = train_valid_test_split(data, (0.8, 0.1, 0.1))
  train_data_date, valid_data_date, test_data_date = train_valid_test_split(df[["Date"]], (0.8, 0.1, 0.1)) # For plotting

  # Create feature using sliding window 
  x_train, y_train = generate_forecast_feature(train_data)
  print(x_train.shape, y_train.shape)
  x_valid, y_valid = generate_forecast_feature(valid_data)
  print(x_valid.shape, y_valid.shape)
  x_test, y_test = generate_forecast_feature(test_data)
  print(x_test.shape, y_test.shape)

  # Build LSTM model
  model = Sequential()
  model.add(LSTM(1024, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(LSTM(1024, return_sequences=False))
  model.add(Dense(512, activation="relu"))
  model.add(Dropout(0.1))
  model.add(Dense(256, activation="relu"))
  model.add(Dense(1))

  # Compile the model
  # model.compile(optimizer='adam', loss='mean_squared_error')
  # Compile the model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
                optimizer=optimizer,
                metrics=["mse"])

  # Train the model
  logs = model.fit(
      x_train, 
      y_train,
      verbose=False,
      epochs=1000, 
      callbacks=[early_stop_callback], 
      validation_data=(x_valid, y_valid),
  )

  # Show Model Performance
  plt.plot(logs.history["loss"])
  plt.plot(logs.history["val_loss"])
  plt.title("Model Loss of %s"%location)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["Train", "Validation"], loc="upper right")
  plt.show()

  plt.plot(logs.history["mse"])
  plt.title("Mean Squared Error")
  plt.xlabel("Epochs")
  plt.ylabel("MSE")
  plt.legend(["Train", "Validation"], loc="lower right")
  plt.show()

  # Try to predict
  forecast_set = model.predict(x_test)

  # Accuracy (RMSE)
  rmse = np.sqrt(np.mean(forecast_set - y_test)**2)

  # visualization of the data
  if ENABLE_OUTPUT: 
    plt.figure(figsize=(16,8))
    plt.title('Train Validation Split %s'%(location))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Rainfall', fontsize=18)
    plt.plot(train_data_date.ravel()[window:], y_train, linewidth=3.5)
    plt.plot(valid_data_date.ravel()[window:], y_valid, linewidth=3.5)
    plt.plot(test_data_date.ravel()[window:], pd.DataFrame({"test":y_test, "predi": forecast_set.ravel()}),  linewidth=3.5)
    plt.legend(['Train', 'Validation', 'Test','Predictions'], loc='upper center')
    plt.show()
    ## Zoom
    plt.figure(figsize=(16,8))
    plt.title('Model %s, RMSE(%s)'%(location, rmse))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Rainfall', fontsize=18)
    plt.plot(test_data_date.ravel()[window:], pd.DataFrame({"test":y_test, "predi": forecast_set.ravel()}),  linewidth=3.5)
    plt.legend(['Test','Predictions'], loc='upper center')
    plt.show()


  model.save("models/lstm-predict-banjir-%s.h5"%location)

"""# Usage In Cloud

### Sklearn
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import timedelta
import os

BASE_DIR = "."

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
    return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= upto_date]
  return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")

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
locations = json.load(open(banjir_mapping_location, "r"))["Location"].values()
for location in locations:
  m = pickle.load(open(os.path.join(BASE_DIR, "models/mreg-predict-banjir-%s.pkl"%location), "rb"))
  models[location] = m

window = 10
location_transform = json.load(open(banjir_mapping_location, "r"))["Location"]
def predict_banjir(location: str, future_days=1, after_date=None):
  # transform location
  location = location_transform[location]
  # get future forecast
  if after_date:
    data = get_rainfall_by_location(location, upto_date=after_date)
  else:  
    data = get_rainfall_by_location(location)
  
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

output, col_name = predict_banjir("Sumur Batu", future_days=3)
pd.DataFrame(output, columns=col_name)

# Kampung Kelapa before 2019-12-24

df_tmrain_tmwater_days[df_tmrain_tmwater_days["Location"] == location_transform["Kampung Kelapa"]].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= "2019-01-01"].tail(10)

from datetime import datetime
output, out_name = predict_banjir("Kampung Kelapa", future_days=10, after_date="2018-11-16")
pd.DataFrame(output, columns=out_name)

"""### Tensorflow"""

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import pickle
import json
from datetime import timedelta
import os

BASE_DIR = "."

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
    return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= upto_date]
  return df_tmrain_tmwater_days[["Date", "Location", "Rainfall", "Banjir"]][df_tmrain_tmwater_days["Location"] == location].sort_values("Date")

# load transfomer
transformer = Transformer()
banjir_mapping_location = transformer.load()

# loads all models to memory for fast response
## classification model
models = {}
kerasbinary_banjir = keras.models.load_model("models/kerasbinary-predict-banjir.h5")
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
models["kerasbinary-banjir"] = kerasbinary_banjir

## regression models
locations = json.load(open(banjir_mapping_location, "r"))["Location"].values()
for location in locations:
  try:
    m = keras.models.load_model("models/lstm-predict-banjir-%s.h5"%location)
    models[location] = m
  except Exception as err:
    print(err)

window = 10
location_transform = json.load(open(banjir_mapping_location, "r"))["Location"]
def reshape_to_3d(X):
  """Reshape 2d np array X to 3d np array X
  Args:
    X (np.array): 2d np array
  Returns:
    X (np.array): 3d np array
  """
  X = np.reshape(X, (X.shape[0], X.shape[1], 1))
  return X

def predict_banjir(location: str, future_days=1, after_date=None):
  # transform location
  location = location_transform[location]
  # get future forecast
  if after_date:
    data = get_rainfall_by_location(location, upto_date=after_date)
  else:  
    data = get_rainfall_by_location(location)
  
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
    x_test = reshape_to_3d(x_test)
    forecast_set = models[location].predict(x_test)
    forecast_set = forecast_set.ravel()

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
  df_banjir_to_predict = df_banjir_to_predict.values.reshape(-1,1)
  df_banjir_to_predict = reshape_to_3d(df_banjir_to_predict)
  predictions = models['kerasbinary-banjir'].predict(df_banjir_to_predict)
  predictions = predictions.ravel()

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

df_tmrain_tmwater_days[df_tmrain_tmwater_days["Location"] == location_transform["Kampung Kelapa"]].sort_values("Date")[df_tmrain_tmwater_days["Date"] <= "2020-12-14"].tail(10)

from datetime import datetime
output, out_name = predict_banjir("Kampung Kelapa", future_days=10, after_date="2020-12-14")
pd.DataFrame(output, columns=out_name)

"""_The model with keras is crazy big. 151 mb each model._

_That size times how much locations and preloaded that into memory, surely gonna blow the server_
"""