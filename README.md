# FLOOD PREDICTION MACHINE LEARNING
PREDICT is our application solution to predict floods in Jakarta. Predict is a mobile application that uses machine learning algorithms like Neural Netowork
(Binary Classifier) and Linear Regression to predict floods in Jakarta and determine the number of flood-affected areas in Jakarta based on rainfall and water level data. With the predict application, users visualize current and future floods effectively.

### Datasets
* [Waterfall](https://docs.google.com/spreadsheets/d/1xy16th0oBqQ9kux8XGKmkk6AO1flGq1hj1kYqDLn4YI/edit#gid=0)
* [Rainfall](https://docs.google.com/spreadsheets/d/1nI8m27noE1mMiXQXde8jyXD6-qhuMQ2tE-gXxBkuxi4/edit#gid=0)
* [Raw Data](https://drive.google.com/drive/folders/11ZXAKdb8YyLUKXAP8goONpyRkjyAXMpE?usp=sharing)

### Data Preprocessing
* LinearRegression via Scikit-learn
* Neural Netowork (Binary Classifier) via Keras

The model with Keras is crazy big, 151 mb each model. That size times how much locations and preloaded that into memory, surely gonna blow the server. Because of that we decided to use the classification determining banjir or not banjir using tensorflow, whereas to predict regression of rain using simple linear regression.


