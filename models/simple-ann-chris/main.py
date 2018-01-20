import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics,activations
from sklearn import metrics
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True

# Reads data from input file
data_raw = pd.read_csv(os.path.join('..', '..', 'data', 'data-0.2.csv'), sep =';', low_memory = False)
data = data_raw.dropna()

# Converts dataframes to numpy arrays
all_features_and_target_value = data.values.astype("float32")

# Trekker ut alle features (5 aromepunkter med 4 features + delayed = 21 features)
x = all_features_and_target_value[:,0:4]

# Trekker ut produksjonen som y-verdi
y = all_features_and_target_value[:,4]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 67)

# Scale the features
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

# In[14]:
dnn_keras_model = models.Sequential()

# Input layer
dnn_keras_model.add(layers.Dense(units=3, input_dim= 4, activation='relu'))


dnn_keras_model.add(layers.Dense(units=10,activation='relu'))
dnn_keras_model.add(layers.Dense(units=10,activation='relu'))
dnn_keras_model.add(layers.Dense(units=10,activation='relu'))
#dnn_keras_model.add(layers.Dense(units=8,activation='relu'))
#dnn_keras_model.add(layers.Dense(units=8,activation='relu'))

# Output Layer
dnn_keras_model.add(layers.Dense(1))

#Kompilere - Alternative lossfunctions: mean_squared_error
dnn_keras_model.compile(optimizer='adam', loss = 'mean_absolute_error')

# Trene modellen
np.random.seed(7)
dnn_keras_model.fit(X_train,y_train, epochs = 5000, batch_size=100, verbose=2)

# Prediksjon på testsett
final_preds = dnn_keras_model.predict(X_test)

# 1000 epocs , b_size = 90, 4(4)[10](1)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_test, final_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_test, final_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_test, final_preds)))


# Prediksjon på treningssett
train_preds = dnn_keras_model.predict(X_train)

# 70% av data (benyttet til treningen)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_train,train_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_train, train_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_train, train_preds)))

# Visualisere resultatene

# Test data
# Bedre figur:
plt.figure(figsize=(20, 12.5))
plt.scatter(y_test,final_preds, s = 20)

plt.xlabel('Reel produksjon Ytre Vika Vindpark')
plt.ylabel('Predikert produksjon')
plt.title('Visualisering av nøyaktigheten av lineær regresjons modell')

# Visualisere begge data

# Bedre figur:
plt.figure(figsize=(20, 12.5))
plt.scatter(y_test,final_preds, s = 20)
plt.scatter(y_train,train_preds, s = 20)

plt.xlabel('Reel produksjon Ytre Vika Vindpark')
plt.ylabel('Predikert produksjon')
plt.title('Visualisering av nøyaktigheten av lineær regresjons modell')

# Ekte plott av data

sept = dnn_keras_model.predict(x[0:200,:])

predictions_sept = list(sept)

predictions_sept_list = []
for pred in predictions_sept:
    predictions_sept_list.append(pred[0])

September = pd.DataFrame(data = {'real': y[0:200], 'predicitions': predictions_sept_list, '2 hours delay': all_features_and_target_value[0:200,3] })

# set time limit
time = 50

ax = September['real'].head(time).plot(figsize=(20,8))
ax = September['predicitions'].head(time).plot(figsize=(20,8))
ax = September['2 hours delay'].head(time).plot(figsize=(20,8))

plt.xlabel('24 ulike timer fra testdataen')
plt.ylabel('Produksjon i Mega Watt (MW)')
plt.title('Prediksjon av produksjonen på Ytre Vika Windpark')

plt.legend(loc='best')

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Saves figure
fig = ax.get_figure()
#fig.savefig('24_punkter.png')