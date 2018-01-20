import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow.contrib.keras import losses, optimizers, metrics, activations, models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['patch.force_edgecolor'] = True

data_raw = pd.read_csv('data-2.3.csv',sep =';', low_memory = False)
data = data_raw.dropna()

# Konverterer pandas.DataFrame til en numpy array
all_features_and_target_value = data.values.astype("float32")

# Trekker ut alle features (5 aromepunkter med 4 features + delayed = 21 features)
x = all_features_and_target_value[:,0:24]

# Trekker ut produksjonen som y-verdi
y = all_features_and_target_value[:,24]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 67)


# Scale dataset
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

dnn_keras_model = models.Sequential()

# Input layer
dnn_keras_model.add(layers.Dense(units=12, input_dim= 24, activation='relu'))

dnn_keras_model.add(layers.Dense(units=12,activation='relu'))
dnn_keras_model.add(layers.Dense(units=12,activation='relu'))
dnn_keras_model.add(layers.Dense(units=12,activation='relu'))

# Output Layer
dnn_keras_model.add(layers.Dense(1))

#Kompilere - Alternative lossfunctions: mean_squared_error
dnn_keras_model.compile(optimizer='adam', loss = 'mean_absolute_error')

# Trene modellen
np.random.seed(42)
dnn_keras_model.fit(X_train,y_train, epochs = 30000, batch_size=100, verbose=2, validation_data=(X_test, y_test))

from sklearn import metrics

final_preds = dnn_keras_model.predict(X_test)

# 5000 epochs , b_size = 10, 24(24)[10](1)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_test, final_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_test, final_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_test, final_preds)))

# Prediksjon på treningdata
train_preds = dnn_keras_model.predict(X_train)

# 70% av data (benyttet til treningen)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_train,train_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_train, train_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_train, train_preds)))


# Visualisere resultater
plt.figure(figsize=(20, 12.5))
plt.scatter(y_test,final_preds, s = 20)

plt.xlabel('Reel produksjon Ytre Vika Vindpark')
plt.ylabel('Predikert produksjon')
plt.title('Visualisering av nøyaktigheten av lineær regresjons modell')

# Bedre figur:
plt.figure(figsize=(20, 12.5))
plt.scatter(y_test,final_preds, s = 20)
plt.scatter(y_train,train_preds, s = 20)

plt.xlabel('Reel produksjon Ytre Vika Vindpark')
plt.ylabel('Predikert produksjon')
plt.title('Visualisering av nøyaktigheten av lineær regresjons modell')

# # Ytterligere undersøkelser av resultatene
predictions = list(final_preds)

predictions_list = []
for pred in predictions:
    predictions_list.append(pred[0])

Oversikt = pd.DataFrame(data = {'real': y_test, 'predicitions': predictions_list})

Oversikt.head(10)

Oversikt['differanse'] = Oversikt['real'] - Oversikt['predicitions']  

Oversikt['abs_diff'] = Oversikt['differanse'].apply(abs)

Oversikt.head()

Oversikt['abs_diff'].hist(bins = 30)

ax = Oversikt.plot.scatter(x='real',y='predicitions',
                   c='abs_diff',cmap='coolwarm', figsize = (20,10))

ax.set_xlabel("x label")

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = ax.get_figure()
#fig.savefig('keras_model.png')

ax = Oversikt['real'].head(24).plot(figsize=(20,8))
ax = Oversikt['predicitions'].head(24).plot(figsize=(20,8))

plt.xlabel('24 ulike timer fra testdataen')
plt.ylabel('Produksjon i Mega Watt (MW)')
plt.title('Prediksjon av produksjonen på Ytre Vika Windpark')

plt.legend(loc='best')

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Saves figure
fig = ax.get_figure()

predictions_train = list(train_preds)

predictions_train_list = []
for pred in predictions_train:
    predictions_train_list.append(pred[0])

Oversikt_trening = pd.DataFrame(data = {'real': y_train, 'predicitions': predictions_train_list})

Oversikt_trening['differanse'] = Oversikt_trening['real'] - Oversikt_trening['predicitions']  
Oversikt_trening['abs_diff'] = Oversikt_trening['differanse'].apply(abs)

ax = Oversikt_trening.plot.scatter(x='real',y='predicitions',
                   c='abs_diff',cmap='coolwarm', figsize = (20,10))

ax.set_xlabel("x label")

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = ax.get_figure()
#fig.savefig('keras_model.png')

Oversikt_komplett = pd.concat([Oversikt,Oversikt_trening])

Oversikt_komplett.info()

ax = Oversikt_komplett.plot.scatter(x='real',y='predicitions',
                   c='abs_diff',cmap='coolwarm', figsize = (20,10))

ax.set_xlabel("x label")

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = ax.get_figure()
fig.savefig('keras_model_komplett.png')

sept = dnn_keras_model.predict(x[0:500,:])

predictions_sept = list(sept)

predictions_sept_list = []
for pred in predictions_sept:
    predictions_sept_list.append(pred[0])

September = pd.DataFrame(data = {'real': y[0:500], 'predicitions': predictions_sept_list})

data.reset_index(inplace=True)

# Justere tidslinjen på plottet
time = 500

ax = September['real'].head(time).plot(figsize=(20,8))
ax = September['predicitions'].head(time).plot(figsize=(20,8))
ax = data['YVIK-YtreVikna1-Sum-produksjon (2 hours before)'].head(time).plot(figsize=(20,8))

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
