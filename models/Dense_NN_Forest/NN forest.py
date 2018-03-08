

# # Keras - Dense neural network (Advanced)


import os
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Making plots look better (Jupyter Notebook spesific)
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True

# Built in jupyter notebook commands
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# #### Importere datasett

data_raw = pd.read_csv(os.path.join('..','..','data','Ytre Vikna','data_ytrevikna_advanced.csv'),sep =';', low_memory = False)
data_raw.info()


data = data_raw.dropna()
data.info()


num_features = len(data.columns) -1
print(num_features)


# #### Konvertere til numpy-arrays

# Konverterer pandas.DataFrame til en numpy array
all_features_and_target_value = data.values.astype("float32")

# Trekker ut alle features (5 aromepunkter med 4 features + delayed = 21 features)
x = all_features_and_target_value[:,0:num_features]

# Trekker ut produksjonen som y-verdi
y = all_features_and_target_value[:,num_features]


# #### Skalere data
from sklearn.preprocessing import MinMaxScaler

# Lage en scaler
scaler = MinMaxScaler()

# Anvende på features, her lagret i variablen x.
x = scaler.fit_transform(x)


# #### Train/test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 67)

# # Keras
# import tensorflow as tf
# from tensorflow.contrib.keras import models



dropout_rate = 0.25
relu_leak = 0.2



# #### Sette opp layers
# from tensorflow.contrib.keras import layers

# dnn_keras_model = models.Sequential()
# # Input layer
# dnn_keras_model.add(layers.Dense(units=32, input_dim= num_features, activation=None))#, kernel_regularizer=regularizers.l2(0.01)))
# dnn_keras_model.add(layers.BatchNormalization())
# dnn_keras_model.add(layers.Dropout(dropout_rate))
# dnn_keras_model.add(layers.LeakyReLU(alpha=relu_leak))

# #Hidden Layers
# dnn_keras_model.add(layers.Dense(units=16,activation=None))#, kernel_regularizer=regularizers.l2(0.01)))
# dnn_keras_model.add(layers.BatchNormalization())
# dnn_keras_model.add(layers.Dropout(dropout_rate))
# dnn_keras_model.add(layers.LeakyReLU(alpha=relu_leak))

# dnn_keras_model.add(layers.Dense(units=8,activation=None))#, kernel_regularizer=regularizers.l2(0.01)))
# dnn_keras_model.add(layers.BatchNormalization())
# dnn_keras_model.add(layers.LeakyReLU(alpha=relu_leak))

# dnn_keras_model.add(layers.Dense(units=2,activation=None))#, kernel_regularizer=regularizers.l2(0.01)))
# dnn_keras_model.add(layers.BatchNormalization())
# dnn_keras_model.add(layers.LeakyReLU(alpha=relu_leak))

# # Output Layer
# dnn_keras_model.add(layers.Dense(1))

from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Average, Dropout
from keras.regularizers import l2

# a = Input(shape=(num_features,))

# b = Dense(units = 32,activation=None)(a)
# b = BatchNormalization()(b)
# b = LeakyReLU(0.2)(b)
# model = Model(inputs=a, outputs=b)

# exit(0)


#Creating larger model (using keras model API)

def dense_block(input_data, units, dropout = False, l2_reg = 0):

    x = Dense(units = units, activation=None, activity_regularizer = l2(l2_reg))(input_data)
    if dropout: x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    return LeakyReLU(relu_leak)(x)

#Dimension of input layer is the number of features
input_layer = Input(shape=(num_features,))

#Hidden model 1
a = dense_block(input_layer, 32)
a = dense_block(a, 16, True)
a = dense_block(a, 8)
a = dense_block(a, 2)

out_1 = Dense(1)(a)


b = dense_block(input_layer, 128)
b = dense_block(b, 64, True)
b = dense_block(b, 32)
b = dense_block(b, 8)

out_2 = Dense(1)(b)

c = dense_block(input_layer, 64)
c = dense_block(input_layer, 16, True)
c = dense_block(input_layer, 6)

out_3 = Dense(1)(c)

d = dense_block(input_layer, 16, True)
d = dense_block(d, 8)
d = dense_block(d, 8)

out_4 = Dense(1)(d)

out = Average()([out_1, out_2, out_3, out_4])

dnn_keras_model = Model(inputs = input_layer, outputs = out)
dnn_keras_model.summary()



# #### Kompilere modellen
# from tensorflow.contrib.keras import losses,optimizers,metrics,activations

# adam_opt = optimizers.Adam(lr=0.0005, decay=0.001)
#Kompilere - Alternative lossfunctions: mean_squared_error
dnn_keras_model.compile(optimizer='adam', loss = 'mean_absolute_error')


# #### Trene modellen
# from tensorflow.contrib.keras import losses,optimizers,metrics,activations
# #### Callbacks og checkpoints
from keras.callbacks import EarlyStopping, ModelCheckpoint


early_stopping = EarlyStopping(monitor='val_loss', patience=750)

checkpoint = ModelCheckpoint('checkpoint_model_advanced.h5', monitor = 'val_loss', 
                                       verbose = 1, save_best_only= True, mode= 'min')


# Trene modellen
np.random.seed(7)
dnn_keras_model.fit(X_train,y_train, epochs = 4000, batch_size=128, verbose=2, validation_data=(X_test, y_test),
                   callbacks=[checkpoint,early_stopping])



# validation_split=0.20
#validation_data=(X_test, y_test)


# # Resultater

# In[21]:


from sklearn import metrics
#finished_model = dnn_keras_model

finished_model = load_model('checkpoint_model_advanced.h5')

# #### Prediksjon på testsett
final_preds = finished_model.predict(X_test)

# 5000 epocs , b_size = 10, 24(24)[10](1)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_test, final_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_test, final_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_test, final_preds)))


# #### Prediksjon på treningdata
train_preds = finished_model.predict(X_train)

# 70% av data (benyttet til treningen)
print('Mean Absolute Error: \t\t\t', metrics.mean_absolute_error(y_train,train_preds))
print('Mean Squared Error: \t\t\t', metrics.mean_squared_error(y_train, train_preds))
print('Root Mean Squared Error: \t\t', np.sqrt(metrics.mean_squared_error(y_train, train_preds)))

exit(0)
# # Visualisere resultater

# #### Lager oversikt over testdataen

# In[27]:


predictions = list(final_preds)

predictions_list = []

for pred in predictions:
    predictions_list.append(pred[0])


# In[28]:


Oversikt = pd.DataFrame(data = {'real': y_test, 'predicitions': predictions_list})

Oversikt['differanse'] = Oversikt['real'] - Oversikt['predicitions'] 

Oversikt['abs_diff'] = Oversikt['differanse'].apply(abs)


# ### Scatterplot med fargekodede prediskjoner

# In[29]:


ax = Oversikt.plot.scatter(x='real',y='predicitions',
                   c='abs_diff',cmap='coolwarm', figsize = (20,10))

ax.set_xlabel("x label")

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = ax.get_figure()
#fig.savefig('keras_model.png')


# ### Feilfordelingen av prediksjoner

# In[30]:


ax = Oversikt['differanse'].hist(bins=60,figsize = (20,10))

plt.xlabel('Verdi av feilprediskjon')
plt.ylabel('Antall timer')
plt.title('Fordeling av feilprediskjoner')

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = ax.get_figure()
#fig.savefig('hist av abs.png')


# # Ekte plott av prediksjonene

# #### Forsøk på å hente ut et utdrag av dataen

# In[31]:


all_predictions = finished_model.predict(x[:,:])

predictions_all = list(all_predictions)

predictions_all_list = []

for pred in predictions_all:
    predictions_all_list.append(pred[0])
    
timeline = pd.DataFrame(data = {'real': y[:], 'predicitions': predictions_all_list})

# Adjust in order to plot 'dagens modell'
data_raw.dropna(inplace=True)
data_raw.reset_index(inplace=True)


# #### Tidsplot av data

# In[32]:


# Justere tidslinjen på plottet
start = 0
slutt = 20

ax = timeline.loc[start:slutt,'real'].plot(figsize=(20,8))
ax = timeline.loc[start:slutt,'predicitions'].plot(figsize=(20,8))
ax = data_raw.loc[start:slutt,'YVIK-YtreVikna1-Sum-produksjon'].plot(figsize=(20,8))

 
plt.xlabel('Tid (antall timer)')
plt.ylabel('Produksjon i Mega Watt (MW)')
plt.title('Utdrag fra tidsperioden')

plt.legend(loc='best')

#remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Saves figure
fig = ax.get_figure()
#fig.savefig('september10.png')

