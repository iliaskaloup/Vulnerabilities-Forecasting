#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Embedding, MaxPool1D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Bidirectional, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import glorot_uniform, RandomUniform, lecun_uniform, Constant
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPool1D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


# In[2]:


# load data of spesific software project
software = "Chrome" #AppleMac # Ubuntu # Explorer # Office # Chrome
softwarePrint = "google_chrome" # google_chrome # microsoft_internet_explorer # apple_mac_os_x # canonical_ubuntu_linux # microsoft_office

# model selection
# 'MLP','LSTM','TimeDistr','CNN','RF', 'GRU', 'BiLSTM'
regressor = 'LSTM'


dataset = pd.read_csv(software+".csv", delimiter=',')
#dataset = dataframe.values

# definition of steps ahead
versions_ahead = 24 # window horizon per months e.g., forecast 2 years (24 months) ahead
look_back = 24 # window to look back e.g., predictions based on 12 months. It a parameter actually.


# In[3]:


# split to train and test sets
'''train_size0 = int(len(data) * 0.80)
test_size = len(data) - train_size0
train0, test = data.iloc[0:train_size0,:], data.iloc[train_size0:len(data),:]
test = test.reset_index()

train_size = int(len(train0) * 0.80)
val_size = len(train0) - train_size
train, val = train0.iloc[0:train_size,:], train0.iloc[train_size:len(train0),:]'''

'''train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]'''
#test = test.reset_index()

train_size = len(dataset) - versions_ahead
train = dataset.iloc[0:train_size,:]
test = dataset.iloc[train_size-look_back:,:]

print(dataset)
print(len(dataset))
print(train)
print(len(train))
print(test)
print(len(test))


# In[4]:


def R_2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (
        1 - SS_res/(SS_tot + K.epsilon()) )


# In[5]:


def buildMLP(n_in, ahead):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=n_in))
    model.add(Dense(ahead))
    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    '''
    my_init = glorot_uniform(seed=seed)
    model = Sequential()
    model.add(Dense(units=200, kernel_initializer=my_init, input_dim=n_in))  
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))  
    model.add(Dense(units=100, kernel_initializer=my_init))  
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))
    model.add(Dense(units=50, kernel_initializer=my_init))  
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))  
    model.add(Dense(units=120, kernel_initializer=my_init))  
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))
    model.add(Dense(units = ahead, kernel_initializer=my_init))

    #sgd = optimizers.SGD(lr=0.001, decay=1e-6,momentum=0.90, nesterov=True )
    sgd = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False )
    #sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
    #sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #sgd = optimizers.Adagrad(lr=0.001, epsilon=None, decay=1e-6)
    model.compile(optimizer = sgd, loss = 'mse', metrics=[R_2])  # metrics=['mae']'''
    return model


# In[6]:


def buildLSTM(n_in, ahead):
    '''my_init = glorot_uniform(seed=seed)
    model = Sequential()
    model.add(LSTM(units=100, kernel_initializer=my_init, return_sequences=True, input_shape=(n_in, 1), stateful=False))  
    model.add(Activation('tanh')) 
    model.add(Dropout(0.2))  
    model.add(LSTM(units=50, kernel_initializer=my_init, return_sequences=True))  
    model.add(Activation('tanh')) 
    model.add(Dropout(0.2))
    model.add(LSTM(units=120, kernel_initializer=my_init))  
    model.add(Activation('tanh')) 
    model.add(Dropout(0.2))  
    model.add(Dense(units = ahead, kernel_initializer=my_init)) 
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6,momentum=0.90, nesterov=True )
    sgd = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False )
    #sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
    #sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #sgd = optimizers.Adagrad(lr=0.001, epsilon=None, decay=1e-6)
    model.compile(optimizer = sgd, loss = 'mse', metrics=[R_2])
    '''
    model = Sequential()
    model.add(LSTM(500, activation='tanh', input_shape=(n_in, 1)))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(ahead))
    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    return model


# In[7]:


def buildGRU(n_in, ahead):
    model = Sequential()
    model.add(GRU(500, activation='tanh', input_shape=(n_in, 1)))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(ahead))
    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    return model


# In[8]:


def buildBiLSTM(n_in, ahead):
    model = Sequential()
    model.add(Bidirectional(LSTM(500, activation='tanh', input_shape=(n_in, 1))))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(ahead))
    model.compile(loss='mae', optimizer='adam', metrics=[R_2])
    return model


# In[9]:


def buildCNN(n_in, ahead):
    '''model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_in, 1)))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))  
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6,momentum=0.90, nesterov=True )
    sgd = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False )
    #sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
    #sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #sgd = optimizers.Adagrad(lr=0.001, epsilon=None, decay=1e-6)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics=[R_2])'''
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_in, 1)))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(ahead))
    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    return model


# In[10]:


def buildCNNLSTM(n_in, ahead):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None,n_in,1))))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(ahead))

    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    return model


# In[11]:


'''def ConvLSTM(n_in, ahead):
    model = Sequential()
    model.add(ConvLSTM2D(filters=256, kernel_size=(1,3), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(ahead))
    model.compile(loss='mse', optimizer='adam', metrics=[R_2])
    return model'''


# In[12]:


# sliding window
'''def create_dataset(dataset, look_back, ahead):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-ahead+1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + ahead - 1, 0])
    return np.array(dataX), np.array(dataY)'''

def create_dataset(dataset, look_back, ahead):
    dataX, dataY = [], []
    for i in range(0, len(dataset)-look_back-ahead+1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back : i+look_back+ahead, 0])
    return np.array(dataX), np.array(dataY)

def makeSequence(x):
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x


# In[13]:


# preprocess
scaler = MinMaxScaler(feature_range = (0, 1))
training_processed = np.reshape(train.iloc[:,1].values,(len(train),1))
training_scaled = scaler.fit_transform(training_processed)
testing_processed = np.reshape(test.iloc[:,1].values,(len(test),1))
testing_scaled = scaler.transform(testing_processed)


# In[14]:


trainX, trainY = create_dataset(training_scaled, look_back, versions_ahead) 
testX, testY = create_dataset(testing_scaled, look_back, versions_ahead)

if regressor != "MLP": 
    trainX = makeSequence(trainX) 
    testX = makeSequence(testX)

#if regressor == "buildCNNLSTM" or regressor == "buildConvLSTM":
    


# In[15]:


# training
if regressor == 'MLP':
    model = buildMLP(look_back, versions_ahead)
elif regressor == 'LSTM':
    model = buildLSTM(look_back, versions_ahead)
elif regressor == 'GRU':
    model = buildLSTM(look_back, versions_ahead)
elif regressor == 'CNN':
    model = buildCNN(look_back, versions_ahead)
elif regressor == 'BiLSTM':
    model = buildBiLSTM(look_back, versions_ahead)
    
#print("model summary\m",model.summary())
csv_logger = CSVLogger('log.csv', append=True, separator=',')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
model.fit(trainX, trainY, epochs = 50, validation_data=(testX, testY), batch_size = 64, shuffle=False, callbacks=[csv_logger,es,mc], verbose=1)
#load best model
model = load_model('best_model.h5', custom_objects={"R_2": R_2})


# In[16]:


# goodness of fit with R squared. Compute R2 on train data.
train_predictions_scaled = model.predict(trainX)
train_predictions = scaler.inverse_transform(train_predictions_scaled)
train_labels = scaler.inverse_transform(trainY)

r2_fit = r2_score(train_labels, train_predictions, multioutput='raw_values')
print("R2_fit:",r2_fit)
r2_fit_mean = np.mean(r2_fit)
print('Mean R2_fit: %.3f' % (r2_fit_mean))


# In[17]:


mse_fit = mean_squared_error(train_labels, train_predictions) # , multioutput='raw_values' if want mse per step
rmse_fit = sqrt(mse_fit) 
mae_fit = mean_absolute_error(train_labels, train_predictions)
print('RMSE-fit: %.3f' % (rmse_fit))
print('MAE-fit: %.3f' % (mae_fit)) 


# In[18]:


#model = RandomForestRegressor(n_estimators=100, bootstrap = True, max_features = 'sqrt')
#model = SVR(C=1.0, epsilon=0.2)
#model.fit(trainX, trainY)

# predictions
predictions_scaled = model.predict(testX)
#predictions_scaled = np.reshape(predictions_scaled,(predictions_scaled.shape[0],1)) # for ml
predictions = scaler.inverse_transform(predictions_scaled)
test_labels = testY.reshape(-1,1) 
test_labels = scaler.inverse_transform(test_labels)                               

predictions = np.reshape(predictions, (predictions.shape[1],1))


# In[19]:


# save test labels and predictions
estim = pd.DataFrame(predictions)
estim.to_csv('predictions_'+regressor+'_'+software+'.csv', index=None, header=False)
trues = pd.DataFrame(test_labels)
trues.to_csv('actuals_'+regressor+'_'+software+'.csv', index=None, header=False)
# save real errors
errors = abs(estim - trues)
errors.to_csv('errors'+regressor+'_'+software+'.csv', index=None, header=False)


# In[20]:


print(errors)


# In[21]:


# evaluation scores                    
mse = mean_squared_error(test_labels, predictions) # , multioutput='raw_values' if want mse per step
rmse = sqrt(mse) 
mae = mean_absolute_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

#print('MSE: %.3f' % (mse)) 
print('RMSE: %.3f' % (rmse))
print('MAE: %.3f' % (mae)) 
#print('R2: %.3f' % (r2))


# In[22]:


# save scores
scores = []
scores.append(round(rmse,2))
scores.append(round(mae,2)) 

scores.append(round(r2_fit_mean,2)) 

scores = pd.DataFrame(scores) 
scores = scores.rename(index={0: 'RMSE', 1: 'MAE', 2: 'R2-fit'})
#print(scores)
scores.to_csv(software + '' + regressor + '' + str(versions_ahead) + '_' + str(look_back) + '.csv', header=False)    


# In[23]:


# plots
with plt.style.context('bmh'):
    fig = plt.figure(figsize=(16, 6))

    df1 = pd.DataFrame(test_labels)
    df2 = pd.DataFrame(predictions)

    date = dataset.iloc[train_size:,:]
    date = date['Datetime']

    plt.plot(date.astype("datetime64"), df1[0], label='real', 
             linewidth=2)
    plt.plot(date.astype("datetime64"), df2[0], 
             label='forecast', linewidth=2)

    plt.title('Vulnerabilities Forecast - Test data only')
    plt.xlabel('Date')
    plt.ylabel('Vulnerabilities')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[24]:


# plots
with plt.style.context('bmh'):
    fig = plt.figure(figsize=(16, 6))
    df1 = dataset['Vulnerabilities']
    df2 = pd.DataFrame(np.empty((train_size,1)),dtype=object)
    df2.iloc[0:train_size,0] = None
    df3 = pd.DataFrame()
    df3 = pd.concat([df2, pd.DataFrame(predictions)], ignore_index = True, axis = 0)

    date = dataset['Datetime']

    plt.plot(date.astype("datetime64"), df1, label='real', 
             linewidth=2)
    plt.plot(date.astype("datetime64"), df3, 
             label='forecast', linewidth=2)

    plt.title('Vulnerabilities Forecast - Full Dataset')
    plt.xlabel('Date')
    plt.ylabel('Vulnerabilities')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[25]:


# plots
with plt.style.context('bmh'):
    fig = plt.figure(figsize=(16, 6))
    df1 = dataset['Vulnerabilities']
    df1 = df1[-48:]
    df2 = pd.DataFrame(np.empty((24,1)),dtype=object)
    df2.iloc[0:24,0] = None
    df3 = pd.DataFrame()
    df3 = pd.concat([df2, pd.DataFrame(predictions)], ignore_index = True, axis = 0)

    date = dataset['Datetime']
    date = date[-48:]

    plt.plot(date.astype("datetime64"), df1, label='vulnerabilities', 
             linewidth=2)
    plt.plot(date.astype("datetime64"), df3, 
             label= regressor+' forecast', linewidth=2)

    plt.title(softwarePrint +' - Vulnerabilities Forecast')
    plt.xlabel('Date')
    plt.ylabel('Vulnerabilities')

    plt.legend() # loc='upper right'
    plt.tight_layout()
    plt.show()


# In[26]:


print('R2_fit: %.3f' % (r2_fit_mean))
print('MAE-fit: %.3f' % (mae_fit))
print('RMSE-fit: %.3f' % (rmse_fit))
print("\n")
print('MAE: %.3f' % (mae))
print('RMSE: %.3f' % (rmse))
#print('R2: %.3f' % (r2))


# In[27]:


#dataset['Vulnerabilities'].describe()


# In[28]:


#boxplot = dataset.boxplot(column=['Vulnerabilities'])


# In[29]:


#histogram = dataset.hist(column=['Vulnerabilities'], bins=10)

