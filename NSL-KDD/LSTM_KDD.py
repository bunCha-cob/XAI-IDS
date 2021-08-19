# importing the libraries
import numpy as np
import pandas as pd

# pre-processing data 

# import training dataset
dataset = pd.read_csv('KDDTrain+.csv')
X_train = dataset.iloc[:,:-2].values
y_train = dataset.iloc[:,-2].values

# import testing dataset
training_set = pd.read_csv('KDDTest+.csv')
X_test = training_set.iloc[:,:-2].values
y_test = training_set.iloc[:,-2].values

# process y_train
for i in range(len(y_train)):
    if (y_train[i] == "normal"): 
        y_train[i] = 0 
    else: y_train[i] = 1
    
# process y_test
for i in range(len(y_test)):
    if (y_test[i] == "normal"): 
        y_test[i] = 0 
    else: y_test[i] = 1
    
# append X_train and X_test then one hot encoding column 1,2,3
X = np.concatenate((X_train, X_test))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# reassign X_train and X_test
X_train = X[:125972]
X_test = X[125972:]

# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = MinMaxScaler().fit(X_test)
X_test = scaler.transform(X_test)

# reshape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# reformat
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# build the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras import optimizers

# settings for the model 
opt = optimizers.Adam(lr = 0.001) # optimizer
num_nodes = 50  # number of hidden nodes
b_size = 32 # batch size
ep = 50 # epoch
dropout_rate = 0.1  

model = Sequential()
model.add(LSTM(num_nodes,input_dim=122, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(LSTM(num_nodes, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(LSTM(num_nodes, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(LSTM(num_nodes, return_sequences=False))  
model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# train the model
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = ep, batch_size = b_size, validation_split=0.1, verbose=1)
score = model.evaluate(X_test, y_test, batch_size = b_size)

# predict test set resutl and make confusion matrix
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_binary)
print(cm)
accuracy_score(y_test, y_pred_binary)

