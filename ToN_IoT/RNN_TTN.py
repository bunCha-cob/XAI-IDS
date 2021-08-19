#%% preprocessing data
import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Train_Test_Network.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:,-2].values
#%%

# label encoding data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
list_encode = [0, 1, 3, 5, 6, 10, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41, 42]

for i in range(len(list_encode)):
    X[:, list_encode[i]] = le.fit_transform(X[:, list_encode[i]])
    
#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
scaler = Normalizer().fit(X_test)
X_test = scaler.transform(X_test)

# reshape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# reformat
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# build the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, SimpleRNN
from keras import optimizers

# settings for the model 
opt = optimizers.Adam(lr = 0.001) # optimizer
num_nodes = 50  # number of hidden nodes
b_size = 32 # batch size
ep = 50 # epoch
dropout_rate = 0.1  

model = Sequential()
model.add(SimpleRNN(num_nodes,input_dim=43, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(SimpleRNN(num_nodes, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(SimpleRNN(num_nodes, return_sequences=True))  
model.add(Dropout(dropout_rate))
model.add(SimpleRNN(num_nodes, return_sequences=False))  
model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# train the model
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = ep, batch_size = b_size, validation_split=0.1, verbose=1)
score = model.evaluate(X_test, y_test, batch_size = b_size)

# predict test set resutl and make confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)