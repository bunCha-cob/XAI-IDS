# importing the libraries
import numpy as np
import pandas as pd

# pre-processing data 
pfi_features = ['dst_host_srv_count', 'protocol_type', 'logged_in', 'dst_host_serror_rate', 
                'hot', 'service', 'srv_serror_rate', 'dst_host_same_src_port_rate', 
                'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_count', 'dst_host_same_srv_rate', 'same_srve_rate', 'Count', 'serror_rate', 
                'root_shell', 'is_guest_login', 'rerror_rate', 'num_shells']

# import training and testing dataset
train_data = pd.read_csv('KDDTrain+.csv')

test_data = pd.read_csv('KDDTest+.csv')

#%%
# binary classifier
y_train = (train_data.label != "normal").astype(int)
features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev")]
X_train = train_data[features]

y_test = (test_data.label != "normal").astype(int)
features = [c for c in test_data.columns if (c != "label") and (c != "diff_lev")]
X_test = test_data[features]
    
# label encoding data  

y_train = np.asarray(y_train).astype('int')
y_test = np.asarray(y_test).astype('int')

# apply a set of features
#X_train = X_train[shap_features]
#X_test = X_test[shap_features]
#X_train = X_train[pfi_features]
#X_test = X_test[pfi_features]

features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev")]

X_train = X_train[features]
X_test = X_test[features]
#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = pd.concat([X_train, X_test])

list_encode = ['protocol_type', 'service', 'flag']

for i in list_encode:
    X[i] = le.fit_transform(X[i])

X_train.iloc[:len(X_train),:] = X.iloc[:len(X_train),:]
X_test.iloc[:len(X_test),:] = X.iloc[len(X_train):,:]  

# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

list_binary = ['land', 'logged_in', 'root_shell', 'su_attempted', 'is_host_login', 'is_guest_login',
               'logged_in']

features = [c for c in features if (c != "label") and (c != "diff_lev") 
                    and (c not in list_binary)]

X_train[features] = scaler.fit_transform(X_train[features].to_numpy())
X_test[features] = scaler.transform(X_test[features].to_numpy())

# reshape
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#%%
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
ep = 10 # epoch
dropout_rate = 0.1  

model = Sequential()
model.add(SimpleRNN(num_nodes,input_dim=41, return_sequences=True))  
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
y_pred_binary = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred_binary).ravel()
accuracy = (TP+TN)/(TP+FP+TN+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = (2*precision*recall)/(precision + recall)

print("TN = ", TN," FP = ", FP, " FN = ", FN, " TP = ", TP, "\n")

print("Accuracy = ", accuracy, "\n")
print("Precision = ", precision, "\n")
print("Recall = ", recall, "\n")
print("F1 = ", F1, "\n")


