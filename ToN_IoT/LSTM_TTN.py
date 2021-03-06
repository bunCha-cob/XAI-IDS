#%%
# preprocessing data
import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Train_Test_Network.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:,-2].values

# label encoding data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
list_encode = [0, 1, 3, 5, 6, 10, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41, 42]

for i in range(len(list_encode)):
    X[:, list_encode[i]] = le.fit_transform(X[:, list_encode[i]])
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
scaler = Normalizer().fit(X_test)
X_test = scaler.transform(X_test)
val_X = X_test
val_y = y_test

# reshape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# reformat
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
#%%
# build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras import optimizers

# settings for the model 
opt = optimizers.Adam(lr = 0.001) # optimizer
num_nodes = 50  # number of hidden nodes
b_size = 32 # batch size
ep = 50 # epoch
dropout_rate = 0.1  

model = Sequential()
model.add(LSTM(num_nodes,input_dim=43, return_sequences=True))  
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

from tensorflow.keras.models import model_from_json
# serialise model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialise weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#%%
from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model 
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# predict test set resutl and make confusion matrix
y_pred = loaded_model.predict(X_test)
y_pred_binary = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

def get_confusion_matrix_values(z_true, z_pred):
    cm = confusion_matrix(z_true, z_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred_binary)
detection_rate = TP/(TP+FN)
false_alarm_rate = FP/(TN+FP)
accuracy = (TP+TN)/(TP+FP+TN+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = (2*precision*recall)/(precision + recall)

print("\n Detection rate = ", detection_rate, "\n")
print("False alarm rate = ", false_alarm_rate, "\n")
print("Accuracy = ", accuracy, "\n")
print("Precision = ", precision, "\n")
print("Recall = ", recall, "\n")
print("F1 = ", F1, "\n")

#%%
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.svm import SVC

svc = SVC().fit(val_X, val_y)
perm = PermutationImportance(svc).fit(val_X, val_y)
eli5.show_weights(perm)

#%%
import shap
explainer = shap.DeepExplainer(loaded_model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
