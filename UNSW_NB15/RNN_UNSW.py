# preprocessing data 
import numpy as np
import pandas as pd

# import training and testing dataset
train_data = pd.read_csv('UNSW_NB15_training-set.csv')
print(train_data.columns)

test_data = pd.read_csv('UNSW_NB15_testing-set.csv')
print(test_data.columns)

#%%
target_outcome = 'DoS'
y_train = (train_data.attack_cat == target_outcome).astype(int)
y_train = y_train.to_frame()
features = [c for c in train_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
X_train = train_data[features]

y_test = (test_data.attack_cat == target_outcome).astype(int)
y_test = y_test.to_frame()
features = [c for c in test_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
X_test = test_data[features]

#%%
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

list_binary = ['is_sm_ips_ports', 'is_ftp_login']
features = [c for c in train_data.columns if (c != "label") and (c != "id") and (c != "attack_cat") 
                    and (c not in list_binary)]

X_train[features] = scaler.fit_transform(X_train[features].to_numpy())
X_test[features] = scaler.transform(X_test[features].to_numpy())

#%%
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#%%
# build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.layers import SimpleRNN

# settings for the model 
opt = optimizers.Adam(lr = 0.001) # optimizer
num_nodes = 50  # number of hidden nodes
b_size = 32 # batch size
ep = 10 # epoch
dropout_rate = 0.1  

model = Sequential()
model.add(SimpleRNN(num_nodes,input_dim=42, return_sequences=True))  
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

from tensorflow.keras.models import model_from_json
# serialise model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialise weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")

#%%
from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model 
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

# predict test set resutl and make confusion matrix
y_pred = loaded_model.predict(X_test)
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

#%%
import shap
import pandas as pd 
import matplotlib.pyplot as plt

sample = pd.DataFrame(X_test).sample(n = 5000)
 
explainer = shap.DeepExplainer(loaded_model, np.array(sample))
shap_values = explainer.shap_values(np.array(sample))

shap.initjs()
#force_plot = shap.force_plot(explainer.expected_value[0].numpy().tolist(), shap_values[0], sample)
shap.summary_plot(shap_values[0], sample, show=False)
plt.savefig('plots/'+ target_outcome +'/summary_plot_' + target_outcome + '.pdf', bbox_inches='tight')
