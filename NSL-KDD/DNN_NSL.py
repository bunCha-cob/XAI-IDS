# preprocessing data 
import numpy as np
import pandas as pd

# import training and testing dataset
train_data = pd.read_csv('KDDTrain+.csv')

test_data = pd.read_csv('KDDTest+.csv')

shap_features = ['dst_host_same_srv_rate', 'same_srve_rate', 'logged_in', 'rerror_rate', 
                 'dst_host_rerror_rate', 'Count', 'dst_host_srv_count', 'dst_host_srv_rerror_rate', 
                 'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                 'protocol_type', 'dst_host_count', 'srv_serror_rate', 'service', 'srv_count', 
                 'dst_host_diff_srv_rate', 'srv_diff_host_rate', 'srv_rerror_rate', 'diff_srv_rate']

pfi_features = ['dst_host_srv_count', 'protocol_type', 'logged_in', 'dst_host_serror_rate', 
                'hot', 'service', 'srv_serror_rate', 'dst_host_same_src_port_rate', 
                'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_count', 'dst_host_same_srv_rate', 'same_srve_rate', 'Count', 'serror_rate', 
                'root_shell', 'is_guest_login', 'rerror_rate', 'num_shells']

new = set(pfi_features) - set(shap_features)

combine_features = shap_features + list(new)

print(combine_features)

#%%
# binary classifier
y_train = (train_data.label != "normal").astype(int)
features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev")]
X_train = train_data[features]

y_test = (test_data.label != "normal").astype(int)
features = [c for c in test_data.columns if (c != "label") and (c != "diff_lev")]
X_test = test_data[features]

#%% label encoding data
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

features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev") 
                    and (c not in list_binary)]

X_train[features] = scaler.fit_transform(X_train[features].to_numpy())
X_test[features] = scaler.transform(X_test[features].to_numpy())

y_train = np.asarray(y_train).astype('int')
y_test = np.asarray(y_test).astype('int')

X_train = X_train[combine_features]
X_test = X_test[combine_features]

#%% processing
from tensorflow.keras.models import model_from_json
from time import time

# load json and create model for ddos
json_file = open('model_dos_tvn.json', 'r')
loaded_model_dos_json = json_file.read()
json_file.close()
loaded_model_dos = model_from_json(loaded_model_dos_json)
# load weights into new model 
loaded_model_dos.load_weights("model_dos_tvn.h5")
print("Loaded model_dos from disk")

# load json and create model for xss
json_file = open('model_probe_tvn.json', 'r')
loaded_model_probe_json = json_file.read()
json_file.close()
loaded_model_probe = model_from_json(loaded_model_probe_json)
# load weights into new model
loaded_model_probe.load_weights('model_probe_tvn.h5')
print("Loaded model_probe from disk")

# load json and create model for other attacks
json_file = open('model_binary.json', 'r')
loaded_model_binary_json = json_file.read()
json_file.close()
loaded_model_binary = model_from_json(loaded_model_binary_json)
# load weights into new model
loaded_model_binary.load_weights('model_binary.h5')
print("Loaded model_binary from disk")

# customized set of features for X_test
X_test_cf = X_test[combine_features]

# predict test set result 
# calculate processing time
start = int(time() * 1000) 
y_pred_binary = loaded_model_binary.predict(X_test_cf)
y_pred_dos = loaded_model_dos.predict(X_test_cf)
y_pred_probe = loaded_model_probe.predict(X_test_cf)
stop = int(time() * 1000) 
print ("Processing time = ", stop - start)

#%%
y_pred_dos = (y_pred_dos > 0.5)
y_pred_probe = (y_pred_probe > 0.5)
y_pred_binary = (y_pred_binary > 0.5)

y_pred = [None] * len(y_pred_dos)
for i in range(len(y_pred_dos)):
    if (y_pred_dos[i]==True) or (y_pred_probe[i]==True) or (y_pred_binary[i]==True):
        y_pred[i]=True
    else: 
        y_pred[i]=False

#%% make confusion matrix
from sklearn.metrics import confusion_matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
accuracy = (TP+TN)/(TP+FP+TN+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = (2*precision*recall)/(precision + recall)

print("TN = ", TN," FP = ", FP, " FN = ", FN, " TP = ", TP, "\n")

print("Accuracy = ", accuracy, "\n")
print("Precision = ", precision, "\n")
print("Recall = ", recall, "\n")
print("F1 = ", F1, "\n")

#%% local explanation
import shap  
import matplotlib.pyplot as plt

sample = X_test.sample(n = 5000)
 
explainer = shap.DeepExplainer(loaded_model_dos, np.array(sample))
shap_values = explainer.shap_values(np.array(sample))

# find the index of attack record
for i in range(len(y_pred_dos)):
    if (y_test[i]==True) and (y_pred_dos[i]==True):
        index = i
        break
row_to_show = X_test.iloc[index:index+1]

print ('row : ', row_to_show)
shap.initjs()
shap.force_plot(explainer.expected_value[0].numpy().tolist(), shap_values[0], row_to_show, show=False)
plt.savefig('local_explanation_dos.pdf', bbox_inches='tight')
plt.close() 
