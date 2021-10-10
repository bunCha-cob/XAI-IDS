#%%
# preprocessing data 
import numpy as np
import pandas as pd

# import training and testing dataset
train_data = pd.read_csv('UNSW_NB15_training-set.csv')

test_data = pd.read_csv('UNSW_NB15_testing-set.csv')

shap_features = ['sttl', 'ct_state_ttl', 'dttl', 'swin', 'dload', 'service', 'ct_dst_sport_ltm', 
                 'ct_dst_src_ltm', 'ct_srv_dst', 'dwin', 'proto', 'dmean', 'stcpb', 'ct_srv_src', 
                 'smean', 'rate', 'tcprtt', 'is_sm_ips_ports', 'synack', 'ct_src_dport_ltm']

pfi_features = ['swin', 'dttl', 'sttl', 'ct_state_ttl', 'dload', 'service', 'smean', 'ct_srv_dst', 
                 'ct_dst_src_ltm', 'ct_dst_ltm', 'is_sm_ips_ports', 'ct_dst_sport_ltm', 
                 'ct_src_dport_ltm', 'synack', 'rate', 'proto', 'dwin', 'tcprtt', 'spkts', 'sbytes']

new = set(pfi_features) - set(shap_features)

combine_features = shap_features + list(new)

print(combine_features)
 
#%% 
# binary classifier
y_train = (train_data.label != 0).astype(int)
features = [c for c in train_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
X_train = train_data[combine_features]

y_test = (test_data.label != 0).astype(int)
X_test = test_data[combine_features]

#%%
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

list_binary = ['is_sm_ips_ports', 'is_ftp_login']

features = [c for c in combine_features if (c not in list_binary)]

X_train[features] = scaler.fit_transform(X_train[features].to_numpy())
X_test[features] = scaler.transform(X_test[features].to_numpy())

#%%
from tensorflow.keras.models import model_from_json
from time import time

# load json and create model for Exploits
json_file = open('model_Exploits_tvn.json', 'r')
loaded_model_exploits_json = json_file.read()
json_file.close()
loaded_model_exploits = model_from_json(loaded_model_exploits_json)
# load weights into new model 
loaded_model_exploits.load_weights("model_Exploits_tvn.h5")
print("Loaded model_exploits from disk")

# load json and create model for Generic
json_file = open('model_Generic_tvn.json', 'r')
loaded_model_xss_json = json_file.read()
json_file.close()
loaded_model_xss = model_from_json(loaded_model_xss_json)
# load weights into new model
loaded_model_xss.load_weights('model_Generic_tvn.h5')
print("Loaded model_xss from disk")

# load json and create model for Reconnaissance
json_file = open('model_Reconnaissance_tvn.json', 'r')
loaded_model_xss_json = json_file.read()
json_file.close()
loaded_model_xss = model_from_json(loaded_model_xss_json)
# load weights into new model
loaded_model_xss.load_weights('model_Reconnaissance_tvn.h5')
print("Loaded model_xss from disk")

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
y_pred_ddos = loaded_model_ddos.predict(X_test_cf)
y_pred_xss = loaded_model_xss.predict(X_test_cf)
stop = int(time() * 1000) 
print ("Processing time = ", stop - start)

#%%
y_pred_ddos = (y_pred_ddos > 0.5)
y_pred_xss = (y_pred_xss > 0.5)
y_pred_binary = (y_pred_binary > 0.5)

y_pred = [None] * len(y_pred_ddos)
for i in range(len(y_pred_ddos)):
    if (y_pred_ddos[i]==True) or (y_pred_xss[i]==True) or (y_pred_binary[i]==True):
        y_pred[i]=True
    else: 
        y_pred[i]=False
    
#%%
# make confusion matrix
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
