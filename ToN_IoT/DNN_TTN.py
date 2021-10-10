#%%
# preprocessing data
import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Train_Test_Network.csv')

shap_features = ['proto', 'conn_state', 'dns_AA', 'dns_RD', 'dns_qtype', 'service', 'dns_rejected', 'dns_query', 
                 'dns_rcode', 'dns_RA', 'dns_qclass', 'http_version', 'weird_notice', 'http_user_agent', 
                 'weird_addl', 'ssl_resumed', 'ssl_established', 'http_status_code', 'http_uri', 
                 'http_resp_mime_types']

pfi_features = ['proto', 'conn_state', 'dns_AA', 'service', 'dns_RD', 'dns_query', 'dns_qtype', 
                'dns_rejected', 'dns_rcode', 'dns_RA', 'dns_qclass', 'weird_notice', 'ssl_resumed',
                 'http_version', 'ssl_established', 'weird_addl', 'weird_name', 'ssl_version', 
                 'http_uri', 'src_bytes']

new = set(pfi_features) - set(shap_features)

combine_features = shap_features + list(new)

print(combine_features)

#%% 
# binary classifier
y = (dataset.label != 0).astype(int)
y = y.to_frame()

features = [c for c in dataset.columns if (c != "label") and (c != "type")]
X = dataset[features]

# delete irrelevant features
list_delete = ['ts','src_ip', 'src_port', 'dst_ip', 'dst_port']
X = X.drop(columns = list_delete)

#%%
# label encoding data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

list_encode = ['proto', 'service', 'conn_state', 'dns_query', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 
               'ssl_version',  'ssl_cipher', 'ssl_resumed', 'ssl_established', 'ssl_subject', 
               'ssl_issuer', 'http_method', 'http_uri', 'http_version', 'http_user_agent' , 
               'http_orig_mime_types', 'http_resp_mime_types', 'weird_name', 'weird_addl',
               'weird_notice', 'http_trans_depth']

for i in list_encode:
    X[i] = le.fit_transform(X[i])
    
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#%%
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train.iloc[:,:] = scaler.fit_transform(X_train.to_numpy())
X_test.iloc[:,:] = scaler.transform(X_test.to_numpy())

#%%
from tensorflow.keras.models import model_from_json
from time import time

# load json and create model for ddos
json_file = open('model_ddos_tvn.json', 'r')
loaded_model_ddos_json = json_file.read()
json_file.close()
loaded_model_ddos = model_from_json(loaded_model_ddos_json)
# load weights into new model 
loaded_model_ddos.load_weights("model_ddos_tvn.h5")
print("Loaded model_ddos from disk")

# load json and create model for xss
json_file = open('model_xss_tvn.json', 'r')
loaded_model_xss_json = json_file.read()
json_file.close()
loaded_model_xss = model_from_json(loaded_model_xss_json)
# load weights into new model
loaded_model_xss.load_weights('model_xss_tvn.h5')
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

#%%
import shap  # package used to calculate Shap values
import numpy as np

sample = X_test.sample(n = 5000)
explainer = shap.DeepExplainer(loaded_model_ddos, np.array(sample))
shap_values = explainer.shap_values(np.array(sample))

#%%

index = 925
row_to_show = X_test.iloc[index:index+1]

shap_values = explainer.shap_values(np.array(row_to_show))
shap.initjs()
shap.force_plot(explainer.expected_value[0].numpy().tolist(), shap_values[0], row_to_show) 


