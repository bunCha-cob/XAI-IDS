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

normal_traffic = 0
attack_traffic = 0
for i in range(len(train_data)):
    if train_data.iloc[i].label == 0:
        normal_traffic += 1
    else:
        attack_traffic += 1
print("Train data: ", len(train_data), " \n")
print("Normal: ", normal_traffic, "\n")
print("Attack: ", attack_traffic, "\n")
print("Ratio(A/N): ", attack_traffic/normal_traffic, "\n")

normal_traffic = 0
attack_traffic = 0
for i in range(len(test_data)):
    if test_data.iloc[i].label == 0:
        normal_traffic += 1
    else:
        attack_traffic += 1
print("Test data: ", len(test_data), " \n")
print("Normal: ", normal_traffic, "\n")
print("Attack: ", attack_traffic, "\n")
print("Ratio(A/N): ", attack_traffic/normal_traffic, "\n")

#%%
# one-vs-all classifiers
target_outcome = 'Generic'
y_train = (train_data.attack_cat == target_outcome).astype(int)
y_train = y_train.to_frame()
features = [c for c in train_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
#X_train = train_data[shap_features]
#X_train = train_data[pfi_features]
X_train = train_data[features]

y_test = (test_data.attack_cat == target_outcome).astype(int)
y_test = y_test.to_frame()
features = [c for c in test_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
#X_test = test_data[shap_features]
#X_test = test_data[pfi_features]
X_test = test_data[features]

#%% 
# binary classifier
y_train = (train_data.label != 0).astype(int)
features = [c for c in train_data.columns if (c != "label") and (c != "id") and (c != "attack_cat")]
#X_train = train_data[features]
#X_train = train_data[shap_features]
#X_train = train_data[pfi_features]
X_train = train_data[combine_features]

y_test = (test_data.label != 0).astype(int)
#X_test = test_data[features]
#X_test = test_data[shap_features]
#X_test = test_data[pfi_features]
X_test = test_data[combine_features]

#%% 
# target-vs-normal classifiers
target_outcome = 'Worms'

train_set = train_data[train_data['label'] == 0]
train_set = train_set.append(train_data[train_data['attack_cat'] == target_outcome])
y_train = (train_set.label != 0).astype(int)
X_train = train_set.drop(columns = ['label', 'id', 'attack_cat'])      
#X_train = train_set[shap_features]
#X_train = train_set[pfi_features]

test_set = test_data[test_data['label'] == 0]
test_set = test_set.append(test_data[test_data['attack_cat'] == target_outcome])
y_test = (test_set.label != 0).astype(int)
X_test = test_set.drop(columns = ['label', 'id', 'attack_cat'])   
#X_test = test_set[shap_features]   
#X_test = test_set[pfi_features]      
                           
#%%

X_test.to_csv(path_or_buf = 'X_test_before.csv', index = False)

#%%
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

list_binary = ['is_sm_ips_ports', 'is_ftp_login']

#features = [c for c in shap_features if (c not in list_binary)]
#features = [c for c in pfi_features if (c not in list_binary)]
features = [c for c in combine_features if (c not in list_binary)]
#features = [c for c in train_data.columns if (c != "label") and (c != "id") and 
            #(c != "attack_cat") and (c not in list_binary)]
#features = [c for c in train_data.columns if (c != "label") and (c != "id") and 
            #(c != "attack_cat") and (c not in list_binary) and (c != "service")]

X_train[features] = scaler.fit_transform(X_train[features].to_numpy())
X_test[features] = scaler.transform(X_test[features].to_numpy())

#%%

X_test.to_csv(path_or_buf = 'X_test_after.csv', index = False)

#%%
# build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import optimizers
from time import time

# settings for the model 
opt = optimizers.Adam(lr = 0.001) # optimizer
num_nodes = 50  # number of hidden nodes
b_size = 32 # batch size
ep = 10 # epoch
dropout_rate = 0.1  

start = int(time() * 1000) 

model = Sequential()
model.add(Dense(num_nodes,input_dim=len(X_train.columns), activation='relu'))  
model.add(Dropout(dropout_rate))
model.add(Dense(num_nodes, activation='relu'))  
model.add(Dropout(dropout_rate))
model.add(Dense(num_nodes, activation='relu'))  
model.add(Dropout(dropout_rate))
model.add(Dense(num_nodes, activation='relu'))  
model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# train the model
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = ep, batch_size = b_size, validation_split=0.1, verbose=1)
score = model.evaluate(X_test, y_test, batch_size = b_size)

stop = int(time() * 1000) 

from tensorflow.keras.models import model_from_json
# serialise model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialise weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print("Training time = ", stop - start)

#%%
from tensorflow.keras.models import model_from_json
from time import time

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model 
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# predict test set resutl and make confusion matrix
start = int(time() * 1000) 

y_pred = loaded_model.predict(X_test)

stop = int(time() * 1000) 

print ("Processing time = ", stop - start)

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
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import accuracy_score

def score(self, X, y, sample_weight=None):
      return accuracy_score(y, self.predict(X)>0.5, sample_weight=sample_weight)

perm = PermutationImportance(loaded_model, score, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

#%%
from matplotlib import pyplot as plt
from pdpbox import pdp

feature_name = "service"
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=loaded_model, dataset=X_test, model_features=X_test.columns, 
                            feature=feature_name)

# plot it
f, axes = pdp.pdp_plot(pdp_goals, feature_name, plot_lines=True, frac_to_plot=5000)
plt.show()
f.savefig("plots/"+ target_outcome + "/PDP_" + target_outcome + "_" + feature_name + ".pdf", bbox_inches='tight')
#%%
import shap
import pandas as pd 
import matplotlib.pyplot as plt

sample = X_test.sample(n = 5000)
 
explainer = shap.DeepExplainer(loaded_model, np.array(sample))
shap_values = explainer.shap_values(np.array(sample))

shap.initjs()
#force_plot = shap.force_plot(explainer.expected_value[0].numpy().tolist(), shap_values[0], sample)
shap.summary_plot(shap_values[0], sample, show=False)
plt.savefig('plots/'+ target_outcome +'/summary_plot_' + target_outcome + '.pdf', bbox_inches='tight')

