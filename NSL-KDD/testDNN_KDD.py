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
# count number of normal and attack traffic
normal_traffic = 0
attack_traffic = 0
for i in range(len(train_data)):
    if train_data.iloc[i].label == "normal":
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
    if test_data.iloc[i].label == "normal":
        normal_traffic += 1
    else:
        attack_traffic += 1
print("Test data: ", len(test_data), " \n")
print("Normal: ", normal_traffic, "\n")
print("Attack: ", attack_traffic, "\n")
print("Ratio(A/N): ", attack_traffic/normal_traffic, "\n")

#%%
# one-vs-all classifiers
attack_class = {'normal': 'normal',
              'probe' : ['ipsweep, nmap, portsweep, satan, saint, mscan'],
              'dos': ['back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, mailbomb'],
              'u2r': ['buffer_overflow, loadmodule, perl, rootkit, xterm, ps, sqlattack'],
              'r2l': ['ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, snmpgetattack, named, xlock, xsnoop, sendmail, httptunnel, worm, snmpguess']}
target_outcome = 'u2r'

y_train = pd.DataFrame(columns=['label']) 
y_test = pd.DataFrame(columns=['label'])

for i in range(train_data.shape[0]):
    y_train = y_train.append({'label' : int(train_data.label[i] in attack_class[target_outcome][0])}, ignore_index = True)
features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev")]
X_train = train_data[features]

for i in range(test_data.shape[0]):
    y_test = y_test.append({'label' : int(test_data.label[i] in attack_class[target_outcome][0])}, ignore_index = True)
features = [c for c in test_data.columns if (c != "label") and (c != "diff_lev")]
X_test = test_data[features]

#%%
# binary classifier
y_train = (train_data.label != "normal").astype(int)
features = [c for c in train_data.columns if (c != "label") and (c != "diff_lev")]
X_train = train_data[features]

y_test = (test_data.label != "normal").astype(int)
features = [c for c in test_data.columns if (c != "label") and (c != "diff_lev")]
X_test = test_data[features]

#%%
# target-vs-normal classifiers
attack_class = {'normal': 'normal',
              'probe' : ['ipsweep, nmap, portsweep, satan, saint, mscan'],
              'dos': ['back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, mailbomb'],
              'u2r': ['buffer_overflow, loadmodule, perl, rootkit, xterm, ps, sqlattack'],
              'r2l': ['ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, snmpgetattack, named, xlock, xsnoop, sendmail, httptunnel, worm, snmpguess']}
target_outcome = 'r2l'
target = attack_class[target_outcome][0].split(", ")

train_set = train_data[train_data['label'] == 'normal']
for i in range(len(target)):
    train_set = train_set.append(train_data[train_data['label'] == target[i]])
y_train = (train_set.label != "normal").astype(int)
X_train = train_set.drop(columns = ['label', 'diff_lev'])
    
test_set = test_data[test_data['label'] == 'normal']
for i in range(len(target)):
    test_set = test_set.append(test_data[test_data['label'] == target[i]])
y_test = (test_set.label != "normal").astype(int)
X_test = test_set.drop(columns = ['label', 'diff_lev'])

#%%
# label encoding data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = pd.concat([X_train, X_test])

list_encode = ['protocol_type', 'service', 'flag']

for i in list_encode:
    X[i] = le.fit_transform(X[i])
    
X_train.iloc[:len(X_train),:] = X.iloc[:len(X_train),:]
X_test.iloc[:len(X_test),:] = X.iloc[len(X_train):,:]    

#%%

X_test.to_csv(path_or_buf = 'X_test_before.csv', index = False)

#%%
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

#%%

X_test.to_csv(path_or_buf = 'X_test_after.csv', index = False)

#%%
# apply a set of features
#X_train = X_train[shap_features]
#X_test = X_test[shap_features]
#X_train = X_train[pfi_features]
#X_test = X_test[pfi_features]
X_train = X_train[combine_features]
X_test = X_test[combine_features]

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

feature_name = "dst_host_srv_count"
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=loaded_model, dataset=X_test, model_features=X_test.columns, 
                            feature=feature_name)

# plot it
f, axes = pdp.pdp_plot(pdp_goals, feature_name, plot_lines=True, frac_to_plot=5000)
plt.show()
f.savefig("PDP_" + target_outcome + "_" + feature_name + ".pdf", bbox_inches='tight')
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
plt.savefig('summary_plot_' + target_outcome + '.pdf', bbox_inches='tight')
