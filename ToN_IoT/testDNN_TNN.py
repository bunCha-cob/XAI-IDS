#%%
# preprocessing data
import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Train_Test_Network.csv')
print(dataset.columns)

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
normal_traffic = 0
attack_traffic = 0
for i in range(len(dataset)):
    if dataset.iloc[i].label == 0:
        normal_traffic += 1
    else:
        attack_traffic += 1
print("Train data: ", len(dataset), " \n")
print("Normal: ", normal_traffic, "\n")
print("Attack: ", attack_traffic, "\n")
print("Ratio(A/N): ", attack_traffic/normal_traffic, "\n")

#%%
# one-vs-all classifiers
target_outcome = 'ddos'
y = (dataset.type == target_outcome).astype(int)
y = y.to_frame()
features = [c for c in dataset.columns if (c != "label") and (c != "type")]
X = dataset[features]

# delete irrelevant features
list_delete = ['ts','src_ip', 'src_port', 'dst_ip', 'dst_port']
X = X.drop(columns = list_delete)

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
# target-vs-normal classifiers
target_outcome = 'xss'

data = dataset[dataset['type'] == 'normal']
data = data.append(dataset[dataset['type'] == target_outcome])
y = (data.type != 'normal').astype(int)

X = data.drop(columns = ['label', 'type'])    

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

X.to_csv(path_or_buf = 'conn_state.csv', index = False)

#%%
# apply a set of features      
#X = X[pfi_features]
#X = X[shap_features]
X = X[combine_features]

#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#%%

X_test.to_csv(path_or_buf = 'X_test_before.csv', index = False)

#%%
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train.iloc[:,:] = scaler.fit_transform(X_train.to_numpy())
X_test.iloc[:,:] = scaler.transform(X_test.to_numpy())

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

feature_name = "dns_RD"
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=loaded_model, dataset=X_test, model_features=X_test.columns, 
                            feature=feature_name)

# plot it
f, axes = pdp.pdp_plot(pdp_goals, feature_name, plot_lines=True, frac_to_plot=5000)
plt.show()
f.savefig("plots/"+ target_outcome + "/PDP_" + target_outcome + "_" + feature_name + ".pdf", bbox_inches='tight')
#%%
# summary_plot
import shap
import pandas as pd 
import matplotlib.pyplot as plt

sample = X_test.sample(n = 5000)
 
explainer = shap.DeepExplainer(loaded_model, np.array(sample))
shap_values = explainer.shap_values(np.array(sample))

shap.initjs()
#force_plot = shap.force_plot(explainer.expected_value[0].numpy().tolist(), shap_values[0], sample)
shap.summary_plot(shap_values[0], sample, show=False)
#plt.savefig('plots/'+ target_outcome +'/summary_plot_' + target_outcome + '.pdf', bbox_inches='tight')
plt.savefig('summary_plot_binary.pdf', bbox_inches='tight')
plt.close()

#%%
# dependence_plot
shap.dependence_plot('conn_state', shap_values[0], sample, interaction_index = 'dns_RD', show=False)
plt.savefig('plots/'+ target_outcome +'/dependence_plot_' + target_outcome + '.pdf', bbox_inches='tight')
plt.close()

#%%
# dependence_plot
shap.dependence_plot('conn_state', shap_values[0], sample, interaction_index = 'proto', show=False)
plt.savefig('plots/'+ target_outcome +'/dependence_plot_' + target_outcome + '_connstate_proto.pdf', bbox_inches='tight')
plt.close()


