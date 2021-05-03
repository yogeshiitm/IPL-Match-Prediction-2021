import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df1 = pd.read_csv('encoded_train_data2.csv')
X = df1.iloc[:, 8:].values
y = df1.iloc[:, 7].values
y = y.reshape(len(y),1)
pca = PCA(n_components=300)
pca.fit(X)
pca_scale = pca.transform(X)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(pca_scale)
y = sc_y.fit_transform(y)
regressor = SVR(kernel = 'poly')
regressor.fit(X, y)

with open('model_pickle', 'wb') as f:
    pickle.dump(regressor,f)
with open('pca_pickel','wb') as p:
    pickle.dump(pca,p)
with open('sc_x_pickle','wb') as x:
    pickle.dump(sc_X,x)
with open('sc_y_pickle','wb') as Y:
    pickle.dump(sc_y,Y)