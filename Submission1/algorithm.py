import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import pickle

df1 = pd.read_csv('encoded_train_data2.csv')
X = df1.iloc[:, 8:].values
y = df1.iloc[:, 7].values
pca = PCA(n_components=250)
pca.fit(X)
pca_scale = pca.transform(X)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(pca_scale, y)

with open('model_pickle', 'wb') as f:
    pickle.dump(regressor,f)
with open('pca_pickel','wb') as p:
    pickle.dump(pca,p)