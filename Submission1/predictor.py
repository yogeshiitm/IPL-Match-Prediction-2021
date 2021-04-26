import pickle
import pandas as pd

with open('model_pickle','rb') as f:
    reg = pickle.load(f)
with open('pca_pickel','rb') as p:
    pca = pickle.load(p)
pd.options.mode.chained_assignment = None  # default='warn'

#Test input to One_hot
df2 = pd.read_csv('18_inn1.csv')
dataset = pd.read_csv('all_matches.csv')
for i in sorted(dataset.venue.unique()):
    df2[i] = 0
    
for i in range(len(df2)):
    df2[df2['venue'][i]][i] = 1
for i in sorted(dataset.innings.unique()):
    df2[i] = 0
    
for i in range(len(df2)):
    df2[df2['innings'][i]][i] = 1
for i in sorted(dataset.batting_team.unique()):
    df2[i+'_batting'] = 0
    
for i in range(len(df2)):
    df2[df2['batting_team'][i]+'_batting'][i] = 1
for i in sorted(dataset.bowling_team.unique()):
    df2[i+'_bowling'] = 0
    
for i in range(len(df2)):
    df2[df2['bowling_team'][i]+'_bowling'][i] = 1
for i in sorted(dataset.striker.unique()):
    df2[i+'_batsman'] = 0
    
for i in range(len(df2)):
    lst = df2['batsmen'][i].split(',')
    #print(lst)
    for batsman in lst:
        df2[batsman+'_batsman'] = 1
for i in sorted(dataset.bowler.unique()):
    df2[i+'_bowler'] = 0
    
for i in range(len(df2)):
    lst = df2['bowlers'][i].split(',')
    for bowler in lst:
        df2[bowler+'_bowler'] = 1
#Prediction
y_pred = reg.predict(pca.transform(df2.iloc[:, 7:].values))

print(y_pred)