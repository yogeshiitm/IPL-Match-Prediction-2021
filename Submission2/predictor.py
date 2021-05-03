import pickle
import pandas as pd

def predictRuns(testInput):
    with open('model_pickle','rb') as f:
        reg = pickle.load(f)
    with open('pca_pickel','rb') as p:
        pca = pickle.load(p)
    with open('sc_x_pickle','rb') as x:
        sc_X = pickle.load(x)
    with open('sc_y_pickle','rb') as Y:
        sc_y = pickle.load(Y)
    pd.options.mode.chained_assignment = None  # default='warn'

    #Test input to One_hot
    df2 = pd.read_csv(testInput)
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
        temp = df2['batsmen'][i].split(', ')
        if len(temp) > 1:
            lst = temp
        else:
            lst = df2['batsmen'][i].split(',')
        #print(lst)
        for batsman in lst:
            df2[batsman+'_batsman'] = 1
    for i in sorted(dataset.bowler.unique()):
        df2[i+'_bowler'] = 0
        
    for i in range(len(df2)):
        temp = df2['bowlers'][i].split(', ')
        if len(temp) > 1:
            lst = temp
        else:
            lst = df2['bowlers'][i].split(',')
        #print(lst)
        for bowler in lst:
            df2[bowler+'_bowler'] = 1
    #Prediction
    try:
        #y_pred = reg.predict(pca.transform(df2.iloc[:, 7:].values))
        y_pred = sc_y.inverse_transform(reg.predict(sc_X.transform(pca.transform(df2.iloc[:, 7:].values))))
    except:
        try:
            #y_pred = reg.predict(pca.transform(df2.iloc[:, 8:].values))
            y_pred = sc_y.inverse_transform(reg.predict(sc_X.transform(pca.transform(df2.iloc[:, 8:].values))))
        except:
            #y_pred = reg.predict(pca.transform(df2.iloc[:, 6:].values))
            y_pred = sc_y.inverse_transform(reg.predict(sc_X.transform(pca.transform(df2.iloc[:, 6:].values))))

    prediction = round(y_pred[0])
    print(prediction)

    return prediction